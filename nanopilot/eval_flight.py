#!/usr/bin/env python3
"""
Closed-loop autopilot evaluation for Nanopilot.

Loads a TorchPilot checkpoint, lets it fly the drone in the running sim for
one or more episodes, and scores each episode from Gazebo ground-truth pose:
survival time, along-track progress, cross-track error, gates passed.
Writes machine-readable JSON so runs/models can be compared over time.

Requires BetaFlight SITL + Gazebo to already be running, for example via:
    GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh --no-manage --airframe=80mm

Or use the one-command wrapper:
    ./scripts/eval_flight.sh --airframe=80mm --world=slalom_gates_80mm \
        --model=models/bc_v4.pth
"""

import argparse
import json
import math
import os
import threading
import time
import xml.etree.ElementTree as ET

import donkeycar as dk
import numpy as np

from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from torch_pilot import TorchPilot

# Matches the hardcoded teleport target in DroneGymEnv._do_reset.
SPAWN = (0.0, 0.0, 0.5)

# outputs[] layout from DroneGymEnv.run_threaded with record_position,
# record_attitude, record_velocity, record_imu all True:
# [0]=image, [1:7]=RC PWMs, [7:10]=pos, [10:13]=att(deg), [13:16]=vel, [16:22]=imu
# DroneGymEnv maps the altitude command to a PWM offset as
# sign(a)*|a|**DRONE_THROTTLE_SCALE; the runtime altitude filters below work in
# that PWM space, because that is where a bias is physically a thrust bias.
THR_SCALE = 0.2

POS = slice(7, 10)
ATT = slice(10, 13)
VEL = slice(13, 16)
IMU = slice(16, 22)


# ---------------------------------------------------------------------------
# World parsing


def world_sdf_path(world):
    return os.path.join(os.path.dirname(__file__), "..", "worlds", f"{world}.sdf")


def parse_world(world):
    """Extract gate poses and landing-pad position from the world SDF.

    Returns (gates, pad_xy) where gates is a list of dicts with x/y/z/yaw
    sorted by model name, and pad_xy is (x, y) or None.
    """
    path = world_sdf_path(world)
    gates = []
    pad_xy = None
    if not os.path.exists(path):
        return gates, pad_xy
    root = ET.parse(path).getroot()
    for model in root.iter("model"):
        name = model.get("name", "")
        pose_el = model.find("pose")
        if pose_el is None or not pose_el.text:
            continue
        vals = [float(v) for v in pose_el.text.split()]
        if len(vals) != 6:
            continue
        x, y, z, _r, _p, yaw = vals
        if name.startswith("gate"):
            gates.append({"name": name, "x": x, "y": y, "z": z, "yaw": yaw})
        elif name == "landing_pad":
            pad_xy = (x, y)
    gates.sort(key=lambda g: g["name"])
    return gates, pad_xy


def detect_task(world, gates, pad_xy):
    if gates:
        return "gates"
    if pad_xy is not None:
        return "line"
    return "generic"


# ---------------------------------------------------------------------------
# Scoring helpers


class GateTracker:
    """Counts gate passes from consecutive positions.

    A pass is a crossing of the gate's local x=0 plane while inside the
    opening (posts at local y = +/-1m, 3m tall). Each gate counts once.
    """

    def __init__(self, gates, half_width=0.9, z_min=0.2, z_max=3.0):
        self.gates = gates
        self.half_width = half_width
        self.z_min = z_min
        self.z_max = z_max
        self.passed = {}  # name -> episode time of pass
        self._prev_local = [None] * len(gates)

    def update(self, t, pos):
        for i, g in enumerate(self.gates):
            c, s = math.cos(g["yaw"]), math.sin(g["yaw"])
            dx, dy = pos[0] - g["x"], pos[1] - g["y"]
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            prev = self._prev_local[i]
            self._prev_local[i] = (lx, ly, pos[2])
            if g["name"] in self.passed or prev is None:
                continue
            plx, ply, pz = prev
            if plx == lx or (plx < 0) == (lx < 0):
                continue
            # Interpolate lateral offset and height at the crossing plane.
            f = plx / (plx - lx)
            ly_c = ply + f * (ly - ply)
            z_c = pz + f * (pos[2] - pz)
            if abs(ly_c) <= self.half_width and self.z_min <= z_c <= self.z_max:
                self.passed[g["name"]] = round(t, 2)


def episode_metrics(track, start_xy, pad_xy):
    """Compute path metrics from a list of (t, x, y, z) samples."""
    xs = np.array([p[1] for p in track])
    ys = np.array([p[2] for p in track])
    zs = np.array([p[3] for p in track])
    dx, dy = xs - start_xy[0], ys - start_xy[1]
    steps = np.hypot(np.diff(xs), np.diff(ys))
    metrics = {
        "path_length_m": round(float(steps.sum()), 3),
        "max_displacement_m": round(float(np.hypot(dx, dy).max()), 3),
        "alt_mean_m": round(float(zs.mean()), 3),
        "alt_std_m": round(float(zs.std()), 3),
        "alt_min_m": round(float(zs.min()), 3),
        "alt_max_m": round(float(zs.max()), 3),
    }
    if pad_xy is not None:
        ax, ay = pad_xy[0] - start_xy[0], pad_xy[1] - start_xy[1]
        norm = math.hypot(ax, ay) or 1.0
        ax, ay = ax / norm, ay / norm
        along = dx * ax + dy * ay
        cross = -dx * ay + dy * ax
        metrics.update(
            {
                "along_track_max_m": round(float(along.max()), 3),
                "cross_track_mean_m": round(float(np.abs(cross).mean()), 3),
                "cross_track_max_m": round(float(np.abs(cross).max()), 3),
                "final_dist_to_pad_m": round(
                    float(math.hypot(xs[-1] - pad_xy[0], ys[-1] - pad_xy[1])), 3
                ),
            }
        )
    return metrics


# ---------------------------------------------------------------------------
# Episode control


def request_reset(env, timeout_s):
    """Teleport the drone to spawn and wait until the pose reflects it.

    The gz set_pose service call can time out when the sim is busy, so
    re-issue the reset request every couple of seconds until the pose sticks.
    """
    deadline = time.time() + timeout_s
    next_request = 0.0
    while time.time() < deadline:
        if time.time() >= next_request:
            env.run_threaded(0.0, 0.0, 0.0, 0.0, web_reset=True)
            next_request = time.time() + 2.0
        outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0)
        pos = outputs[POS]
        if math.hypot(pos[0] - SPAWN[0], pos[1] - SPAWN[1]) < 0.5 and abs(
            pos[2] - SPAWN[2]
        ) < 0.5:
            return
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for pose reset to spawn")


def settle(env, seconds):
    start = time.time()
    while time.time() - start < seconds:
        env.run_threaded(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.02)


def takeoff(env, args):
    """Climb to the episode start altitude before handing over to the pilot.

    Returns True once altitude is reached, False on timeout.
    """
    deadline = time.time() + args.takeoff_timeout
    while time.time() < deadline:
        outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=args.climb_altitude)
        if outputs[POS][2] >= args.start_alt:
            break
        time.sleep(0.02)
    else:
        return False

    # Arrest the climb before handing over. A fixed 0.5s hold was not enough:
    # momentum carried the drone ~0.9m past start_alt, and because the altitude
    # hold anchors its setpoint wherever the command last sat outside the
    # deadband, episodes began at 2.3-2.4m instead of 1.5m -- above the gates,
    # with only the trim range to recover. Wait for real calm instead.
    calm = 0
    while time.time() < deadline:
        outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0)
        if abs(outputs[VEL][2]) < 0.15:
            calm += 1
            if calm >= 4:
                break
        else:
            calm = 0
        time.sleep(0.05)

    # Then put the hold exactly on start_alt and let the PD fly it there, so
    # every episode begins from the same height regardless of overshoot.
    if getattr(env, "altitude_hold_kp", 0.0) > 0.0:
        env.set_altitude_setpoint(args.start_alt)
        while time.time() < deadline:
            outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0)
            if abs(outputs[POS][2] - args.start_alt) < 0.15 and abs(outputs[VEL][2]) < 0.15:
                break
            time.sleep(0.05)
    settle(env, 0.3)
    return True


def fly_episode(env, pilot, args, gates, pad_xy, task):
    """One closed-loop flight. Returns a per-episode result dict."""
    pilot._imu_history[:] = 0.0
    tracker = GateTracker(gates) if task == "gates" else None
    track = []
    blank_frames = 0
    result = "survived"
    controls = (0.0, 0.0, 0.0, 0.0)
    period = 1.0 / float(args.rate_hz)

    outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0)
    start_xy = (outputs[POS][0], outputs[POS][1])

    start = time.time()
    next_tick = start
    alt_cmds = []          # commanded altitude trace, for diagnosing oscillation
    alt_prev = 0.0         # EMA state for --alt-smooth
    alt_dc = 0.0           # slow DC estimate for --alt-highpass (PWM space)
    while True:
        t = time.time() - start
        if t >= args.duration:
            break

        outputs = env.run_threaded(*controls)
        image = outputs[0]
        pos = outputs[POS]
        att = outputs[ATT]
        vel = outputs[VEL]
        imu = outputs[IMU]
        track.append((round(t, 3), pos[0], pos[1], pos[2], vel[2]))
        if tracker is not None:
            tracker.update(t, pos)

        # Termination checks against ground truth.
        near_pad = pad_xy is not None and (
            math.hypot(pos[0] - pad_xy[0], pos[1] - pad_xy[1]) <= args.pad_radius
        )
        if task == "line" and near_pad:
            result = "completed"
            break
        if tracker is not None and len(tracker.passed) == len(gates):
            result = "completed"
            break
        if pos[2] < args.crash_z:
            result = "crashed"
            break
        if abs(att[0]) > args.crash_tilt_deg or abs(att[1]) > args.crash_tilt_deg:
            result = "crashed"
            break
        if pos[2] > args.flyaway_z or math.hypot(
            pos[0] - start_xy[0], pos[1] - start_xy[1]
        ) > args.flyaway_xy:
            result = "flyaway"
            break

        if not np.any(image):
            blank_frames += 1
        # att[2] is yaw in degrees; TorchPilot rotates world velocity into the
        # heading frame itself, matching how the tub labels were built.
        yaw, pitch, roll, altitude = pilot.run(image, *imu, *vel, att[2])
        if args.zero_altitude:
            # Diagnostic: discard the model's altitude axis and let the vertical
            # damper hold height alone. RL-teacher altitude labels are tiny trims
            # (std ~0.02) that sit inside DRONE_ALTITUDE_HOLD_DEADBAND, so a
            # student that failed to learn the axis emits near-constant noise on
            # it. Isolates "axis 4 is broken" from ordinary BC drift.
            altitude = 0.0
        else:
            # The distilled student tracks the teacher's mean height but
            # oscillates about it (alt_std ~0.7m) until it clips the floor. The
            # model is memoryless on this axis, so damp the command itself:
            # slew-limit first (bounds per-step change), then EMA.
            if args.alt_slew is not None:
                delta = max(-args.alt_slew, min(args.alt_slew, altitude - alt_prev))
                altitude = alt_prev + delta
            if args.alt_smooth is not None:
                altitude = args.alt_smooth * altitude + (1.0 - args.alt_smooth) * alt_prev
            if args.alt_highpass is not None:
                # Remove the DC component of the altitude command in PWM space.
                # A constant thrust bias B is opposed only by the vertical damper
                # (-K*vz), so the drone settles at vz = B/K forever: a 5 PWM bias
                # is a 0.055 m/s sink that reaches the floor in ~30s. Measured on
                # this student (mean cmd -4.6..-9.3 PWM) it predicts the observed
                # sink to within 0.005 m/s. The teacher's own DC is ~0, so the
                # model should supply modulation, not steady thrust.
                _p = math.copysign(abs(altitude) ** THR_SCALE, altitude) * 100.0
                alt_dc += (_p - alt_dc) * (period / args.alt_highpass)
                _p -= alt_dc
                altitude = math.copysign(
                    min(1.0, abs(_p) / 100.0) ** (1.0 / THR_SCALE), _p
                )
            if args.alt_gain is not None:
                # Multiply the effective THRUST command, not the raw command.
                # DroneGymEnv maps raw -> PWM as sign(a)*|a|**0.2, so a PWM gain
                # of G is raw * G**5. Measured: the student's damping gain is
                # ~7.8 PWM per m/s against the teacher's 51.9, so it commands
                # roughly a quarter of the thrust the teacher does.
                altitude = max(-1.0, min(1.0, altitude * (args.alt_gain ** 5)))
            alt_prev = altitude
        alt_cmds.append(float(altitude))
        controls = tuple(max(-1.0, min(1.0, float(v))) for v in (yaw, pitch, roll, altitude))

        next_tick += period
        time.sleep(max(0.0, next_tick - time.time()))

    survival = round(min(t, args.duration), 2)
    episode = {
        "result": result,
        "survival_s": survival,
        "blank_frames": blank_frames,
        "samples": len(track),
    }
    if args.trace and track:
        # Aligned per-step trace for oscillation analysis. track gets an entry
        # before the termination check and alt_cmds one after, so the final
        # (crashing) step has no command -- truncate to the common length.
        n = min(len(track), len(alt_cmds))
        episode["trace"] = {
            "t": [round(float(track[i][0]), 3) for i in range(n)],
            "x": [round(float(track[i][1]), 3) for i in range(n)],
            "y": [round(float(track[i][2]), 3) for i in range(n)],
            "z": [round(float(track[i][3]), 4) for i in range(n)],
            "vz": [round(float(track[i][4]), 4) for i in range(n)],
            "alt_cmd": [round(float(alt_cmds[i]), 6) for i in range(n)],
        }
    if alt_cmds:
        arr = np.asarray(alt_cmds, dtype=float)
        episode["alt_cmd_mean"] = round(float(arr.mean()), 4)
        episode["alt_cmd_std"] = round(float(arr.std()), 4)
        episode["alt_cmd_min"] = round(float(arr.min()), 4)
        episode["alt_cmd_max"] = round(float(arr.max()), 4)
        # How much the command moves step to step: the signature of oscillation.
        if arr.size > 1:
            episode["alt_cmd_absdiff_mean"] = round(float(np.abs(np.diff(arr)).mean()), 4)
    if track:
        episode.update(episode_metrics(track, start_xy, pad_xy))
    if tracker is not None:
        episode["gates_passed"] = len(tracker.passed)
        episode["gates_total"] = len(gates)
        episode["gate_pass_times_s"] = tracker.passed
    # Zero controls immediately so a crashed/finished pilot stops commanding.
    env.run_threaded(0.0, 0.0, 0.0, 0.0)
    return episode


# ---------------------------------------------------------------------------


def aggregate(episodes, task):
    n = len(episodes)
    agg = {
        "episodes": n,
        "success_rate": round(
            sum(1 for e in episodes if e["result"] == "completed") / n, 3
        ),
        "crash_rate": round(sum(1 for e in episodes if e["result"] == "crashed") / n, 3),
        "mean_survival_s": round(sum(e["survival_s"] for e in episodes) / n, 2),
    }
    if task == "gates":
        agg["mean_gates_passed"] = round(
            sum(e.get("gates_passed", 0) for e in episodes) / n, 2
        )
    if task == "line":
        agg["mean_along_track_m"] = round(
            sum(e.get("along_track_max_m", 0.0) for e in episodes) / n, 3
        )
        agg["mean_cross_track_m"] = round(
            sum(e.get("cross_track_mean_m", 0.0) for e in episodes) / n, 3
        )
    if task == "generic":
        agg["mean_max_displacement_m"] = round(
            sum(e.get("max_displacement_m", 0.0) for e in episodes) / n, 3
        )
    return agg


def evaluate(cfg, args):
    world = os.environ.get("GZ_WORLD", f"drone_course_{args.airframe}")
    gates, pad_xy = parse_world(world)
    task = args.task if args.task != "auto" else detect_task(world, gates, pad_xy)
    print(f"World: {world}  task: {task}  gates: {len(gates)}  pad: {pad_xy}")

    _wait_for_tcp(
        getattr(cfg, "BETAFLIGHT_RC_HOST", "127.0.0.1"),
        5761,
        args.betaflight_timeout,
        "BetaFlight MSP",
    )

    # The altitude post-processing diagnostics below operate in THRUST/PWM
    # space and only make sense when the altitude axis is raw motor throttle.
    # Under DRONE_VELOCITY_CONTROL that axis is a linear height trim, so they
    # would silently distort a command that means something else entirely --
    # exactly the class of silent mismatch that cost this project weeks (the
    # 120x160 resolution gotcha, the PWM-space flag). Refuse rather than warn.
    if getattr(cfg, "DRONE_VELOCITY_CONTROL", False):
        stale = [n for n, v in (("--alt-highpass", args.alt_highpass),
                                ("--alt-gain", args.alt_gain)) if v is not None]
        if stale:
            raise SystemExit(
                f"{', '.join(stale)} assume the altitude axis is raw thrust, but "
                "DRONE_VELOCITY_CONTROL makes it a linear height trim. Drop the "
                "flag, or set DRONE_VELOCITY_CONTROL=False to use it."
            )

    pilot = TorchPilot(
        input_shape=(
            getattr(cfg, "IMAGE_DEPTH", 3),
            getattr(cfg, "TRAIN_IMAGE_H", cfg.IMAGE_H),
            getattr(cfg, "TRAIN_IMAGE_W", cfg.IMAGE_W),
        ),
        seq_len=getattr(cfg, "SEQUENCE_LENGTH", 3),
        altitude_pwm_scale=(getattr(cfg, "DRONE_THROTTLE_SCALE", None)
                            if getattr(cfg, "TRAIN_ALTITUDE_PWM_SPACE", False) else None),
    )
    pilot.load(args.model)
    # Warm the inference graph (first MPS call can stall multiple seconds,
    # which would leave the drone uncontrolled at episode start).
    dummy = np.zeros((cfg.IMAGE_H, cfg.IMAGE_W, 3), dtype=np.uint8)
    for _ in range(3):
        pilot.run(dummy)

    env = build_drone_env(
        cfg,
        airframe=args.airframe,
        record_position=True,
        record_attitude=True,
        record_velocity=True,
        record_imu=True,
    )
    update_thread = threading.Thread(target=env.update, daemon=True)
    update_thread.start()

    episodes = []
    try:
        print(f"Waiting for readiness (timeout {args.ready_timeout:.1f}s)...")
        _wait_for_ready(env, args.ready_timeout)

        for i in range(args.runs):
            print(f"--- Episode {i + 1}/{args.runs} ---")
            request_reset(env, args.reset_timeout)
            settle(env, args.warmup)
            if not takeoff(env, args):
                episodes.append(
                    {"episode": i + 1, "result": "takeoff_failed", "survival_s": 0.0}
                )
                print(
                    f"Episode {i + 1}: takeoff failed to reach "
                    f"{args.start_alt}m within {args.takeoff_timeout}s"
                )
                continue
            episode = fly_episode(env, pilot, args, gates, pad_xy, task)
            episode["episode"] = i + 1
            episodes.append(episode)
            summary = f"{episode['result']} after {episode['survival_s']}s"
            if task == "gates":
                summary += f", gates {episode.get('gates_passed', 0)}/{len(gates)}"
            if task == "line":
                summary += (
                    f", along-track {episode.get('along_track_max_m', 0.0)}m,"
                    f" cross-track mean {episode.get('cross_track_mean_m', 0.0)}m"
                )
            print(f"Episode {i + 1}: {summary}")
    finally:
        env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        env.shutdown()
        update_thread.join(timeout=3.0)

    report = {
        "model": args.model,
        "world": world,
        "task": task,
        "airframe": args.airframe,
        "duration_s": args.duration,
        "rate_hz": args.rate_hz,
        "zero_altitude": bool(args.zero_altitude),
        "alt_smooth": args.alt_smooth,
        "alt_gain": args.alt_gain,
        "alt_highpass": args.alt_highpass,
        "alt_slew": args.alt_slew,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": episodes,
        "aggregate": aggregate(episodes, task) if episodes else {},
    }

    json_path = args.json_output
    if not json_path:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(log_dir, f"eval_flight_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"=== eval_flight: {args.model} on {world} ({task}) ===")
    for key, value in report["aggregate"].items():
        print(f"  {key}: {value}")
    print(f"JSON: {json_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Closed-loop Nanopilot autopilot eval")
    parser.add_argument("--model", required=True, help="TorchPilot .pth checkpoint")
    parser.add_argument("--airframe", choices=["65mm", "80mm"], default="80mm")
    parser.add_argument("--myconfig", default=None)
    parser.add_argument(
        "--task",
        choices=["auto", "line", "gates", "generic"],
        default="auto",
        help="auto = infer from world SDF (gates present / landing_pad present)",
    )
    parser.add_argument("--runs", type=int, default=3, help="episodes to fly")
    parser.add_argument("--duration", type=float, default=30.0, help="max seconds per episode")
    parser.add_argument("--warmup", type=float, default=6.0, help="hover settle before each episode")
    parser.add_argument("--start-alt", type=float, default=1.5, help="takeoff to this altitude before handing over (m)")
    parser.add_argument("--climb-altitude", type=float, default=0.15, help="normalized altitude command during takeoff")
    parser.add_argument("--takeoff-timeout", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--reset-timeout", type=float, default=10.0)
    parser.add_argument("--betaflight-timeout", type=float, default=10.0)
    parser.add_argument("--alt-highpass", type=float, default=None, metavar="TAU_S",
                        help="high-pass the altitude command with this time constant "
                             "(seconds), removing its DC thrust bias and leaving the "
                             "modulation. DC station-keeping is the damper's job.")
    parser.add_argument("--alt-gain", type=float, default=None, metavar="G",
                        help="multiply the altitude command by G in THRUST/PWM space "
                             "(raw command is scaled by G**5). Diagnostic for a "
                             "student whose control law is right but gain too low.")
    parser.add_argument("--trace", action="store_true",
                        help="store the per-step z / vz / altitude-command trace in the "
                             "metrics JSON (for oscillation period & phase analysis)")
    parser.add_argument("--alt-smooth", type=float, default=None, metavar="ALPHA",
                        help="EMA-smooth the altitude command: a = ALPHA*new + (1-ALPHA)*prev. "
                             "Lower = smoother. Omit for no smoothing.")
    parser.add_argument("--alt-slew", type=float, default=None, metavar="MAX_DELTA",
                        help="limit altitude command change per control step")
    parser.add_argument("--zero-altitude", action="store_true",
                        help="command altitude 0.0 instead of the model's 4th output "
                             "(diagnostic: leaves height to the vertical damper)")
    parser.add_argument("--crash-z", type=float, default=0.12, help="crash below this altitude (m)")
    parser.add_argument("--crash-tilt-deg", type=float, default=70.0)
    parser.add_argument("--flyaway-z", type=float, default=12.0)
    parser.add_argument("--flyaway-xy", type=float, default=60.0)
    parser.add_argument("--pad-radius", type=float, default=1.0, help="line task: success within this distance of the pad (m)")
    parser.add_argument("--json-output", default=None, help="metrics JSON path (default logs/eval_flight_<ts>.json)")
    args = parser.parse_args()

    myconfig = args.myconfig or f"drone_config_{args.airframe}.py"
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        myconfig=myconfig,
    )
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
