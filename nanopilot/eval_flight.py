#!/usr/bin/env python3
"""
Closed-loop autopilot evaluation for Nanopilot.

Loads a TorchPilot checkpoint, lets it fly the drone in the running sim for
one or more episodes, and scores each episode from Gazebo ground-truth pose:
survival time, along-track progress, cross-track error, gates passed.
Writes machine-readable JSON so runs/models can be compared over time.

Tasks, inferred from the world SDF unless --task says otherwise:

    gates    fly every gate in order; success = all of them
    land     fly to the pad and touch down on it; success = gentle, stopped,
             on the pad. Ground contact is scored by --touchdown-z /
             --settle-vz / --settle-speed, NOT by --crash-z, because a
             textbook landing passes crash_z on the way down
    line     fly to the pad and stop there; success = horizontal proximity
             while still airborne. Kept as an ablation of the land task
    generic  survival and displacement only

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
from PIL import Image

from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from gate_expert import GateExpert
from line_expert import LineExpert
from rl_env import takeoff as rl_takeoff
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

# Seconds of descent history the touchdown test looks back over to decide how
# hard the drone arrived. The sample AT contact is already decelerating -- the
# pose-derived velocity is low-passed with tau=0.1 and the ground stops the
# drone in one frame -- so classifying on that sample alone scores every impact
# as gentle. The fastest descent in the window before contact is the honest
# number, and it is what probe_descent.py's landing legs report.
TOUCHDOWN_VZ_WINDOW_S = 0.5


# ---------------------------------------------------------------------------
# World parsing


def world_sdf_path(world):
    return os.path.join(os.path.dirname(__file__), "..", "worlds", f"{world}.sdf")


# Margin between a gate's physical vertical clearance and the height window a
# pass is scored in, in metres. Small enough that the window still means "went
# through the hoop", large enough that a crossing interpolated between two 50ms
# samples is not scored a pass on a grazing clip of the frame.
GATE_Z_MARGIN = 0.2

# Fallback height window for a gate whose SDF has no box geometry to measure,
# in absolute metres. This is what every course used before the window was made
# gate-relative, and it is only reached if the model cannot be measured at all.
GATE_Z_ABS = (0.2, 3.0)


def gate_z_window(model, gate_z):
    """Vertical pass window for one gate model, in absolute metres.

    Derived from the gate's own geometry rather than hardcoded, because a
    course with VARIED GATE HEIGHTS needs a window that moves with the gate:
    an absolute 0.2-3.0m band would score a pass through a gate at z=2.35 for a
    drone cruising at 1.5m, i.e. it would not measure height at all.

    The measurement is the tallest box in the model -- the posts -- which spans
    the full opening by construction. On the flat courses the posts are 3m, so
    this yields 0.2-2.8m at a gate of z=1.5, which is the band those courses
    were always scored in.
    """
    tallest = 0.0
    for size_el in model.iter("size"):
        try:
            dims = [float(v) for v in (size_el.text or "").split()]
        except ValueError:
            continue
        if len(dims) == 3:
            tallest = max(tallest, dims[2])
    if tallest <= 0.0:
        return GATE_Z_ABS
    half = tallest / 2.0 - GATE_Z_MARGIN
    if half <= 0.0:
        return GATE_Z_ABS
    return (gate_z - half, gate_z + half)


def parse_world(world):
    """Extract gate poses and landing-pad position from the world SDF.

    Returns (gates, pad_xy) where gates is a list of dicts with x/y/z/yaw plus
    the per-gate vertical pass window z_min/z_max, sorted by model name, and
    pad_xy is (x, y) or None.
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
            z_min, z_max = gate_z_window(model, z)
            gates.append({"name": name, "x": x, "y": y, "z": z, "yaw": yaw,
                          "z_min": z_min, "z_max": z_max})
        elif name == "landing_pad":
            pad_xy = (x, y)
    gates.sort(key=lambda g: g["name"])
    return gates, pad_xy


def detect_task(world, gates, pad_xy):
    """Pick the scorer from what the world contains.

    A pad and no gates now means "land", not "line". The two differ in what
    counts as finishing: "line" completes on HORIZONTAL proximity to the pad
    while still airborne, which is the right test for a fly-the-line ablation
    and the wrong one for a task whose last five metres are vertical. "line"
    stays reachable with --task=line for exactly that ablation.
    """
    if gates:
        return "gates"
    if pad_xy is not None:
        return "land"
    return "generic"


# ---------------------------------------------------------------------------
# Scoring helpers


class GateTracker:
    """Counts gate passes from consecutive positions.

    A pass is a crossing of the gate's local x=0 plane while inside the
    opening: |lateral| <= half_width, and height inside the gate's own
    z_min/z_max as measured from the SDF by parse_world. Each gate counts once.

    The height window is PER GATE, not absolute. On a course with varied gate
    heights an absolute band would score a pass through a gate at z=2.35 for a
    drone cruising at 1.5m, so gate count would say nothing about height.

    Every crossing is also recorded in `crossings`, pass or miss, with its
    lateral offset and its height error against the gate. That is the metric
    that discriminates height control: gate COUNT only reports whether the
    drone was inside the window, while the error says how well it tracked.
    """

    def __init__(self, gates, half_width=0.9, z_min=0.2, z_max=3.0, near_width=None):
        self.gates = gates
        self.half_width = half_width
        # Lateral distance out to which a plane crossing still counts as an
        # attempt at this gate, for the crossing record only.
        self.near_width = 2.0 * half_width if near_width is None else near_width
        self.z_min = z_min
        self.z_max = z_max
        self.passed = {}  # name -> episode time of pass
        self.crossings = []  # dicts: name, t, ly, z, dz, passed
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
            z_min = g.get("z_min", self.z_min)
            z_max = g.get("z_max", self.z_max)
            ok = abs(ly_c) <= self.half_width and z_min <= z_c <= z_max
            # A gate's local x=0 plane is infinite, so the drone crosses it
            # whenever it happens to move past that line ANYWHERE in the world.
            # Only a crossing near the opening is an attempt at this gate;
            # wider ones would poison the height-error statistics with samples
            # taken 10m off to the side. Pass counting is unaffected either way,
            # since `ok` already requires being inside the opening.
            if abs(ly_c) > self.near_width:
                continue
            self.crossings.append(
                {
                    "gate": g["name"],
                    "t": round(t, 2),
                    "lateral_m": round(float(ly_c), 3),
                    "z_m": round(float(z_c), 3),
                    "height_err_m": round(float(z_c - g["z"]), 3),
                    "passed": bool(ok),
                }
            )
            if ok:
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


def classify_touchdown(recent, pos, pad_xy, pad_radius, settle_vz, settle_speed):
    """Decide what a ground contact WAS, from the approach that preceded it.

    `recent` is the tail of the track covering TOUCHDOWN_VZ_WINDOW_S before
    contact, as (t, x, y, z, vz, speed_xy) tuples. Returns (result, info).

    Three ways to reach the ground that are not a landing, each needing a
    different fix, so each gets its own result rather than being folded into
    "crashed":

      crashed        arrived faster than settle_vz vertically -- a fall
      skidded        gentle vertically but still travelling horizontally,
                     i.e. a belly-flop at cruise speed. Without this test the
                     scorer would call a drone that flew into the ground at
                     1.5 m/s a successful landing, because its VERTICAL speed
                     at contact can be tiny.
      landed_off_pad a real landing, in the wrong place

    Only the fourth case -- gentle, stopped, on the pad -- is "completed", so
    success_rate keeps meaning the same thing it does for the gates task.
    """
    impact_vz = min(p[4] for p in recent)
    impact_speed = max(p[5] for p in recent)
    pad_offset = (
        math.hypot(pos[0] - pad_xy[0], pos[1] - pad_xy[1])
        if pad_xy is not None else float("nan")
    )
    info = {
        "touchdown_vz_ms": round(float(impact_vz), 3),
        "touchdown_speed_ms": round(float(impact_speed), 3),
        "touchdown_pad_offset_m": round(float(pad_offset), 3),
    }
    if abs(impact_vz) >= settle_vz:
        result = "crashed"
    elif impact_speed >= settle_speed:
        result = "skidded"
    elif pad_xy is None or pad_offset > pad_radius:
        result = "landed_off_pad"
    else:
        result = "completed"
    return result, info


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

    Shares rl_env.takeoff with the collectors, so a policy is scored from the
    same starting state it was trained on. The old local version powered all
    the way to start_alt (~0.9m overshoot, then a slow sink back) and handed
    over on "within 0.15m and |vz| < 0.15", which the drone satisfies on the
    way THROUGH -- it then rang +-0.12m all flight (see rl_env._arrest_climb).

    Returns True once settled at start_alt, False on timeout.
    """
    return rl_takeoff(
        env, start_alt=args.start_alt, climb_altitude=args.climb_altitude,
        takeoff_timeout=args.takeoff_timeout)


def fly_episode(env, pilot, args, gates, pad_xy, task, expert=None):
    """One closed-loop flight. Returns a per-episode result dict.

    `expert` flies a scripted policy instead of a checkpoint, scored by exactly
    the same rules. That is the point: the expert is the CEILING for anything
    distilled from it, and a ceiling measured with a different scorer than the
    student is not a ceiling.
    """
    if pilot is not None:
        pilot._imu_history[:] = 0.0
    tracker = GateTracker(gates) if task == "gates" else None
    # GateExpert's target gate. Advances on ANY crossing of the current gate's
    # plane, pass or miss, exactly as GateChaseEnv does -- that is the labelling
    # the students were trained on, and it never sends the drone back behind a
    # missed gate.
    gate_idx = 0
    prev_gate_lx = None
    frame_dir = None
    if args.record_dir:
        frame_dir = os.path.join(args.record_dir, f"ep{args._episode:02d}")
        os.makedirs(frame_dir, exist_ok=True)
    n_frames = 0
    track = []
    blank_frames = 0
    result = "survived"
    controls = (0.0, 0.0, 0.0, 0.0)
    period = 1.0 / float(args.rate_hz)

    outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0)
    start_xy = (outputs[POS][0], outputs[POS][1])
    touchdown = None

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
        track.append((round(t, 3), pos[0], pos[1], pos[2], vel[2],
                      math.hypot(vel[0], vel[1])))
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
        if task == "land" and pos[2] < args.touchdown_z:
            # Ground contact on the landing task. Which OUTCOME it is depends on
            # how the drone arrived, so this must run instead of the crash_z
            # test rather than after it: crash_z defaults to 0.12 and a textbook
            # landing passes 0.12 on its way down (measured 2026-09-09: z<0.20
            # at 5.4s, touchdown at 8.6s), so the old scorer called every
            # successful landing a crash.
            recent = [p for p in track if p[0] >= t - TOUCHDOWN_VZ_WINDOW_S]
            result, touchdown = classify_touchdown(
                recent, pos, pad_xy, args.pad_radius,
                args.settle_vz, args.settle_speed,
            )
            touchdown["touchdown_s"] = round(t, 2)
            break
        if task != "land" and pos[2] < args.crash_z:
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
        elif frame_dir is not None:
            # Exactly the frame the pilot is handed this step (the expert
            # ignores it), so a video shows what a student had to act on.
            Image.fromarray(image).save(
                os.path.join(frame_dir, f"{n_frames:05d}.jpg"), quality=92)
            n_frames += 1
        if isinstance(expert, GateExpert):
            g = gates[gate_idx]
            lx = (math.cos(g["yaw"]) * (pos[0] - g["x"])
                  + math.sin(g["yaw"]) * (pos[1] - g["y"]))
            if prev_gate_lx is not None and prev_gate_lx < 0.0 <= lx:
                gate_idx = min(gate_idx + 1, len(gates) - 1)
                g = gates[gate_idx]
                lx = (math.cos(g["yaw"]) * (pos[0] - g["x"])
                      + math.sin(g["yaw"]) * (pos[1] - g["y"]))
            prev_gate_lx = lx
            yaw, pitch, roll, altitude = expert.act(
                pos, math.radians(att[2]), gate_idx)
        elif expert is not None:
            # Privileged state, no image: the scripted expert is a pure
            # function of pose, which is what makes it a free DAgger labeller.
            yaw, pitch, roll, altitude = expert.act(pos, math.radians(att[2]))
        else:
            # att[2] is yaw in degrees; TorchPilot rotates world velocity into
            # the heading frame itself, matching how the tub labels were built.
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
    if touchdown is not None:
        episode.update(touchdown)
    if tracker is not None:
        episode["gates_passed"] = len(tracker.passed)
        episode["gates_total"] = len(gates)
        episode["gate_pass_times_s"] = tracker.passed
        episode["gate_crossings"] = tracker.crossings
        if tracker.crossings:
            herr = np.array([abs(c["height_err_m"]) for c in tracker.crossings])
            lat = np.array([abs(c["lateral_m"]) for c in tracker.crossings])
            episode["crossing_height_err_mean_m"] = round(float(herr.mean()), 3)
            episode["crossing_height_err_max_m"] = round(float(herr.max()), 3)
            episode["crossing_lateral_mean_m"] = round(float(lat.mean()), 3)
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
        # Height error at the crossing plane, over every crossing in every
        # episode. On a varied-height course this is the metric that reports
        # altitude tracking; gate count only says whether the drone was inside
        # the window at all.
        errs = [abs(c["height_err_m"]) for e in episodes for c in e.get("gate_crossings", [])]
        if errs:
            agg["mean_crossing_height_err_m"] = round(sum(errs) / len(errs), 3)
            agg["max_crossing_height_err_m"] = round(max(errs), 3)
    if task in ("line", "land"):
        agg["mean_along_track_m"] = round(
            sum(e.get("along_track_max_m", 0.0) for e in episodes) / n, 3
        )
        agg["mean_cross_track_m"] = round(
            sum(e.get("cross_track_mean_m", 0.0) for e in episodes) / n, 3
        )
    if task == "land":
        # success_rate above already counts only clean landings. These split
        # the failures, which is the point of the task: "did not land on the
        # pad" and "fell out of the sky" need different fixes.
        for label in ("landed_off_pad", "skidded"):
            agg[f"{label}_rate"] = round(
                sum(1 for e in episodes if e["result"] == label) / n, 3
            )
        touched = [e for e in episodes if "touchdown_s" in e]
        agg["touchdown_rate"] = round(len(touched) / n, 3)
        if touched:
            for key, digits in (("touchdown_vz_ms", 3),
                                ("touchdown_speed_ms", 3),
                                ("touchdown_pad_offset_m", 3),
                                ("touchdown_s", 2)):
                agg[f"mean_{key}"] = round(
                    sum(e[key] for e in touched) / len(touched), digits
                )
            agg["max_touchdown_vz_abs_ms"] = round(
                max(abs(e["touchdown_vz_ms"]) for e in touched), 3
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
    for g in gates:
        print(
            f"  {g['name']}: ({g['x']:.2f}, {g['y']:.2f}, {g['z']:.2f}) "
            f"yaw {math.degrees(g['yaw']):+.1f}  pass z {g['z_min']:.2f}-{g['z_max']:.2f}"
        )

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

    pilot = None if args.expert else TorchPilot(
        input_shape=(
            getattr(cfg, "IMAGE_DEPTH", 3),
            getattr(cfg, "TRAIN_IMAGE_H", cfg.IMAGE_H),
            getattr(cfg, "TRAIN_IMAGE_W", cfg.IMAGE_W),
        ),
        seq_len=getattr(cfg, "SEQUENCE_LENGTH", 3),
        altitude_pwm_scale=(getattr(cfg, "DRONE_THROTTLE_SCALE", None)
                            if getattr(cfg, "TRAIN_ALTITUDE_PWM_SPACE", False) else None),
    )
    if pilot is not None:
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
    if args.trim_range is not None:
        # In-flight override of DRONE_ALTITUDE_TRIM_RANGE, so the landing task
        # can be validated at the range it needs WITHOUT changing the config
        # and therefore without changing the plant for the gate courses. That
        # is a separate, gated step. The env reads this every control cycle.
        print(f"*** TRIM_RANGE OVERRIDE: {env.altitude_trim_range} -> "
              f"{args.trim_range} (config is NOT changed) ***")
        env.altitude_trim_range = float(args.trim_range)

    expert = None
    if args.expert and task == "gates":
        expert = GateExpert.from_env(
            env, gates,
            cruise_speed=args.cruise_speed,
            lookahead=args.lookahead,
            height_gain=args.height_gain,
        )
        print(
            f"GateExpert: {len(gates)} gates cruise {args.cruise_speed} m/s "
            f"lookahead {args.lookahead}m Kz {args.height_gain}"
        )
    elif args.expert:
        if task != "land":
            raise SystemExit(
                f"--expert flies GateExpert on gate courses and LineExpert on "
                f"the land task; this world scored as '{task}'."
            )
        expert = LineExpert.from_env(
            env, SPAWN[:2], pad_xy,
            cruise_speed=args.cruise_speed,
            lookahead=args.lookahead,
            height_gain=args.height_gain,
            approach_gain=args.approach_gain,
            descend_radius=args.descend_radius,
            land_radius=args.land_radius,
            flare_alt=args.flare_alt,
            ground_target=args.ground_target,
        )
        print(
            f"LineExpert: pad {pad_xy} cruise {args.cruise_speed} m/s "
            f"lookahead {args.lookahead}m Kz {args.height_gain} "
            f"approach_gain {args.approach_gain} descend {args.descend_radius}m "
            f"-> land {args.land_radius}m target {args.ground_target}m "
            f"trim {env.altitude_trim_range}"
        )
        reach = env.altitude_trim_range - (args.start_alt - args.ground_target)
        if reach < 0:
            print(
                f"WARNING: trim range {env.altitude_trim_range} cannot reach "
                f"{args.ground_target}m from a {args.start_alt}m cruise (short "
                f"by {-reach:.2f}m). The command will clip at -1 and the drone "
                f"will hover at {args.start_alt - env.altitude_trim_range:.2f}m "
                f"instead of landing."
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
            args._episode = i + 1
            episode = fly_episode(env, pilot, args, gates, pad_xy, task, expert)
            episode["episode"] = i + 1
            episodes.append(episode)
            summary = f"{episode['result']} after {episode['survival_s']}s"
            if task == "gates":
                summary += f", gates {episode.get('gates_passed', 0)}/{len(gates)}"
                if "crossing_height_err_mean_m" in episode:
                    summary += (
                        f", height err {episode['crossing_height_err_mean_m']}m "
                        f"(max {episode['crossing_height_err_max_m']})"
                    )
            if task in ("line", "land"):
                summary += (
                    f", along-track {episode.get('along_track_max_m', 0.0)}m,"
                    f" cross-track mean {episode.get('cross_track_mean_m', 0.0)}m"
                )
            if task == "land":
                if "touchdown_s" in episode:
                    summary += (
                        f", touchdown vz {episode['touchdown_vz_ms']} m/s"
                        f" speed {episode['touchdown_speed_ms']} m/s"
                        f" {episode['touchdown_pad_offset_m']}m from pad"
                    )
                else:
                    summary += (
                        ", NEVER TOUCHED DOWN"
                        f" (final {episode.get('final_dist_to_pad_m', '?')}m"
                        f" from pad, z {episode.get('alt_min_m', '?')}m)"
                    )
            print(f"Episode {i + 1}: {summary}")
    finally:
        env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        env.shutdown()
        update_thread.join(timeout=3.0)

    report = {
        "model": args.model or f"scripted:{type(expert).__name__}",
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
    print(f"=== eval_flight: {report['model']} on {world} ({task}) ===")
    for key, value in report["aggregate"].items():
        print(f"  {key}: {value}")
    print(f"JSON: {json_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Closed-loop Nanopilot autopilot eval")
    parser.add_argument("--model", default=None, help="TorchPilot .pth checkpoint")
    parser.add_argument("--expert", action="store_true",
                        help="fly the scripted expert instead of a checkpoint "
                             "(GateExpert on gate courses, LineExpert on the "
                             "land task), scored by the same rules. Use this to "
                             "establish the ceiling before distilling anything")
    parser.add_argument("--trim-range", type=float, default=None, metavar="M",
                        help="override DRONE_ALTITUDE_TRIM_RANGE in flight, "
                             "without touching the config (and so without "
                             "changing the plant for any other course)")
    parser.add_argument("--cruise-speed", type=float, default=1.5,
                        help="--expert: cruise speed along the line (m/s)")
    parser.add_argument("--lookahead", type=float, default=3.0,
                        help="--expert: pure-pursuit lookahead (m)")
    parser.add_argument("--height-gain", type=float, default=0.75,
                        help="--expert: Kz, matching GateExpert's default")
    parser.add_argument("--approach-gain", type=float, default=0.6,
                        help="--expert: speed = approach_gain * distance to "
                             "pad, capped at cruise (1/s)")
    parser.add_argument("--descend-radius", type=float, default=8.0,
                        help="--expert: start descending inside this radius "
                             "(m). A GLIDE SLOPE, sized so the pad stays in "
                             "the camera -- see line_expert's docstring")
    parser.add_argument("--land-radius", type=float, default=0.5,
                        help="--expert: full-down command inside this radius (m)")
    parser.add_argument("--flare-alt", type=float, default=-0.1,
                        help="--expert: floor on the glide -- the reference "
                             "may not go below this until the drone is within "
                             "--land-radius of the pad. Set equal to "
                             "--ground-target to disable the floor")
    parser.add_argument("--ground-target", type=float, default=-0.1,
                        help="--expert: height reference at the pad (m). Below "
                             "ground on purpose -- aiming at 0.0 approaches "
                             "asymptotically and never touches down")
    parser.add_argument("--airframe", choices=["65mm", "80mm"], default="80mm")
    parser.add_argument("--myconfig", default=None)
    parser.add_argument(
        "--task",
        choices=["auto", "line", "land", "gates", "generic"],
        default="auto",
        help="auto = infer from world SDF (gates present / landing_pad present)",
    )
    parser.add_argument("--runs", type=int, default=3, help="episodes to fly")
    parser.add_argument("--duration", type=float, default=30.0, help="max seconds per episode")
    parser.add_argument("--warmup", type=float, default=1.0, help="on-ground settle after each reset, before takeoff (GateChaseEnv settle_s)")
    parser.add_argument("--start-alt", type=float, default=1.5, help="takeoff to this altitude before handing over (m)")
    parser.add_argument("--climb-altitude", type=float, default=0.6, help="cap on the takeoff thrust ramp (direct_thrust), as in GateChaseEnv")
    parser.add_argument("--takeoff-timeout", type=float, default=15.0,
                        help="climb + arrest budget (the at-rest handover wait is separate)")
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
    parser.add_argument("--record-dir", default=None, metavar="DIR",
                        help="save every camera frame the pilot sees as "
                             "DIR/epNN/NNNNN.jpg, one per control step")
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
    parser.add_argument("--crash-z", type=float, default=0.12,
                        help="crash below this altitude (m). NOT used by the "
                             "land task, where ground contact is scored by "
                             "--touchdown-z / --settle-vz instead")
    parser.add_argument("--touchdown-z", type=float, default=0.06,
                        help="land task: ground contact below this altitude (m)")
    parser.add_argument("--settle-vz", type=float, default=0.15, metavar="MS",
                        help="land task: arriving faster than this vertically "
                             "is a crash, not a landing (m/s). Measured "
                             "2026-09-09: a full-down command from a 1.5m "
                             "cruise touches at 0.02 m/s at TRIM_RANGE=1.6 and "
                             "0.18 at 2.0, so this sits between them")
    parser.add_argument("--settle-speed", type=float, default=0.5, metavar="MS",
                        help="land task: still travelling faster than this "
                             "horizontally at contact is a skid, not a landing "
                             "(m/s). Deliberately generous -- a third of the "
                             "2.0 m/s cruise -- because its job is to refuse to "
                             "score a belly-flop as a success, not to grade "
                             "landing style")
    parser.add_argument("--crash-tilt-deg", type=float, default=70.0)
    parser.add_argument("--flyaway-z", type=float, default=12.0)
    parser.add_argument("--flyaway-xy", type=float, default=60.0)
    parser.add_argument("--pad-radius", type=float, default=1.0,
                        help="line task: success within this distance of the "
                             "pad (m). land task: touching down further out "
                             "than this is landed_off_pad, not a success")
    parser.add_argument("--json-output", default=None, help="metrics JSON path (default logs/eval_flight_<ts>.json)")
    args = parser.parse_args()
    if bool(args.model) == bool(args.expert):
        parser.error("give exactly one of --model or --expert")

    myconfig = args.myconfig or f"drone_config_{args.airframe}.py"
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        myconfig=myconfig,
    )
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
