"""Diagnostic rollout of a trained RL teacher: HOW does it miss the gates?

Flies the policy deterministically and, at each gate-plane crossing, records
the lateral offset (ly) and height (z) against the pass window, plus approach
speed and how close the drone got to the gate center. This distinguishes:

  * incentive problem  — arrives slowly, in control, but wide/unaligned
  * geometry/dynamics  — arrives fast, overshoots, can't turn in time

Requires a running stack (start.sh --no-manage) on the gate world.

    uv run --env-file .env python gate_rollout.py \
        --policy=models/rl/ppo_gates_v3/checkpoints \
        --model=ppo_gates_280000_steps.zip --episodes=5
"""
import argparse
import json
import math
import os
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "nanopilot"))

import donkeycar as dk

from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from eval_flight import parse_world
from gate_expert import GateExpert
from rl_env import GateChaseEnv, POS, VEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", help="dir holding the model zip")
    p.add_argument("--model", help="model zip filename")
    p.add_argument("--vecnorm", help="vecnormalize pkl path")
    p.add_argument("--expert", action="store_true",
                   help="fly the scripted GateExpert instead of a trained "
                        "policy; needs DRONE_VELOCITY_CONTROL")
    p.add_argument("--cruise-speed", type=float, default=1.5)
    p.add_argument("--lookahead", type=float, default=3.0)
    p.add_argument("--height-gain", type=float, default=0.75)
    p.add_argument("--height-lookahead", type=float, default=0.0)
    p.add_argument("--world", default="slalom_gates_80mm")
    p.add_argument("--airframe", default="80mm")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=20.0)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--trace-output", default=None,
                   help="write per-step z / vz / altitude-command traces here (JSON), "
                        "in the same shape as eval_flight --trace")
    p.add_argument("--control-altitude", dest="control_altitude",
                   action="store_true", default=None,
                   help="override env_config.json; policy drives the 4th axis")
    p.add_argument("--altitude-scale", type=float, default=None)
    args = p.parse_args()

    if args.expert:
        model_path = "scripted GateExpert"
    else:
        if not (args.policy and args.model and args.vecnorm):
            raise SystemExit("--policy/--model/--vecnorm are required without --expert")
        model_path = os.path.join(args.policy, args.model)
        for path in (model_path, args.vecnorm):
            if not os.path.exists(path):
                raise SystemExit(f"Missing {path}")

    os.environ["GZ_WORLD"] = args.world
    gates, _pad = parse_world(args.world)
    if not gates:
        raise SystemExit(f"World {args.world!r} has no gates")
    print(f"World {args.world}: {len(gates)} gates")
    for i, g in enumerate(gates):
        print(f"  gate {i}: x={g['x']:.2f} y={g['y']:.2f} z={g['z']:.2f} "
              f"yaw={math.degrees(g['yaw']):.1f}deg")

    cfg = dk.load_config(
        config_path=str(pathlib.Path(__file__).resolve().parent.parent / "nanopilot" / "config.py"),
        myconfig=f"drone_config_{args.airframe}.py",
    )
    _wait_for_tcp("127.0.0.1", 5761, 10, "BetaFlight MSP")

    drone_env = build_drone_env(
        cfg, airframe=args.airframe, gz_world=args.world,
        record_position=True, record_attitude=True,
        record_velocity=True, record_imu=True,
    )
    threading.Thread(target=drone_env.update, daemon=True).start()
    _wait_for_ready(drone_env, 30)

    # The action/obs space must match what the checkpoint was trained with, or
    # VecNormalize.load fails on shape. Prefer env_config.json (written when a
    # run finishes); --policy often points at <out>/checkpoints, so look up one
    # level too. Explicit flags win.
    env_cfg = {}
    cands = () if args.expert else (
        os.path.join(args.policy, "env_config.json"),
        os.path.join(os.path.dirname(args.policy.rstrip("/")), "env_config.json"),
    )
    for cand in cands:
        if os.path.exists(cand):
            with open(cand) as fh:
                env_cfg = json.load(fh)
            print(f"env_config: {cand} -> {env_cfg}")
            break
    else:
        print("env_config: none found; using CLI defaults")
    if args.expert:
        # The expert emits a height TRIM in [-1, 1] that the env applies
        # linearly, so it must not be rescaled on the way through.
        control_altitude = True if args.control_altitude is None else args.control_altitude
        altitude_scale = 1.0 if args.altitude_scale is None else args.altitude_scale
    else:
        control_altitude = (env_cfg.get("control_altitude", False)
                            if args.control_altitude is None else args.control_altitude)
        altitude_scale = (env_cfg.get("altitude_scale", 0.3)
                          if args.altitude_scale is None else args.altitude_scale)
    print(f"Env: control_altitude={control_altitude} altitude_scale={altitude_scale}")

    gate_env = GateChaseEnv(
        drone_env, gates, rate_hz=args.rate_hz,
        max_episode_steps=args.max_episode_steps,
        control_altitude=control_altitude,
        altitude_scale=altitude_scale,
    )
    if args.expert:
        if not drone_env.velocity_control:
            raise SystemExit(
                "--expert needs DRONE_VELOCITY_CONTROL=True; it emits speed "
                "setpoints, which mean nothing in direct angle mode."
            )
        norm = None
        expert = GateExpert.from_env(
            drone_env, gates,
            cruise_speed=args.cruise_speed,
            lookahead=args.lookahead,
            height_gain=args.height_gain,
            height_lookahead=args.height_lookahead,
        )
        print(f"Expert: cruise={args.cruise_speed} lookahead={args.lookahead} "
              f"height_gain={args.height_gain} "
              f"height_lookahead={args.height_lookahead}")
    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        norm = VecNormalize.load(args.vecnorm, DummyVecEnv([lambda: gate_env]))
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(model_path, device="cpu")

    HALF_W = gate_env.gate_half_width

    def z_window(g):
        """Per-gate height window, measured from the SDF by parse_world."""
        return (g.get("z_min", gate_env.gate_z_min), g.get("z_max", gate_env.gate_z_max))

    print(f"\nPass window: |ly| <= {HALF_W}m, height per gate:")
    for g in gates:
        zlo, zhi = z_window(g)
        print(f"  {g['name']}: z={g['z']:.2f}  pass {zlo:.2f}-{zhi:.2f}")
    print()

    crossings = []
    traces = []
    for ep in range(args.episodes):
        ep_trace = {"t": [], "z": [], "vz": [], "alt_cmd": []}
        t0 = time.time()
        obs, _ = gate_env.reset(seed=1000 + ep)
        t0 = time.time()
        terminated = truncated = False
        info = {}
        prev_idx = gate_env._gate_idx
        prev_lx = None
        min_dist = {}
        print(f"--- episode {ep} ---")
        while not (terminated or truncated):
            idx = gate_env._gate_idx
            gate = gates[idx]

            if args.expert:
                tel = gate_env.last_telemetry
                action = expert.act_from_telemetry(tel, idx) if tel else np.zeros(4)
            else:
                action, _ = model.predict(
                    norm.normalize_obs(obs), deterministic=not args.stochastic
                )
            obs, _r, terminated, truncated, info = gate_env.step(action)

            # last_telemetry is TELEMETRY = pos(0:3) + att(3:6) + vel(6:9) + imu
            tel = gate_env.last_telemetry
            if not tel:
                continue
            pos = tuple(tel[0:3])
            vel = tuple(tel[6:9])

            # Same trace eval_flight --trace produces, so student and teacher
            # z/command traces can be compared directly. last_applied_action is
            # the command actually flown (altitude already scaled).
            ep_trace["t"].append(round(time.time() - t0, 3))
            ep_trace["z"].append(round(float(pos[2]), 4))
            ep_trace["vz"].append(round(float(vel[2]), 4))
            applied = getattr(gate_env, "last_applied_action", None)
            ep_trace["alt_cmd"].append(
                round(float(applied[3]), 6) if applied is not None and len(applied) > 3 else 0.0
            )

            lx, ly, z = gate_env._gate_local(gate, pos)
            d = gate_env._gate_dist(gate, pos)
            min_dist[idx] = min(min_dist.get(idx, 1e9), d)

            # detect the crossing ourselves for reporting
            if prev_lx is not None and prev_lx < 0.0 <= lx and idx == prev_idx:
                speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
                zlo, zhi = z_window(gate)
                dz = z - gate["z"]
                through = abs(ly) <= HALF_W and zlo <= z <= zhi
                why = []
                if abs(ly) > HALF_W:
                    why.append(f"lateral {ly:+.2f}m (limit {HALF_W})")
                if not (zlo <= z <= zhi):
                    why.append(f"height {z:.2f}m (band {zlo:.2f}-{zhi:.2f})")
                crossings.append(
                    dict(ep=ep, gate=idx, ly=ly, z=z, dz=dz, speed=speed,
                         through=through, z_lo=zlo, z_hi=zhi)
                )
                verdict = "PASS" if through else "MISS: " + ", ".join(why)
                print(f"  gate {idx} crossing: ly={ly:+.2f} z={z:.2f} "
                      f"dz={dz:+.2f} speed={speed:.2f}m/s  {verdict}")
            prev_lx = gate_env._prev_gate_lx
            if gate_env._gate_idx != prev_idx:
                prev_idx = gate_env._gate_idx
                prev_lx = None

        print(f"  end: {info.get('episode_result')} gates={info.get('gates_passed')}"
              f" closest approach per gate: "
              + ", ".join(f"g{k}={v:.2f}m" for k, v in sorted(min_dist.items())))
        ep_trace["result"] = info.get("episode_result")
        ep_trace["gates_passed"] = info.get("gates_passed")
        traces.append(ep_trace)

    if args.trace_output:
        with open(args.trace_output, "w") as fh:
            json.dump({"policy": model_path, "episodes": traces}, fh)
        print(f"\ntrace written: {args.trace_output}")

    print("\n=== crossing summary ===")
    if not crossings:
        print("no gate-plane crossings recorded")
    else:
        n = len(crossings)
        passes = sum(1 for c in crossings if c["through"])
        lat = [abs(c["ly"]) for c in crossings]
        spd = [c["speed"] for c in crossings]
        zs = [c["z"] for c in crossings]
        dzs = [abs(c["dz"]) for c in crossings]
        lat_miss = sum(1 for c in crossings if abs(c["ly"]) > HALF_W)
        z_miss = sum(1 for c in crossings if not (c["z_lo"] <= c["z"] <= c["z_hi"]))
        print(f"crossings={n} passes={passes} ({passes/n*100:.0f}%)")
        print(f"|lateral offset|: mean={np.mean(lat):.2f}m median={np.median(lat):.2f}m "
              f"max={np.max(lat):.2f}m  (window {HALF_W}m)")
        print(f"height at crossing: mean={np.mean(zs):.2f}m min={np.min(zs):.2f} "
              f"max={np.max(zs):.2f}")
        # The metric that reports height CONTROL, as opposed to whether the
        # drone happened to be inside the window: error against the gate.
        print(f"|height error| vs gate: mean={np.mean(dzs):.2f}m "
              f"median={np.median(dzs):.2f}m max={np.max(dzs):.2f}m")
        print(f"approach speed: mean={np.mean(spd):.2f} max={np.max(spd):.2f} m/s")
        print(f"misses by lateral={lat_miss}, by height={z_miss}")
        near = sum(1 for c in crossings if HALF_W < abs(c["ly"]) <= 2 * HALF_W)
        print(f"near misses (within 2x window laterally): {near}")

    drone_env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
    drone_env.shutdown()


if __name__ == "__main__":
    main()
