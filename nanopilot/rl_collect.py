#!/usr/bin/env python3
"""
Teacher-to-student data collection: fly a trained RL policy (rl_train.py
output) and record standard tubs for the vision behavior-cloning pipeline.

The policy flies from privileged ground-truth state; the tub records what the
camera saw plus the four-axis commands, so torch_train.py can distill the
teacher into a deployable vision model. Episodes that end in a crash or
flyaway are dropped (rows are buffered per episode and only flushed on a
clean ending) so bad labels never reach the tub.

Requires BetaFlight SITL + Gazebo to already be running, for example via:
    GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh --no-manage --airframe=80mm

Example:
    uv run --env-file .env python nanopilot/rl_collect.py \
        --policy=models/rl/ppo_gates --episodes=10 --min-gates=3
"""

import argparse
import json
import os
import threading
import time

import donkeycar as dk
import numpy as np
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.tub_v2 import TubWriter

from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from eval_flight import parse_world
from gate_expert import GateExpert
from rl_env import GateChaseEnv
from tub_schema import DRONE_TUB_INPUTS, DRONE_TUB_TYPES


def main():
    parser = argparse.ArgumentParser(description="Record tubs flown by an RL teacher policy")
    parser.add_argument("--policy", help="rl_train.py output dir (model.zip + vecnorm.pkl)")
    parser.add_argument("--expert", action="store_true",
                        help="record the scripted GateExpert instead of a trained "
                             "policy; needs DRONE_VELOCITY_CONTROL")
    parser.add_argument("--cruise-speed", type=float, default=1.5)
    parser.add_argument("--lookahead", type=float, default=3.0)
    parser.add_argument("--height-gain", type=float, default=0.5)
    parser.add_argument("--world", default=os.environ.get("GZ_WORLD", "slalom_gates_80mm"))
    parser.add_argument("--airframe", choices=["65mm", "80mm"], default="80mm")
    parser.add_argument("--myconfig", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--min-gates", type=int, default=0,
                        help="drop episodes that pass fewer gates than this")
    parser.add_argument("--keep-crashed", action="store_true",
                        help="keep rows from crashed/flyaway episodes (default: drop)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample the policy instead of deterministic actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--betaflight-timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.expert:
        model_path = vecnorm_path = None
    else:
        if not args.policy:
            raise SystemExit("--policy is required without --expert")
        model_path = os.path.join(args.policy, "model.zip")
        vecnorm_path = os.path.join(args.policy, "vecnorm.pkl")
        for path in (model_path, vecnorm_path):
            if not os.path.exists(path):
                raise SystemExit(f"Missing {path}; --policy must be an rl_train.py output dir")

    # Some helpers (e.g. the altitude-hold pose subscriber) read GZ_WORLD.
    os.environ["GZ_WORLD"] = args.world

    gates, _pad = parse_world(args.world)
    if not gates:
        raise SystemExit(f"World {args.world!r} has no gate models")

    myconfig = args.myconfig or f"drone_config_{args.airframe}.py"
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        myconfig=myconfig,
    )

    _wait_for_tcp(
        getattr(cfg, "BETAFLIGHT_RC_HOST", "127.0.0.1"),
        5761,
        args.betaflight_timeout,
        "BetaFlight MSP",
    )

    drone_env = build_drone_env(
        cfg,
        airframe=args.airframe,
        gz_world=args.world,
        record_position=True,
        record_attitude=True,
        record_velocity=True,
        record_imu=True,
    )
    update_thread = threading.Thread(target=drone_env.update, daemon=True)
    update_thread.start()

    tub_writer = None
    kept = dropped = 0
    try:
        print(f"Waiting for readiness (timeout {args.ready_timeout:.1f}s)...")
        _wait_for_ready(drone_env, args.ready_timeout)

        # Fly the teacher exactly as it was trained; a mismatched control rate
        # or altitude scale would put the policy out of distribution and write
        # labels the student can't reproduce.
        env_cfg = {}
        cfg_path = os.path.join(args.policy or "", "env_config.json")
        if args.expert:
            # The expert emits a linear height trim; scaling it would change
            # what the recorded labels mean.
            env_cfg = {"control_altitude": True, "altitude_scale": 1.0}
        elif os.path.exists(cfg_path):
            with open(cfg_path) as f:
                env_cfg = json.load(f)
            print(f"Using training env config: {env_cfg}")
        else:
            print(f"WARNING: no env_config.json in {args.policy}; using CLI defaults")

        gate_env = GateChaseEnv(
            drone_env,
            gates,
            rate_hz=env_cfg.get("rate_hz", args.rate_hz),
            max_episode_steps=env_cfg.get("max_episode_steps", args.max_episode_steps),
            control_altitude=env_cfg.get("control_altitude", False),
            altitude_scale=env_cfg.get("altitude_scale", 0.3),
        )

        if args.expert:
            if not drone_env.velocity_control:
                raise SystemExit(
                    "--expert needs DRONE_VELOCITY_CONTROL=True; it emits speed "
                    "setpoints, which mean nothing in direct angle mode."
                )
            expert = GateExpert.from_env(
                drone_env, gates,
                cruise_speed=args.cruise_speed, lookahead=args.lookahead,
                height_gain=args.height_gain,
            )
            norm = model = None
            print(f"Expert: cruise={args.cruise_speed} lookahead={args.lookahead}")
        else:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            # VecNormalize is only used for its frozen observation statistics.
            norm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: gate_env]))
            norm.training = False
            norm.norm_reward = False
            model = PPO.load(model_path, device="cpu")

        tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path()
        tub_writer = TubWriter(
            tub_path, inputs=DRONE_TUB_INPUTS, types=DRONE_TUB_TYPES, metadata=[]
        )
        print(f"TUB_PATH={tub_path}")

        for ep in range(args.episodes):
            obs, _info = gate_env.reset(seed=args.seed + ep)
            rows = []
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                if args.expert:
                    tel = gate_env.last_telemetry
                    action = (expert.act_from_telemetry(tel, gate_env._gate_idx)
                              if tel else np.zeros(4))
                else:
                    action, _ = model.predict(
                        norm.normalize_obs(obs), deterministic=not args.stochastic
                    )
                obs, _reward, terminated, truncated, info = gate_env.step(action)
                # The commands actually sent to the drone, not the raw policy
                # output (the altitude axis is scaled inside the env).
                steering, throttle, roll, altitude = (
                    float(v) for v in gate_env.last_applied_action
                )
                rows.append(
                    (
                        gate_env.last_image,
                        steering,
                        throttle,
                        roll,
                        altitude,
                        "pilot",
                        *gate_env.last_telemetry,
                    )
                )

            result = info.get("episode_result", "unknown")
            gates_passed = info.get("gates_passed", 0)
            keep = (args.keep_crashed or result in ("completed", "timeout")) and (
                gates_passed >= args.min_gates
            )
            if keep:
                for row in rows:
                    tub_writer.run(*row)
                kept += 1
            else:
                dropped += 1
            print(
                f"Episode {ep + 1}/{args.episodes}: {result}, "
                f"gates {gates_passed}/{len(gates)}, {len(rows)} rows "
                f"({'kept' if keep else 'dropped'})"
            )
    finally:
        if tub_writer is not None:
            tub_writer.shutdown()
        drone_env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        drone_env.shutdown()
        update_thread.join(timeout=3.0)

    print(f"Episodes kept: {kept}, dropped: {dropped}")
    if tub_writer is not None:
        print(f"Done: {tub_path}")


if __name__ == "__main__":
    main()
