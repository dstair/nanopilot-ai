#!/usr/bin/env python3
"""
State-based PPO training for gate flying (the RL "teacher").

Requires BetaFlight SITL + Gazebo to already be running, for example via:
    GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh --no-manage --airframe=80mm

Or use the one-command wrapper:
    ./scripts/rl_train.sh --world=slalom_gates_80mm --total-steps=200000

Trains a small MLP policy on privileged Gazebo ground-truth state (no
camera) with PPO. The sim runs in real time, so budget roughly 70k env
steps per hour at the default 20Hz control rate. Outputs into --out:
    <out>/model.zip        final SB3 PPO policy
    <out>/vecnorm.pkl      VecNormalize statistics (needed at inference)
    <out>/checkpoints/     periodic snapshots

Resume an interrupted run with --resume=<out dir>.
Distill the trained teacher into vision tubs with rl_collect.py.
"""

import argparse
import json
import os
import threading
import time

import donkeycar as dk
import numpy as np

from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from eval_flight import parse_world
from rl_env import GateChaseEnv


def build_gate_env(cfg, args, drone_env, gates):
    return GateChaseEnv(
        drone_env,
        gates,
        rate_hz=args.rate_hz,
        max_episode_steps=args.max_episode_steps,
        control_altitude=args.control_altitude,
        altitude_scale=args.altitude_scale,
        start_alt=args.start_alt,
        spawn_jitter_xy=(args.spawn_jitter_x, args.spawn_jitter_y),
        spawn_jitter_yaw=args.spawn_jitter_yaw,
    )


def save_env_config(out_dir, args):
    """Record the env settings a policy was trained under.

    rl_collect reads this so the teacher is flown exactly as it was trained;
    a mismatched rate or altitude scale would silently corrupt the tubs.
    """
    path = os.path.join(out_dir, "env_config.json")
    with open(path, "w") as f:
        json.dump(
            {
                "world": args.world,
                "airframe": args.airframe,
                "rate_hz": args.rate_hz,
                "max_episode_steps": args.max_episode_steps,
                "control_altitude": args.control_altitude,
                "altitude_scale": args.altitude_scale,
                "start_alt": args.start_alt,
            },
            f,
            indent=2,
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="State-based PPO gate-flying trainer")
    parser.add_argument("--world", default=os.environ.get("GZ_WORLD", "slalom_gates_80mm"))
    parser.add_argument("--airframe", choices=["65mm", "80mm"], default="80mm")
    parser.add_argument("--myconfig", default=None)
    parser.add_argument("--out", default="models/rl/ppo_gates", help="output directory")
    parser.add_argument("--resume", default=None, help="output dir of a previous run to resume")
    parser.add_argument(
        "--total-steps", type=int, default=200_000,
        help="env steps to train; with --resume this many ADDITIONAL steps",
    )
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument(
        "--control-altitude", action="store_true",
        help="give the policy the altitude axis (default: damper holds height, "
             "policy flies yaw/pitch/roll only)",
    )
    parser.add_argument(
        "--altitude-scale", type=float, default=0.3,
        help="with --control-altitude, scale on that axis; full scale climbs "
             "out of the world",
    )
    parser.add_argument("--start-alt", type=float, default=1.5)
    parser.add_argument("--spawn-jitter-x", type=float, default=1.0)
    parser.add_argument("--spawn-jitter-y", type=float, default=1.5)
    parser.add_argument("--spawn-jitter-yaw", type=float, default=0.4)
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu is fastest for a small MLP")
    parser.add_argument("--checkpoint-every", type=int, default=20_000, help="env steps")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--betaflight-timeout", type=float, default=10.0)
    args = parser.parse_args()

    # Imports deferred so --help stays fast.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Some helpers (e.g. the altitude-hold pose subscriber) read GZ_WORLD.
    os.environ["GZ_WORLD"] = args.world

    gates, _pad = parse_world(args.world)
    if not gates:
        raise SystemExit(f"World {args.world!r} has no gate models; nothing to train on")
    print(f"World: {args.world}  gates: {[g['name'] for g in gates]}")

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

    out_dir = args.resume or args.out
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.zip")
    vecnorm_path = os.path.join(out_dir, "vecnorm.pkl")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="nanopilot-rl", config=vars(args), sync_tensorboard=True)

    try:
        print(f"Waiting for readiness (timeout {args.ready_timeout:.1f}s)...")
        _wait_for_ready(drone_env, args.ready_timeout)

        gate_env = build_gate_env(cfg, args, drone_env, gates)
        venv = DummyVecEnv([lambda: Monitor(gate_env)])
        if args.resume and os.path.exists(vecnorm_path):
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = True
        else:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

        try:
            import tensorboard  # noqa: F401

            tb_log = os.path.join(out_dir, "tb")
        except ImportError:
            tb_log = None

        if args.resume and os.path.exists(model_path):
            print(f"Resuming from {model_path}")
            model = PPO.load(model_path, env=venv, device=args.device)
        else:
            model = PPO(
                "MlpPolicy",
                venv,
                policy_kwargs=dict(
                    net_arch=dict(pi=[128, 128], vf=[128, 128]),
                    # Full-scale random sticks end episodes in ~1s; start
                    # exploration gentler (std ~0.37 instead of 1.0).
                    log_std_init=-1.0,
                ),
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                seed=args.seed,
                device=args.device,
                verbose=1,
                tensorboard_log=tb_log,
            )

        class NeutralHoldCallback(BaseCallback):
            """Hold hover while PPO pauses the rollout to run its update.

            The sim keeps flying in real time during gradient updates; without
            this the drone drifts on the last commanded sticks the whole pause.
            """

            def _on_rollout_end(self):
                gate_env.hold_hover()

            def _on_step(self):
                return True

        callbacks = [
            NeutralHoldCallback(),
            CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=ckpt_dir,
                name_prefix="ppo_gates",
                save_vecnormalize=True,
            ),
        ]
        if run is not None:
            from wandb.integration.sb3 import WandbCallback

            callbacks.append(WandbCallback(verbose=2))

        steps_per_hour = int(args.rate_hz * 3600)
        print(
            f"Training {args.total_steps} steps at {args.rate_hz:g}Hz "
            f"(~{args.total_steps / steps_per_hour:.1f}h of sim flight, real time)"
        )
        try:
            model.learn(
                total_timesteps=args.total_steps,
                callback=callbacks,
                reset_num_timesteps=not args.resume,
            )
        except KeyboardInterrupt:
            print("Interrupted; saving current model...")

        model.save(model_path)
        venv.save(vecnorm_path)
        cfg_path = save_env_config(out_dir, args)
        print(f"Saved policy: {model_path}")
        print(f"Saved VecNormalize stats: {vecnorm_path}")
        print(f"Saved env config: {cfg_path}")
    finally:
        drone_env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        drone_env.shutdown()
        update_thread.join(timeout=3.0)
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
