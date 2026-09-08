"""DAgger round: fly the STUDENT, label with the TEACHER.

Plain behavior cloning trains only on the teacher's own state distribution, so
the student's small errors compound into states the teacher never visited and
never labeled. DAgger fixes exactly that: roll out the STUDENT so the visited
states are the ones it actually reaches, but write the TEACHER's action as the
label for each of those states.

That is unusually cheap here because the teacher is state-based: it can label
any state on demand from privileged Gazebo ground truth, so no human is needed
in the loop.

Crashed episodes are KEPT by default (unlike rl_collect.py). The states just
before a crash are precisely the corrective examples that plain BC lacks --
dropping them would throw away the entire point of the round.

    GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh \
        --no-manage --airframe=80mm
    uv run --env-file .env python nanopilot/dagger_collect.py \
        --policy=models/rl/ppo_gates_v4_best \
        --student=models/bc_rl_v4best_pwm.pth --episodes=25
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
from torch_pilot import TorchPilot
from tub_schema import DRONE_TUB_INPUTS, DRONE_TUB_TYPES


def main():
    parser = argparse.ArgumentParser(
        description="DAgger: fly the student, record teacher labels")
    parser.add_argument("--policy",
                        help="teacher dir (model.zip + vecnorm.pkl + env_config.json); "
                             "not needed with --expert")
    parser.add_argument("--expert", action="store_true",
                        help="label with the scripted GateExpert instead of a trained "
                             "policy; needs DRONE_VELOCITY_CONTROL. The expert is a pure "
                             "function of privileged state, so it labels any state the "
                             "student reaches with no checkpoint to load.")
    parser.add_argument("--cruise-speed", type=float, default=1.5)
    parser.add_argument("--lookahead", type=float, default=3.0)
    parser.add_argument("--height-gain", type=float, default=0.5)
    parser.add_argument("--student", required=True,
                        help="student .pth flown to generate the state distribution")
    parser.add_argument("--world", default="slalom_gates_80mm")
    parser.add_argument("--airframe", default="80mm", choices=["65mm", "80mm"])
    parser.add_argument("--myconfig", default=None)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="probability of flying the TEACHER's action instead of the "
                             "student's on a given step. 0.0 = pure student rollout "
                             "(standard DAgger after round 0); raise it if the student "
                             "crashes too fast to gather useful states.")
    parser.add_argument("--drop-crashed", action="store_true",
                        help="drop crashed episodes (default: KEEP them -- the "
                             "pre-crash states are the corrective data)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--betaflight-timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.expert:
        model_path = vecnorm_path = None
        required = (args.student,)
    else:
        if not args.policy:
            raise SystemExit("--policy is required without --expert")
        model_path = os.path.join(args.policy, "model.zip")
        vecnorm_path = os.path.join(args.policy, "vecnorm.pkl")
        required = (model_path, vecnorm_path, args.student)
    for path in required:
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")

    os.environ["GZ_WORLD"] = args.world
    gates, _pad = parse_world(args.world)
    if not gates:
        raise SystemExit(f"World {args.world!r} has no gate models")

    myconfig = args.myconfig or f"drone_config_{args.airframe}.py"
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        myconfig=myconfig,
    )

    _wait_for_tcp(getattr(cfg, "BETAFLIGHT_RC_HOST", "127.0.0.1"), 5761,
                  args.betaflight_timeout, "BetaFlight MSP")

    drone_env = build_drone_env(
        cfg, airframe=args.airframe, gz_world=args.world,
        record_position=True, record_attitude=True,
        record_velocity=True, record_imu=True,
    )
    update_thread = threading.Thread(target=drone_env.update, daemon=True)
    update_thread.start()

    rng = np.random.default_rng(args.seed)
    tub_writer = None
    kept = dropped = 0
    try:
        print(f"Waiting for readiness (timeout {args.ready_timeout:.1f}s)...")
        _wait_for_ready(drone_env, args.ready_timeout)

        env_cfg = {}
        if args.expert:
            # Match rl_collect.py: the expert emits a LINEAR height trim in
            # DonkeyCar units, so it must reach the env unscaled or the recorded
            # altitude labels mean something different from the tubs it is
            # aggregated with.
            env_cfg = {"control_altitude": True, "altitude_scale": 1.0}
        else:
            cfg_path = os.path.join(args.policy, "env_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    env_cfg = json.load(f)
                print(f"Using teacher env config: {env_cfg}")
            else:
                print(f"WARNING: no env_config.json in {args.policy}; using CLI defaults")

        gate_env = GateChaseEnv(
            drone_env, gates,
            rate_hz=env_cfg.get("rate_hz", args.rate_hz),
            max_episode_steps=env_cfg.get("max_episode_steps", args.max_episode_steps),
            control_altitude=env_cfg.get("control_altitude", False),
            altitude_scale=env_cfg.get("altitude_scale", 0.3),
        )
        alt_scale = gate_env.altitude_scale
        control_alt = gate_env.control_altitude

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
            norm = teacher = None
            print(f"Expert labeller: cruise={args.cruise_speed} "
                  f"lookahead={args.lookahead} height_gain={args.height_gain}")
        else:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            norm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: gate_env]))
            norm.training = False
            norm.norm_reward = False
            expert = None
            teacher = PPO.load(model_path, device="cpu")

        # The student must be built exactly as it was trained, or its outputs are
        # silently wrong: resolution (AdaptiveAvgPool hides a mismatch) and the
        # PWM-space altitude flag (no error either way).
        pilot = TorchPilot(
            input_shape=(getattr(cfg, "IMAGE_DEPTH", 3),
                         getattr(cfg, "TRAIN_IMAGE_H", cfg.IMAGE_H),
                         getattr(cfg, "TRAIN_IMAGE_W", cfg.IMAGE_W)),
            seq_len=getattr(cfg, "SEQUENCE_LENGTH", 3),
            altitude_pwm_scale=(getattr(cfg, "DRONE_THROTTLE_SCALE", None)
                                if getattr(cfg, "TRAIN_ALTITUDE_PWM_SPACE", False) else None),
        )
        pilot.load(args.student)

        tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path()
        tub_writer = TubWriter(tub_path, inputs=DRONE_TUB_INPUTS,
                               types=DRONE_TUB_TYPES, metadata=[])
        print(f"TUB_PATH={tub_path}")
        print(f"beta={args.beta} (0 = pure student rollout)")

        n_teacher_steps = n_student_steps = 0
        for ep in range(args.episodes):
            obs, _info = gate_env.reset(seed=args.seed + ep)
            rows = []
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                # TEACHER action at the current state -> this is the LABEL.
                # Note it is evaluated at the state the STUDENT reached, which is
                # the whole point: it covers the student's own error distribution,
                # which the expert's flawless rollouts never visit.
                if expert is not None:
                    tel_now = gate_env.last_telemetry
                    t_action = (expert.act_from_telemetry(tel_now, gate_env._gate_idx)
                                if tel_now is not None else np.zeros(4, dtype=np.float32))
                else:
                    t_action, _ = teacher.predict(norm.normalize_obs(obs), deterministic=True)
                t_action = np.clip(np.asarray(t_action, dtype=np.float32).flatten(), -1.0, 1.0)
                label = np.zeros(4, dtype=np.float32)
                label[:3] = t_action[:3]
                if control_alt:
                    label[3] = t_action[3] * alt_scale

                # STUDENT action -> this is what we FLY, so the states we visit
                # are the student's own distribution.
                image = gate_env.last_image
                telem = gate_env.last_telemetry
                # telem is TELEMETRY (pos, att, vel, imu): [0:3] pos, [3:6] att(deg),
                # [6:9] vel(world), [9:15] imu.
                if telem is not None and len(telem) >= 15:
                    imu = telem[9:15]
                    vel = telem[6:9]
                    yaw_deg = telem[5]
                else:
                    imu, vel, yaw_deg = [0.0] * 6, [0.0] * 3, 0.0
                s_yaw, s_pitch, s_roll, s_alt = pilot.run(image, *imu, *vel, yaw_deg)

                # Record BEFORE stepping: image/telemetry are the state the
                # teacher just labeled.
                if image is not None and telem is not None:
                    rows.append((image, float(label[0]), float(label[1]),
                                 float(label[2]), float(label[3]), "pilot", *telem))

                if args.beta > 0.0 and rng.random() < args.beta:
                    drive = t_action
                    n_teacher_steps += 1
                else:
                    # GateChaseEnv.step takes POLICY-space actions and rescales
                    # the altitude axis internally, so undo that scaling here.
                    drive = np.zeros(gate_env.n_actions, dtype=np.float32)
                    drive[:3] = (s_yaw, s_pitch, s_roll)
                    if control_alt:
                        drive[3] = np.clip(s_alt / alt_scale, -1.0, 1.0)
                    n_student_steps += 1

                obs, _reward, terminated, truncated, info = gate_env.step(drive)

            result = info.get("episode_result", "unknown")
            gates_passed = info.get("gates_passed", 0)
            keep = not (args.drop_crashed and result not in ("completed", "timeout"))
            if keep:
                for row in rows:
                    tub_writer.run(*row)
                kept += 1
            else:
                dropped += 1
            print(f"Episode {ep + 1}/{args.episodes}: {result}, "
                  f"gates {gates_passed}/{len(gates)}, {len(rows)} rows "
                  f"({'kept' if keep else 'dropped'})")
    finally:
        if tub_writer is not None:
            tub_writer.shutdown()
        drone_env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        drone_env.shutdown()
        update_thread.join(timeout=3.0)

    print(f"Episodes kept: {kept}, dropped: {dropped}")
    print(f"Steps flown by student: {n_student_steps}, by teacher: {n_teacher_steps}")
    if tub_writer is not None:
        print(f"Done: {tub_path}")


if __name__ == "__main__":
    main()
