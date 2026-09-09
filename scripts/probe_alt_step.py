"""Step response of the PD altitude hold: how well does it track a MOVING target?

The hold was sized on 2026-09-05b for the case that mattered then -- rejecting a
constant thrust bias about a FIXED setpoint -- and validated by a neutral-stick
hover that never moved the target. A varied-height course moves the target every
gate, and that is a different question entirely.

Lifts off, pins the setpoint at --start-alt, then steps the altitude trim between
+A and -A (metres, via DRONE_ALTITUDE_TRIM_RANGE) with the drone stationary, and
reports rise time, overshoot and the implied second-order w_n / zeta.

    uv run --env-file .env python scripts/probe_alt_step.py --amplitude=0.5

--gains sweeps several PD pairs against ONE stack and ONE takeoff, which is the
only affordable way to size the loop: a stack cycle costs ~40s and a takeoff ~25s,
against ~40s for a whole 3-leg gain pair. The gains are read fresh every control
cycle (env.altitude_hold_kp, tracker.k_pwm), so they can be swapped in flight.

    uv run --env-file .env python scripts/probe_alt_step.py \
        --amplitude=0.5 --gains=30:90,60:127,90:156,120:180

Hold zeta constant while raising bandwidth by scaling Kp with r**2 and K with r:
w_n = sqrt(s*Kp) and zeta = sqrt(s)*K/(2*sqrt(Kp)) for a thrust slope s. Doubling
BOTH gains does not hold damping -- it raises w_n by 1.41 and zeta by 1.41.
"""
import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, "/Users/Dan/dev/nanopilot-ai/nanopilot")

import donkeycar as dk
from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from rl_env import POS, VEL


def analyse(samples, z0, z_target, label):
    """samples: list of (t, z, bias). Reports overshoot and the implied w_n/zeta.

    `bias` is the UNCLIPPED PD term in PWM. It matters as much as the step shape:
    the command is clipped to +-200 before it reaches the mixer, and a saturated
    proportional loop is a relay, not a linear regulator -- so a clean-looking
    step that spent half its rise pinned at the clip is not evidence the gains
    are usable.
    """
    step = z_target - z0
    if abs(step) < 1e-6 or not samples:
        return
    zs = [z for _t, z, _b in samples]
    biases = [abs(b) for _t, _z, b in samples]
    peak_bias = max(biases)
    sat_frac = sum(1 for b in biases if b >= 199.5) / len(biases)
    peak = max(zs) if step > 0 else min(zs)
    overshoot = (peak - z_target) / step  # positive = past the target
    final = sum(zs[-10:]) / min(10, len(zs))
    # Time to first cross the target.
    t_cross = None
    for t, z, _b in samples:
        if (step > 0 and z >= z_target) or (step < 0 and z <= z_target):
            t_cross = t
            break
    line = (f"{label}: step {z0:.2f} -> {z_target:.2f} ({step:+.2f}m)  "
            f"peak {peak:.2f}  overshoot {overshoot * 100:+.0f}%  "
            f"final {final:.2f}  err {final - z_target:+.2f}m")
    line += f"  |bias|max {peak_bias:.0f}pwm sat {sat_frac * 100:.0f}%"
    if t_cross is not None:
        line += f"  t_cross {t_cross:.2f}s"
    print(line)
    if 0.0 < overshoot < 1.0:
        # Mp = exp(-pi*zeta/sqrt(1-zeta^2))
        lm = math.log(overshoot)
        zeta = -lm / math.sqrt(math.pi ** 2 + lm ** 2)
        if t_cross and t_cross > 0:
            # First crossing of a 2nd-order step is at t = (pi - acos(zeta))/wd
            wd = (math.pi - math.acos(zeta)) / t_cross
            wn = wd / math.sqrt(max(1e-6, 1 - zeta ** 2))
            print(f"        implied zeta={zeta:.2f}  w_n={wn:.2f} rad/s "
                  f"(settle ~{4 / (zeta * wn):.1f}s)")
        else:
            print(f"        implied zeta={zeta:.2f}")
    elif overshoot >= 1.0:
        print("        overshoot >= 100%: effectively undamped")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amplitude", type=float, default=0.5,
                   help="trim step in metres (must be <= DRONE_ALTITUDE_TRIM_RANGE)")
    p.add_argument("--hold-s", type=float, default=12.0)
    p.add_argument("--start-alt", type=float, default=1.5)
    p.add_argument("--settle-s", type=float, default=12.0,
                   help="seconds allowed to reach --start-alt before each gain "
                        "pair. The takeoff coasts well above 1.5m, so a LOW "
                        "start-alt needs far longer than the default: the hold "
                        "is soft (w_n ~0.46) and the descent is unforced.")
    p.add_argument("--airframe", default="80mm")
    p.add_argument("--gains", default=None,
                   help="comma-separated KP:K pairs to sweep in one flight, "
                        "e.g. 30:90,60:127. Default: the configured gains.")
    args = p.parse_args()

    world = os.environ.get("GZ_WORLD", "s_curve_heights_80mm")
    cfg = dk.load_config(
        config_path="/Users/Dan/dev/nanopilot-ai/nanopilot/config.py",
        myconfig=f"drone_config_{args.airframe}.py",
    )
    trim = getattr(cfg, "DRONE_ALTITUDE_TRIM_RANGE", 1.0)
    if args.gains:
        gains = [tuple(float(v) for v in pair.split(":"))
                 for pair in args.gains.split(",")]
    else:
        gains = [(cfg.DRONE_ALTITUDE_HOLD_KP, cfg.DRONE_ALTITUDE_HOLD_K)]
    print(f"world={world} HOLD_KP={cfg.DRONE_ALTITUDE_HOLD_KP} "
          f"HOLD_K={cfg.DRONE_ALTITUDE_HOLD_K} TRIM_RANGE={trim}")
    print("sweeping gains: " + "  ".join(f"Kp={kp:g}/K={k:g}" for kp, k in gains))

    _wait_for_tcp("127.0.0.1", 5761, 10, "BetaFlight MSP")
    env = build_drone_env(
        cfg, airframe=args.airframe, gz_world=world,
        record_position=True, record_attitude=True,
        record_velocity=True, record_imu=True,
    )
    threading.Thread(target=env.update, daemon=True).start()
    _wait_for_ready(env, 60)
    print(f"armed={env.bf_armed}")

    # Ramp to liftoff, then let the climb bleed off (same pattern as rl_env).
    climb = 0.2
    t0 = time.time()
    while time.time() - t0 < 15:
        out = env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=climb)
        if out[VEL][2] > 1.0:
            break
        climb = min(0.6, climb + 0.01)
        time.sleep(0.05)
    t0 = time.time()
    while time.time() - t0 < 8:
        out = env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=0.0)
        if abs(out[VEL][2]) < 0.15:
            break
        time.sleep(0.05)
    a = args.amplitude / trim
    tracker = env._pose_tracker

    for kp, k in gains:
        env.altitude_hold_kp = float(kp)
        tracker.k_pwm = float(k)
        print(f"\n=== Kp={kp:g} K={k:g} ===")

        # Re-settle on the fixed setpoint before every pair, so each ladder rung
        # starts from the same state rather than inheriting the last one's tail.
        env.set_altitude_setpoint(args.start_alt)
        t0 = time.time()
        while time.time() - t0 < args.settle_s:
            out = env.run_threaded(0.0, 0.0, 0.0, 0.0)
            if abs(out[POS][2] - args.start_alt) < 0.1 and abs(out[VEL][2]) < 0.1:
                break
            time.sleep(0.05)
        print(f"settled at z={out[POS][2]:.2f} vz={out[VEL][2]:.2f}")

        for label, cmd in (("UP", a), ("DOWN", -a), ("BACK", 0.0)):
            z0 = env.run_threaded(0.0, 0.0, 0.0, 0.0)[POS][2]
            target = args.start_alt + cmd * trim
            samples = []
            t0 = time.time()
            while time.time() - t0 < args.hold_s:
                out = env.run_threaded(0.0, 0.0, 0.0, cmd)
                t = time.time() - t0
                # Mirror drone_gym's PD term, UNCLIPPED, to see how much of the
                # response was actually spent against the +-200 clip.
                bias = -kp * (out[POS][2] - target) - k * tracker.get_vz()
                samples.append((t, out[POS][2], bias))
                if len(samples) % 20 == 1:
                    print(f"  {label} t={t:5.1f}s z={out[POS][2]:6.2f} "
                          f"vz={out[VEL][2]:+6.2f} thr={env.last_throttle_pwm} "
                          f"bias={bias:+7.1f}")
                time.sleep(0.05)
            analyse(samples, z0, target, label)

    env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
    time.sleep(0.5)
    env.shutdown()


if __name__ == "__main__":
    main()
