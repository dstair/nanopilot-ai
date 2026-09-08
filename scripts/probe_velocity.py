"""Step-response probe for the velocity cascade (DRONE_VELOCITY_CONTROL).

Takes off, then commands a sequence of speed setpoints and logs commanded vs
measured heading-frame velocity. This is where DRONE_VELOCITY_KP / _KFF get
tuned, and where the SIGN of each axis is proved rather than assumed.

The yaw leg matters: it repeats the forward step after turning ~45 degrees, so
a world-frame/heading-frame rotation error shows up as cross-axis leakage that
the un-yawed legs cannot reveal.

Requires a running stack:

    GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh \
        --no-manage --airframe=80mm
    uv run --env-file .env python scripts/probe_velocity.py

Pass criteria (printed as a verdict per leg):
  * correct sign on both axes
  * settling within 2.0s
  * steady-state error under 15%
  * height held within 0.3m of the takeoff setpoint
"""
import argparse
import math
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "nanopilot"))

import donkeycar as dk

from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from rl_env import POS, VEL, ATT

SETTLE_S = 2.0
SS_ERR_FRAC = 0.15
Z_TOL = 0.3


def heading_frame(vel, yaw_deg):
    """World velocity -> (forward, rightward), matching the env cascade."""
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    fwd = vel[0] * cy + vel[1] * sy
    left = -vel[0] * sy + vel[1] * cy
    return fwd, -left


def takeoff(env, args):
    """Ramped climb, cut at liftoff, then anchor the hold at start_alt.

    Mirrors rl_env._takeoff/_arrest_climb: the throttle band between "won't
    leave the ground" and "rockets away" is narrow, so ramp and cut on vz.
    """
    deadline = time.time() + args.takeoff_timeout
    climb = 0.2
    while time.time() < deadline:
        out = env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=climb)
        if out[VEL][2] > 1.0 or out[POS][2] >= args.start_alt:
            break
        climb = min(0.6, climb + 0.01)
        time.sleep(0.05)
    else:
        return False

    # Arrest the climb. Deliberately NO height requirement here: on the PD plant
    # the hold freezes its setpoint wherever the climb stops, which after a
    # single ramp off the ground is only ~0.3m. Height is established below by
    # commanding the setpoint, which is what the position term exists for.
    calm = 0
    while time.time() < deadline:
        out = env.run_threaded(0.0, 0.0, 0.0, 0.0)
        if abs(out[VEL][2]) < 0.5:
            calm += 1
            if calm >= 4:
                break
        else:
            calm = 0
        time.sleep(0.05)
    else:
        return False

    if getattr(env, "altitude_hold_kp", 0.0) <= 0.0:
        raise SystemExit("This probe needs DRONE_ALTITUDE_HOLD_KP > 0")

    # Fly to the start height under the hold. The loop is ~0.55 rad/s, so allow
    # several seconds to cover a metre.
    env.set_altitude_setpoint(args.start_alt)
    climb_deadline = time.time() + args.climb_timeout
    while time.time() < climb_deadline:
        o = env.run_threaded(0.0, 0.0, 0.0, 0.0)
        if abs(o[POS][2] - args.start_alt) < 0.15 and abs(o[VEL][2]) < 0.15:
            break
        time.sleep(0.05)
    o = env.run_threaded(0.0, 0.0, 0.0, 0.0)
    print(f"Takeoff settled at z={o[POS][2]:.2f} vz={o[VEL][2]:+.2f} "
          f"(target {args.start_alt})")
    return True


def run_leg(env, name, fwd_cmd, right_cmd, yaw_cmd, seconds, rate_hz, args):
    """Hold one setpoint and report the response."""
    period = 1.0 / rate_hz
    t0 = time.time()
    samples = []
    while time.time() - t0 < seconds:
        out = env.run_threaded(yaw_cmd, fwd_cmd, right_cmd, 0.0)
        t = time.time() - t0
        f, r = heading_frame(out[VEL], out[ATT][2])
        sat = max(abs(env.last_pitch_cmd), abs(env.last_roll_cmd)) >= 0.999
        samples.append((t, f, r, out[POS][2], out[ATT][2], sat))
        time.sleep(period)

    tgt_f = fwd_cmd * env.max_forward_speed
    tgt_r = right_cmd * env.max_lateral_speed
    # Steady state = last third of the leg.
    tail = samples[int(len(samples) * 2 / 3):]
    ss_f = sum(s[1] for s in tail) / len(tail)
    ss_r = sum(s[2] for s in tail) / len(tail)
    zs = [s[3] for s in samples]

    # Rise time: first sample reaching 90% of target on the driven axis.
    driven, tgt, ss = ("fwd", tgt_f, ss_f) if abs(tgt_f) >= abs(tgt_r) else ("right", tgt_r, ss_r)
    idx = 1 if driven == "fwd" else 2
    t_rise = None
    if abs(tgt) > 1e-6:
        for s in samples:
            if s[idx] / tgt >= 0.9:
                t_rise = s[0]
                break

    print(f"\n--- {name} ---")
    print(f"  command      fwd={tgt_f:+.2f}  right={tgt_r:+.2f} m/s  yaw_stick={yaw_cmd:+.2f}")
    print(f"  steady state fwd={ss_f:+.2f}  right={ss_r:+.2f} m/s")
    print(f"  yaw {samples[0][4]:+.1f} -> {samples[-1][4]:+.1f} deg")
    print(f"  z {min(zs):.2f}-{max(zs):.2f} (start {samples[0][3]:.2f})")
    print(f"  sticks pitch={env.last_pitch_cmd:+.2f} roll={env.last_roll_cmd:+.2f}")

    verdict = []
    if abs(tgt) > 1e-6:
        if ss * tgt <= 0:
            verdict.append(f"FAIL sign: commanded {tgt:+.2f}, measured {ss:+.2f}")
        else:
            err = abs(ss - tgt) / abs(tgt)
            verdict.append(
                ("ok" if err <= SS_ERR_FRAC else "FAIL")
                + f" steady-state error {err * 100:.0f}%"
            )
            if t_rise is None:
                verdict.append(f"FAIL never reached 90% of {tgt:+.2f}")
            else:
                # A slow rise while the stick is pinned at full deflection is an
                # AUTHORITY limit, not a gain that needs raising -- more gain
                # cannot move a saturated actuator. Roll is capped at 10 degrees
                # against pitch's 20, so the lateral axis hits this first.
                sat_frac = sum(1 for s in samples if s[5]) / len(samples)
                if t_rise <= SETTLE_S:
                    verdict.append(f"ok rise {t_rise:.2f}s")
                elif sat_frac > 0.1:
                    verdict.append(
                        f"note rise {t_rise:.2f}s, stick saturated "
                        f"{sat_frac * 100:.0f}% of the leg (authority-limited, "
                        f"raising gain will not help)"
                    )
                else:
                    verdict.append(f"FAIL rise {t_rise:.2f}s")
        # Cross-axis leakage: the undriven axis should stay near zero.
        cross = ss_r if driven == "fwd" else ss_f
        verdict.append(
            ("ok" if abs(cross) < 0.3 else "FAIL")
            + f" cross-axis {cross:+.2f} m/s"
        )
    dz = max(abs(z - samples[0][3]) for z in zs)
    verdict.append(("ok" if dz <= Z_TOL else "FAIL") + f" height drift {dz:.2f}m")
    for v in verdict:
        print(f"    {v}")
    return [v for v in verdict if v.startswith("FAIL")]


def yaw_rate_leg(env, steer, seconds, rate_hz):
    """Unwrapped yaw rate for one stick value.

    MUST unwrap and sample densely. Reading only the endpoints of a leg cannot
    tell +119 degrees from -241, and that exact aliasing put the wrong sign in
    the gate expert on 2026-09-06 (it flew the whole course facing backwards).
    """
    period = 1.0 / rate_hz
    t0 = time.time()
    prev = None
    total = 0.0
    while time.time() - t0 < seconds:
        out = env.run_threaded(steer, 0.0, 0.0, 0.0)
        y = out[ATT][2]
        if prev is not None:
            total += (y - prev + 180.0) % 360.0 - 180.0
        prev = y
        time.sleep(period)
    elapsed = time.time() - t0
    rate = total / elapsed
    print(f"  steer={steer:+.2f}: {total:+8.1f} deg over {elapsed:.1f}s "
          f"=> {rate:+7.1f} deg/s")
    return rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="slalom_gates_80mm")
    p.add_argument("--airframe", default="80mm")
    p.add_argument("--rate-hz", type=float, default=20.0)
    p.add_argument("--start-alt", type=float, default=1.5)
    p.add_argument("--takeoff-timeout", type=float, default=20.0)
    p.add_argument("--climb-timeout", type=float, default=20.0)
    p.add_argument("--step-s", type=float, default=5.0)
    p.add_argument("--rest-s", type=float, default=3.0)
    p.add_argument("--speed", type=float, default=1.0,
                   help="commanded speed in m/s for each step leg")
    args = p.parse_args()

    import os
    os.environ["GZ_WORLD"] = args.world

    cfg = dk.load_config(
        config_path=str(pathlib.Path(__file__).resolve().parent.parent
                        / "nanopilot" / "config.py"),
        myconfig=f"drone_config_{args.airframe}.py",
    )
    _wait_for_tcp("127.0.0.1", 5761, 10, "BetaFlight MSP")
    env = build_drone_env(
        cfg, airframe=args.airframe, gz_world=args.world,
        record_position=True, record_attitude=True,
        record_velocity=True, record_imu=True,
    )
    if not env.velocity_control:
        raise SystemExit(
            "DRONE_VELOCITY_CONTROL is off; this probe measures the velocity "
            "cascade and has nothing to measure without it."
        )
    print(f"cascade: fwd KP={env.velocity_kp_fwd} KFF={env.velocity_kff_fwd} | "
          f"lat KP={env.velocity_kp_lat} KFF={env.velocity_kff_lat} | "
          f"tau={env.velocity_tau} max_fwd={env.max_forward_speed} "
          f"max_lat={env.max_lateral_speed}")

    threading.Thread(target=env.update, daemon=True).start()
    _wait_for_ready(env, 30)

    failures = []
    try:
        if not takeoff(env, args):
            raise SystemExit("Takeoff failed; cannot probe velocity")

        f_cmd = args.speed / env.max_forward_speed
        r_cmd = args.speed / env.max_lateral_speed

        legs = [
            ("forward", f_cmd, 0.0, 0.0, args.step_s),
            ("rest", 0.0, 0.0, 0.0, args.rest_s),
            ("right", 0.0, r_cmd, 0.0, args.step_s),
            ("rest", 0.0, 0.0, 0.0, args.rest_s),
            ("left", 0.0, -r_cmd, 0.0, args.step_s),
            ("rest", 0.0, 0.0, 0.0, args.rest_s),
            # Yaw, then repeat forward. A heading-frame rotation error is
            # invisible at yaw=0 and shows up here as cross-axis leakage.
            ("yaw turn", 0.0, 0.0, 0.5, 2.0),
            ("forward after yaw", f_cmd, 0.0, 0.0, args.step_s),
            ("rest", 0.0, 0.0, 0.0, args.rest_s),
        ]
        for name, f, r, y, secs in legs:
            failures += run_leg(env, name, f, r, y, secs, args.rate_hz, args)

        print("\n--- yaw rate vs stick (positive rate = ENU yaw increasing "
              "= turning LEFT) ---")
        rates = {}
        for st in (0.25, 0.5, 1.0, -0.5):
            rates[st] = yaw_rate_leg(env, st, 3.0, args.rate_hz)
            yaw_rate_leg(env, 0.0, 1.0, args.rate_hz)
        full = abs(rates[1.0])
        print(f"  full-stick yaw rate {full:.0f} deg/s vs configured "
              f"DRONE_MAX_YAW_RATE={env.max_yaw_rate:.0f}")
        if rates[0.5] > 0:
            failures.append(
                "FAIL yaw sign: a POSITIVE steer increased ENU yaw; "
                "GateExpert.YAW_STICK_SIGN assumes it decreases it"
            )
        if full > 1.5 * env.max_yaw_rate:
            failures.append(
                f"FAIL full-stick yaw {full:.0f} deg/s is far above the "
                f"configured {env.max_yaw_rate:.0f}; _YAW_DEG_PER_SEC_PER_PWM "
                f"is mis-calibrated"
            )
    finally:
        env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        env.shutdown()

    print("\n=== verdict ===")
    if failures:
        print(f"{len(failures)} failing checks:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
