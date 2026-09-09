"""Descent authority of the PD altitude hold, and a ground-effect check.

Stage 0 of the 2026-09-09 landing plan. `probe_alt_step.py` answers "how fast
does the hold reach a MOVED target" with the drone stationary about a FIXED
setpoint; a landing asks two different questions it cannot answer:

  1. How fast can the drone be commanded DOWN before the loop stops tracking,
     and what runs out first -- the +-200 clip on `alt_hold_bias`, or the
     [-1, 1] clip on the altitude command itself?
  2. Does thrust demand change near the ground (ground effect)? The Gazebo
     plugin models no vertical drag, so probably not -- but "probably" is how
     2026-09-08 lost a session.

The descent legs drive the ENV THROUGH THE EXPERT'S OWN HEIGHT LAW,

    altitude = ((z_ref - setpoint) + Kz * (z_ref - z)) / TRIM_RANGE

with `z_ref` ramping down at a commanded rate, because that is exactly what a
`LineExpert` will do and it is the only way to see the command clip bite. A
raw step to alt=-1 measures the plant; a ramp measures the interface.

Why a ramp cannot be free: the sim has no vertical drag, so holding a constant
descent rate needs ZERO net force, i.e. `alt_hold_bias ~ 0`. With
`bias = -Kp*(z - z_target) - K*vz` that means the drone must fly a STEADY LAG
above its own target to sustain rate r. Closing Kz around it,

    lag  e = (K/Kp) * r / (1 + Kz)          = 1.71 m per m/s at Kp=30 K=90 Kz=0.75
    cmd |alt| * TRIM = D + (K/Kp)*Kz*r/(1+Kz)   for a descent of depth D

so BOTH the depth descended and the rate are spent out of the same trim range,
and the command clips at -1 when their sum exceeds it. That prediction is what
these legs test, and it is the arithmetic behind the plan's TRIM_RANGE = 1.6.

A leg therefore descends a bounded DEPTH (--depth), not down to a fixed floor:
with the setpoint frozen at start_alt the feed-forward term alone would clip
the command after TRIM_RANGE metres of descent no matter how slowly it was
flown, which would measure the range and tell us nothing about the rate.

    GZ_HEADLESS=1 GZ_WORLD=landing_target_80mm bash scripts/start_gated.sh \
        --no-manage --airframe=80mm
    uv run --env-file .env python -u scripts/probe_descent.py \
        --mode=all --trim-range=1.0,1.6

--trim-range sweeps the value IN FLIGHT (the env reads it every control cycle),
so the descent authority of the proposed TRIM_RANGE=1.6 can be measured without
changing the config -- and therefore without changing the plant for the gate
courses, which is a separate, gated step.
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, "/Users/Dan/dev/nanopilot-ai/nanopilot")

import donkeycar as dk
from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from rl_env import POS, VEL

RATE_HZ = 20.0
DT = 1.0 / RATE_HZ


class Probe:
    """Thin wrapper over the env: one control cycle, one sample."""

    def __init__(self, env, cfg):
        self.env = env
        self.tracker = env._pose_tracker
        self.kp = float(cfg.DRONE_ALTITUDE_HOLD_KP)
        self.k = float(cfg.DRONE_ALTITUDE_HOLD_K)

    def step(self, alt=0.0, direct_thrust=None):
        """One cycle. Returns a sample dict with the UNCLIPPED PD term.

        `alt_raw` is what the control law asked for; the env clips it to
        [-1, 1] before it becomes a height target, so a leg where |alt_raw| > 1
        is one where the interface, not the plant, is the limit.
        """
        alt_clipped = max(-1.0, min(1.0, alt))
        out = self.env.run_threaded(0.0, 0.0, 0.0, alt_clipped,
                                    direct_thrust=direct_thrust)
        z = out[POS][2]
        vz_hold = self.tracker.get_vz()      # what the hold actually differentiates
        setpoint = self.env.get_altitude_setpoint()
        z_target = (0.0 if setpoint is None else setpoint) \
            + alt_clipped * self.env.altitude_trim_range
        bias = -self.kp * (z - z_target) - self.k * vz_hold
        return {
            "z": z,
            "vz": out[VEL][2],          # filtered, for reporting
            "vz_hold": vz_hold,         # raw diff, what the loop sees
            "alt_raw": alt,
            "alt": alt_clipped,
            "z_target": z_target,
            "bias": bias,
            "thr": self.env.last_throttle_pwm,
        }

    def liftoff(self, target=2.0, timeout=15.0):
        """Ramp on direct thrust until rising, then let the climb bleed off.

        Same shape as rl_env._takeoff, INCLUDING its height cutoff: the
        filtered vz lags the real climb badly enough that a vz-only test let
        the drone coast to 11m on the first attempt here, which then costs a
        very long unforced descent before any leg can start.
        """
        climb = 0.2
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.step(direct_thrust=climb)
            if s["vz"] > 1.0 or s["z"] >= target:
                break
            climb = min(0.6, climb + 0.01)
            time.sleep(DT)
        # Arrest on the HOLD, not on direct_thrust=0.0. A direct thrust of 0.0
        # is not "no command" -- it pins the throttle at exactly hover and
        # suppresses alt_hold_bias entirely, and with no vertical drag in the
        # sim the drone then coasts upward forever on its climb momentum. That
        # is how the first run of this probe reached 11m before any leg began.
        # (scripts/probe_alt_step.py still has this bug in its takeoff.)
        t0 = time.time()
        while time.time() - t0 < 8.0:
            s = self.step()
            if abs(s["vz"]) < 0.15:
                break
            time.sleep(DT)
        return s

    def settle(self, z, timeout=40.0, tol=0.08, quiet=0.06):
        """Park at height z on the frozen setpoint.

        Lifts off first if the drone is sitting on the ground -- a descent leg
        can legitimately end there, and no altitude command recovers from it.
        """
        s = self.step()
        if s["z"] < 0.25 and abs(s["vz"]) < 0.1 and z > 0.5:
            self.liftoff(target=z)
        self.env.set_altitude_setpoint(float(z))
        t0 = time.time()
        stable = 0
        while time.time() - t0 < timeout:
            s = self.step()
            if abs(s["z"] - z) < tol and abs(s["vz"]) < quiet:
                stable += 1
                if stable >= 10:
                    break
            else:
                stable = 0
            time.sleep(DT)
        return s


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ground_effect(probe, heights, dwell_s):
    """Steady-state thrust demand at a ladder of heights.

    With no ground effect the hold settles the same distance off its setpoint
    at every height, because that offset is purely `residual_bias / Kp`. Extra
    lift near the ground would show up as a settling point ABOVE the setpoint
    and a lower mean throttle -- both measured here, since either alone could
    be read as drift.
    """
    print("\n=== GROUND EFFECT: steady-state thrust vs height ===")
    print(f"{'setpoint':>9} {'z_ss':>7} {'offset':>8} {'thr_pwm':>8} "
          f"{'bias':>7} {'vz':>7}")
    rows = []
    for h in heights:
        probe.settle(h, timeout=45.0)
        samples = []
        t0 = time.time()
        while time.time() - t0 < dwell_s:
            samples.append(probe.step())
            time.sleep(DT)
        z_ss = mean([s["z"] for s in samples])
        thr = mean([s["thr"] for s in samples])
        bias = mean([s["bias"] for s in samples])
        vz = mean([s["vz"] for s in samples])
        rows.append((h, z_ss, thr))
        print(f"{h:9.2f} {z_ss:7.3f} {z_ss - h:+8.3f} {thr:8.1f} "
              f"{bias:+7.1f} {vz:+7.3f}")
    if len(rows) > 1:
        offs = [z - h for h, z, _ in rows]
        thrs = [t for _h, _z, t in rows]
        print(f"offset spread {max(offs) - min(offs):.3f}m   "
              f"throttle spread {max(thrs) - min(thrs):.1f} pwm")
        print("VERDICT: " + (
            "no measurable ground effect"
            if max(thrs) - min(thrs) < 5.0
            else "thrust demand VARIES with height -- investigate before landing"))
    return rows


def descent_ramp(probe, rate, trim, kz, start_alt, floor, settle_first=True):
    """One descent leg: z_ref ramps down at `rate` through the expert's law.

    Reports the achieved rate against the commanded one, the steady lag, and
    which of the two clips (command or PD bias) was actually reached.

    The ramp must last several time constants (1/w_n ~ 2.1s) or the lag never
    develops and the leg measures a transient. A leg is scored only over
    t > 2s, and one whose ramp is shorter than MIN_RAMP_S is reported as
    inconclusive rather than as a low achieved rate -- the first run of this
    probe reported "0% achieved" at 1.0 m/s purely because a 0.6m ramp at
    1.0 m/s is over in 0.6 seconds.
    """
    MIN_RAMP_S = 4.0
    if settle_first:
        probe.settle(start_alt)
    setpoint = probe.env.get_altitude_setpoint()
    probe.env.altitude_trim_range = float(trim)
    samples = []
    t0 = time.time()
    z_ref = start_alt
    s = probe.step()              # current state; the law commands on it
    while z_ref > floor:
        t = time.time() - t0
        z_ref = start_alt - rate * t
        alt = ((z_ref - setpoint) + kz * (z_ref - s["z"])) / trim
        s = probe.step(alt=alt)
        s["t"] = t
        s["z_ref"] = z_ref
        samples.append(s)
        if t > (start_alt - floor) / rate + 15.0:
            break
        time.sleep(DT)

    ramp_s = (start_alt - floor) / rate
    # Steady window: skip the first 2s of transient.
    steady = [s for s in samples if s["t"] > 2.0]
    if ramp_s < MIN_RAMP_S or not steady:
        print(f"  rate {rate:4.2f}  trim {trim:3.1f}  INCONCLUSIVE: ramp lasts "
              f"{ramp_s:.1f}s, under the {MIN_RAMP_S:.0f}s needed for the lag to "
              f"develop. Increase --depth or lower the rate.")
        return None
    span = steady[-1]["t"] - steady[0]["t"]
    achieved = (steady[0]["z"] - steady[-1]["z"]) / span if span > 0 else 0.0
    lag = [s["z"] - s["z_ref"] for s in steady]
    clip_frac = sum(1 for s in steady if abs(s["alt_raw"]) > 1.0) / len(steady)
    sat_frac = sum(1 for s in steady if abs(s["bias"]) >= 199.5) / len(steady)
    print(f"  rate {rate:4.2f}  trim {trim:3.1f}  achieved {achieved:5.2f} m/s "
          f"({achieved / rate * 100:3.0f}%)  lag {mean(lag):+5.2f}/"
          f"{max(lag, key=abs):+5.2f}m  |alt_raw|max {max(abs(s['alt_raw']) for s in steady):4.2f} "
          f"clip {clip_frac * 100:3.0f}%  |bias|max {max(abs(s['bias']) for s in steady):5.0f} "
          f"sat {sat_frac * 100:3.0f}%  z_end {samples[-1]['z']:.2f}")
    return {
        "rate": rate, "trim": trim, "achieved": achieved,
        "lag_mean": mean(lag), "clip_frac": clip_frac, "sat_frac": sat_frac,
        "z_end": samples[-1]["z"],
    }


def terminal_descent(probe, trim, start_alt, max_s=20.0):
    """Full-down command against a frozen setpoint: the plant's descent limit.

    This is the fastest the drone can be asked to come down through the
    interface, and it is what a saturating landing degenerates into.
    """
    probe.settle(start_alt)
    probe.env.altitude_trim_range = float(trim)
    samples = []
    t0 = time.time()
    while time.time() - t0 < max_s:
        s = probe.step(alt=-1.0)
        s["t"] = time.time() - t0
        samples.append(s)
        if s["z"] < 0.12:
            break
        time.sleep(DT)
    vzs = [s["vz"] for s in samples]
    peak = min(vzs)
    late = [s for s in samples if s["t"] > 1.5]
    print(f"  trim {trim:3.1f}  target {start_alt - trim:+5.2f}m  "
          f"peak vz {peak:+5.2f} m/s  z_end {samples[-1]['z']:5.2f} "
          f"t_end {samples[-1]['t']:5.1f}s  "
          f"|bias|max {max(abs(s['bias']) for s in samples):5.0f}  "
          f"sat {sum(1 for s in late if abs(s['bias']) >= 199.5) / max(1, len(late)) * 100:3.0f}%")
    return {"trim": trim, "peak_vz": peak, "z_end": samples[-1]["z"]}


def landing_profile(probe, trim, cruise, max_s=45.0, touchdown_z=0.06):
    """Full-down command from cruise height: the FASTEST landing the interface
    can express, and therefore the one that sizes TRIM_RANGE.

    The altitude axis commands a target HEIGHT, never a rate, so the quickest
    descent available is `alt = -1` -- aiming at `setpoint - TRIM_RANGE`. Any
    ramped z_ref is strictly slower. Descent speed is therefore set by how far
    BELOW GROUND that target can be placed, which is a pure function of
    TRIM_RANGE, and the drone decelerates as it approaches the target: a target
    just below ground lands softly but slowly, a deeper one lands faster and
    harder. This leg measures both ends of that trade.
    """
    probe.settle(cruise)
    probe.env.altitude_trim_range = float(trim)
    target = cruise - trim
    samples = []
    t0 = time.time()
    landed_at = None
    while time.time() - t0 < max_s:
        s = probe.step(alt=-1.0)
        s["t"] = time.time() - t0
        samples.append(s)
        if landed_at is None and s["z"] < touchdown_z:
            landed_at = s
            break
        time.sleep(DT)

    def first_below(z_mark):
        for s in samples:
            if s["z"] < z_mark:
                return s
        return None

    marks = []
    for z_mark in (cruise - (cruise - 0.0) / 2.0, 0.50, 0.20):
        s = first_below(z_mark)
        marks.append(f"z<{z_mark:.2f}@{s['t']:5.1f}s" if s else f"z<{z_mark:.2f}: NEVER")
    peak = min(s["vz"] for s in samples)
    td = ("touchdown %5.1fs vz %+.2f" % (landed_at["t"], landed_at["vz"])
          if landed_at else
          "NO TOUCHDOWN (z_end %.2f)" % samples[-1]["z"])
    print(f"  trim {trim:3.1f}  target {target:+5.2f}m  " + "  ".join(marks)
          + f"  peak vz {peak:+5.2f}  {td}"
          + f"  |bias|max {max(abs(s['bias']) for s in samples):4.0f}")
    return {"trim": trim, "peak_vz": peak,
            "touchdown_s": landed_at["t"] if landed_at else None,
            "touchdown_vz": landed_at["vz"] if landed_at else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="all",
                   choices=["all", "ground", "ramp", "terminal", "landing"])
    p.add_argument("--start-alt", type=float, default=2.0,
                   help="height each descent leg starts from")
    p.add_argument("--depth", type=float, default=0.6,
                   help="metres of descent per ramp leg. Kept well inside "
                        "TRIM_RANGE so the command clips (if at all) because of "
                        "the RATE, not because the frozen setpoint ran out of "
                        "range -- see the module docstring.")
    p.add_argument("--rates", default="0.15,0.3,0.5,0.75,1.0",
                   help="commanded descent rates, m/s")
    p.add_argument("--trim-range", default="1.0",
                   help="TRIM_RANGE values to sweep in flight (config is not "
                        "touched). 1.0 is current; 1.6 is the landing plan's.")
    p.add_argument("--height-gain", type=float, default=0.75,
                   help="Kz, matching GateExpert's validated default")
    p.add_argument("--ground-heights", default="1.50,1.00,0.70,0.50,0.35,0.25")
    p.add_argument("--dwell-s", type=float, default=5.0)
    p.add_argument("--cruise-alt", type=float, default=1.5,
                   help="height a landing leg starts from; the frozen hold "
                        "setpoint sits here, so TRIM_RANGE is measured from it")
    p.add_argument("--airframe", default="80mm")
    args = p.parse_args()

    world = os.environ.get("GZ_WORLD", "landing_target_80mm")
    cfg = dk.load_config(
        config_path="/Users/Dan/dev/nanopilot-ai/nanopilot/config.py",
        myconfig=f"drone_config_{args.airframe}.py",
    )
    kp = cfg.DRONE_ALTITUDE_HOLD_KP
    k = cfg.DRONE_ALTITUDE_HOLD_K
    cfg_trim = getattr(cfg, "DRONE_ALTITUDE_TRIM_RANGE", 1.0)
    rates = [float(v) for v in args.rates.split(",")]
    trims = [float(v) for v in args.trim_range.split(",")]
    print(f"world={world} HOLD_KP={kp} HOLD_K={k} TRIM_RANGE(config)={cfg_trim} "
          f"Kz={args.height_gain}")
    lag_per_ms = (k / kp) / (1.0 + args.height_gain)
    print(f"predicted lag {lag_per_ms:.2f} m per m/s of descent "
          f"(= (K/Kp)/(1+Kz))")
    for t in trims:
        r_clip = (t - args.depth) / ((k / kp) * args.height_gain
                                     / (1.0 + args.height_gain))
        print(f"predicted command clip at trim {t:.1f}, depth {args.depth:.2f}m: "
              f"rate > {r_clip:.2f} m/s")

    _wait_for_tcp("127.0.0.1", 5761, 10, "BetaFlight MSP")
    env = build_drone_env(
        cfg, airframe=args.airframe, gz_world=world,
        record_position=True, record_attitude=True,
        record_velocity=True, record_imu=True,
    )
    threading.Thread(target=env.update, daemon=True).start()
    _wait_for_ready(env, 60)
    print(f"armed={env.bf_armed}")

    probe = Probe(env, cfg)
    s = probe.liftoff(target=args.start_alt)
    print(f"airborne at z={s['z']:.2f} vz={s['vz']:+.2f}")

    try:
        if args.mode in ("all", "ground"):
            ground_effect(probe, [float(v) for v in args.ground_heights.split(",")],
                          args.dwell_s)
        if args.mode in ("all", "ramp"):
            print("\n=== DESCENT RAMPS (expert height law, z_ref ramping down) ===")
            for trim in trims:
                for rate in rates:
                    descent_ramp(probe, rate, trim, args.height_gain,
                                 args.start_alt, args.start_alt - args.depth)
        if args.mode in ("all", "landing"):
            print("\n=== LANDING PROFILE (alt = -1.0 from cruise to ground) ===")
            for trim in trims:
                landing_profile(probe, trim, args.cruise_alt)
        if args.mode in ("all", "terminal"):
            print("\n=== TERMINAL DESCENT (alt = -1.0, frozen setpoint) ===")
            for trim in trims:
                terminal_descent(probe, trim, args.start_alt)
    finally:
        env.altitude_trim_range = float(cfg_trim)
        env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
        time.sleep(0.5)
        env.shutdown()


if __name__ == "__main__":
    main()
