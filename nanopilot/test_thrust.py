#!/usr/bin/env python3
"""
Thrust / yaw test: validate BF SITL + Gazebo behavior.

Modes (`--mode`):
    thrust   Ramp throttle 1000→2000 PWM, 0.5s per step. Used to find
             hover PWM and tune `motorConstant` in model.sdf.
    yaw      Hold throttle=1000 (motors should be idle) and apply a
             full yaw stick. Watch for altitude climb — that's the
             "yaw shoots drone into sky" bug. With AIRMODE disabled
             in BF, altitude should stay ~0.
    yaw-airborne
             Climb to altitude first, then apply yaw at hover throttle,
             then cut CH3 to 1000 while keeping yaw. Checks the user-
             reported bug: "after yaw input, drone climbs and won't
             come down even with throttle at 1000."
    hover    Fine sweep near hover (default 1450-1550, 5 PWM steps,
             2s per step) with altitude rate per PWM. Pick the PWM
             closest to 0 m/s rate as your DRONE_HOVER_THROTTLE.
    lateral-coast
             Climb, apply sideways roll for a few seconds, level roll, then
             sample XY speed decay and drift distance. Used to quantify the
             "skating rink" / missing translational drag effect.
    both     Run yaw test first, then thrust ramp (default).

Easiest way to run (brings up the stack, runs the test, tears it down):

    ./scripts/test_thrust.sh
    ./scripts/test_thrust.sh --mode=yaw

This test sends its OWN RC packets to BetaFlight SITL on UDP 9004, so
drone_manage.py must NOT be running — the two would fight over the port.
"""

import argparse
import logging
import os
import socket
import struct
import time
from utils.betaflight import pack_rc_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC_HOST = "127.0.0.1"
RC_PORT = 9004

# World name defaults to the 65mm airframe; test_thrust.sh exports GZ_WORLD
# when --airframe is passed. Pose topic must match the loaded world.
_GZ_WORLD = os.environ.get("GZ_WORLD", "drone_course_65mm")
POSE_TOPIC = f"/world/{_GZ_WORLD}/dynamic_pose/info"


def send_rc(rc_sock, channels):
    """Send 40-byte RC packet: timestamp + 16 x uint16."""
    rc_sock.sendto(pack_rc_frame(channels), (RC_HOST, RC_PORT))


FLIGHT_MODE_ANGLE = True  # set False for Acro (CH6 = 1000)


def make_channels(
    throttle_pwm, armed=True, yaw_pwm=1500, pitch_pwm=1500, roll_pwm=1500
):
    """Create 16-channel RC packet with throttle, roll/yaw/pitch, and arm/disarm."""
    channels = [1000] * 16
    channels[0] = roll_pwm  # CH1 roll
    channels[1] = pitch_pwm  # CH2 pitch
    channels[2] = throttle_pwm  # CH3 throttle
    channels[3] = yaw_pwm  # CH4 yaw
    channels[4] = 2000 if armed else 1000  # CH5 AUX1 arm switch
    channels[5] = 2000 if FLIGHT_MODE_ANGLE else 1000  # CH6 AUX2 angle/acro
    return channels


_pose_state = {"node": None, "latest": None, "latest_quat": None}


def _init_pose_subscriber():
    """Subscribe once at startup; the callback keeps the latest drone pose in
    a shared slot. Previous per-call subscription missed most samples because
    the 150ms settle wasn't long enough for a fresh subscription to receive.
    """
    if _pose_state["node"] is not None:
        return True
    try:
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
    except Exception as e:
        logger.warning("gz-transport not available: %s", e)
        return False

    node = Node()

    def on_pose(msg):
        for p in msg.pose:
            if "betaloop" in p.name.lower():
                _pose_state["latest"] = (p.position.x, p.position.y, p.position.z)
                q = p.orientation
                _pose_state["latest_quat"] = (q.w, q.x, q.y, q.z)
                return

    if not node.subscribe(Pose_V, POSE_TOPIC, on_pose):
        logger.warning("Could not subscribe to %s", POSE_TOPIC)
        return False

    _pose_state["node"] = node
    return True


def query_pose():
    """Return latest drone pose (x, y, z) or None if unknown."""
    return _pose_state["latest"]


def query_attitude_deg():
    """Return latest drone attitude (roll, pitch, yaw) in degrees, or None."""
    q = _pose_state["latest_quat"]
    if q is None:
        return None
    w, x, y, z = q
    # Standard ZYX quaternion → Euler conversion
    import math as _m
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = _m.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = _m.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = _m.atan2(siny_cosp, cosy_cosp)
    return (_m.degrees(roll), _m.degrees(pitch), _m.degrees(yaw))


def hold(rc_sock, channels, duration_s):
    """Send channels at 50Hz for duration_s seconds."""
    n = int(duration_s / 0.02)
    for _ in range(n):
        send_rc(rc_sock, channels)
        time.sleep(0.02)


def arm_sequence(rc_sock):
    """Disarm→arm boot sequence BF SITL requires to clear NOT_DISARMED flag."""
    logger.info("Sending disarm packets (1s)...")
    hold(rc_sock, make_channels(1000, armed=False), 1.0)
    logger.info("Arming and idling...")
    hold(rc_sock, make_channels(1000, armed=True), 1.0)


def run_yaw_test(rc_sock, throttle_pwm=1000, yaw_pwm=2000, duration_s=3.0):
    """Test whether yaw input causes altitude to climb at the given throttle.

    At throttle=1000 this validates AIRMODE-disabled state (motors idle, yaw
    should do nothing). At throttle=hover_pwm this validates the quadrotor
    motor-mixer asymmetry (yaw intrinsically adds thrust via ω²)."""
    logger.info(
        "=== Yaw bleed test: throttle=%d, yaw=%d, %.1fs ===",
        throttle_pwm,
        yaw_pwm,
        duration_s,
    )

    # Baseline: same throttle, yaw centered
    logger.info("Baseline: throttle=%d, yaw centered, 2s", throttle_pwm)
    hold(rc_sock, make_channels(throttle_pwm, armed=True, yaw_pwm=1500), 2.0)
    baseline_alt_start = query_pose()[2] if query_pose() else None
    hold(rc_sock, make_channels(throttle_pwm, armed=True, yaw_pwm=1500), duration_s)
    baseline_alt_end = query_pose()[2] if query_pose() else None
    baseline_delta = (
        baseline_alt_end - baseline_alt_start
        if baseline_alt_start is not None and baseline_alt_end is not None
        else None
    )
    logger.info(
        "Baseline: %s → %s (delta %s)",
        f"{baseline_alt_start:.3f}m" if baseline_alt_start is not None else "?",
        f"{baseline_alt_end:.3f}m" if baseline_alt_end is not None else "?",
        f"{baseline_delta:+.3f}m" if baseline_delta is not None else "?",
    )

    # Reset throttle low briefly to try to settle back down (only effective
    # at low altitudes; above ~0.5m the drone keeps coasting upward).
    logger.info("Resetting (CH3=1000, 2s)...")
    hold(rc_sock, make_channels(1000, armed=True), 2.0)

    # Yaw applied
    logger.info(
        "Yaw applied: throttle=%d, yaw=%d, sampling altitude %.1fs...",
        throttle_pwm,
        yaw_pwm,
        duration_s,
    )
    hold(rc_sock, make_channels(throttle_pwm, armed=True, yaw_pwm=1500), 1.0)  # settle
    yaw_alt_start = query_pose()[2] if query_pose() else None
    samples = []
    n_samples = int(duration_s / 0.25)
    for i in range(n_samples):
        hold(rc_sock, make_channels(throttle_pwm, armed=True, yaw_pwm=yaw_pwm), 0.25)
        pos = query_pose()
        alt = pos[2] if pos else None
        samples.append(((i + 1) * 0.25, alt))

    yaw_alt_end = samples[-1][1] if samples else None
    yaw_delta = (
        yaw_alt_end - yaw_alt_start
        if yaw_alt_start is not None and yaw_alt_end is not None
        else None
    )

    for t, alt in samples:
        print(
            f"  yaw=on t={t:.2f}s alt={alt:.3f}m"
            if alt is not None
            else f"  yaw=on t={t:.2f}s alt=?"
        )

    logger.info("--- Yaw test summary ---")
    logger.info(
        "Baseline delta: %s",
        f"{baseline_delta:+.3f}m" if baseline_delta is not None else "?",
    )
    logger.info(
        "Yaw-applied delta: %s (start=%s end=%s)",
        f"{yaw_delta:+.3f}m" if yaw_delta is not None else "?",
        f"{yaw_alt_start:.3f}m" if yaw_alt_start is not None else "?",
        f"{yaw_alt_end:.3f}m" if yaw_alt_end is not None else "?",
    )
    if baseline_delta is not None and yaw_delta is not None:
        bleed = yaw_delta - baseline_delta
        verdict = "YAW BLEED (thrust rises on yaw)" if bleed > 0.2 else "clean"
        logger.info("Yaw-induced altitude bleed: %+.3fm → %s", bleed, verdict)


def _sample_alt_over(rc_sock, channels, duration_s, dt=0.25):
    """Hold `channels` for duration_s, sampling altitude every dt seconds."""
    samples = []
    n = int(duration_s / dt)
    for i in range(n):
        hold(rc_sock, channels, dt)
        pos = query_pose()
        alt = pos[2] if pos else None
        samples.append(((i + 1) * dt, alt))
    return samples


def _print_samples(label, samples):
    for t, alt in samples:
        print(
            f"  {label} t={t:.2f}s alt={alt:.3f}m"
            if alt is not None
            else f"  {label} t={t:.2f}s alt=?"
        )


def run_yaw_airborne_test(
    rc_sock, hover_pwm=1495, climb_pwm=1600, climb_s=4.0, phase_s=3.0, yaw_pwm=2000
):
    """Reproduce "yaw while airborne → won't descend" bug.

    Phases (arm sequence already done by main):
      A. Climb to altitude at `climb_pwm` for `climb_s` seconds.
      B. Baseline descent: CH3=1000, yaw centered, sample altitude for phase_s.
         Re-climb after so phase C starts from altitude too.
      C. Yaw at hover: CH3=hover_pwm, yaw=yaw_pwm — does it climb vs. baseline hover?
      D. Yaw + throttle cut: CH3=1000, yaw=yaw_pwm — does it descend at all?

    The bug manifests as: phase D altitude stays ~flat or keeps climbing,
    while phase B descends normally.
    """
    logger.info(
        "=== Yaw-airborne test (hover=%d climb=%d yaw=%d) ===",
        hover_pwm,
        climb_pwm,
        yaw_pwm,
    )

    # --- Phase A: climb ---
    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    a_samples = _sample_alt_over(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    _print_samples("A-climb", a_samples)
    alt_after_climb = a_samples[-1][1] if a_samples else None
    logger.info(
        "A: altitude after climb = %s",
        f"{alt_after_climb:.3f}m" if alt_after_climb is not None else "?",
    )

    # --- Phase B: baseline descent, no yaw ---
    logger.info("B: baseline descent, CH3=1000 yaw=centered, %.1fs", phase_s)
    pos = query_pose()
    b_start = pos[2] if pos else None
    b_samples = _sample_alt_over(
        rc_sock, make_channels(1000, armed=True, yaw_pwm=1500), phase_s
    )
    _print_samples("B-cut-no-yaw", b_samples)
    b_end = b_samples[-1][1] if b_samples else None

    # --- Re-climb so phase C/D start from altitude again ---
    logger.info("Re-climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    pos = query_pose()
    alt_after_reclimb = pos[2] if pos else None
    logger.info(
        "Altitude after re-climb = %s",
        f"{alt_after_reclimb:.3f}m" if alt_after_reclimb is not None else "?",
    )

    # --- Phase C: yaw at hover throttle ---
    logger.info("C: yaw at hover, CH3=%d yaw=%d, %.1fs", hover_pwm, yaw_pwm, phase_s)
    pos = query_pose()
    c_start = pos[2] if pos else None
    c_samples = _sample_alt_over(
        rc_sock, make_channels(hover_pwm, armed=True, yaw_pwm=yaw_pwm), phase_s
    )
    _print_samples("C-yaw-hover", c_samples)
    c_end = c_samples[-1][1] if c_samples else None

    # --- Phase D: throttle cut while keeping yaw ---
    logger.info(
        "D: throttle cut WITH yaw held, CH3=1000 yaw=%d, %.1fs", yaw_pwm, phase_s
    )
    pos = query_pose()
    d_start = pos[2] if pos else None
    d_samples = _sample_alt_over(
        rc_sock, make_channels(1000, armed=True, yaw_pwm=yaw_pwm), phase_s
    )
    _print_samples("D-cut-yaw-held", d_samples)
    d_end = d_samples[-1][1] if d_samples else None

    def delta(s, e):
        return e - s if s is not None and e is not None else None

    b_d = delta(b_start, b_end)
    c_d = delta(c_start, c_end)
    d_d = delta(d_start, d_end)

    logger.info("--- Yaw-airborne summary ---")
    logger.info(
        "B baseline (cut, no yaw):   start=%s end=%s delta=%s",
        f"{b_start:.3f}m" if b_start is not None else "?",
        f"{b_end:.3f}m" if b_end is not None else "?",
        f"{b_d:+.3f}m" if b_d is not None else "?",
    )
    logger.info(
        "C yaw at hover:             start=%s end=%s delta=%s",
        f"{c_start:.3f}m" if c_start is not None else "?",
        f"{c_end:.3f}m" if c_end is not None else "?",
        f"{c_d:+.3f}m" if c_d is not None else "?",
    )
    logger.info(
        "D cut WHILE yaw held:       start=%s end=%s delta=%s",
        f"{d_start:.3f}m" if d_start is not None else "?",
        f"{d_end:.3f}m" if d_end is not None else "?",
        f"{d_d:+.3f}m" if d_d is not None else "?",
    )

    # Phase C is the primary signal: at hover PWM, yaw should NOT add net
    # thrust. Any significant climb in C means the motor mixer is producing
    # more-than-hover thrust when yaw is commanded — the "yaw shoots drone
    # into sky" bug.
    if c_d is not None:
        if c_d > 1.0:
            logger.info(
                "Verdict C (yaw at hover): BUG — climbed %+.2fm at hover+yaw", c_d
            )
        elif c_d < -1.0:
            logger.info("Verdict C (yaw at hover): yaw reduces thrust (%+.2fm)", c_d)
        else:
            logger.info("Verdict C (yaw at hover): clean (%+.2fm)", c_d)
    if b_d is not None and d_d is not None:
        # Phase D interpretation depends on whether climb came from motors
        # or just momentum carried over from C. In sim, drag is low so a
        # drone with upward velocity coasts for a while even at CH3=1000.
        if d_d > 1.0:
            logger.info(
                "Verdict D (throttle cut + yaw): still climbing %+.2fm — "
                "likely phase-C momentum carry-over, not a new thrust bug",
                d_d,
            )
        elif d_d > b_d * 0.5:
            logger.info(
                "Verdict D: partial descent (%+.3f vs baseline %+.3f)", d_d, b_d
            )
        else:
            logger.info("Verdict D: descends similarly to baseline")


def run_hover_sweep(rc_sock, pwm_low=1450, pwm_high=1550, step=5, hold_s=2.0):
    """Fine-grained sweep near hover. Reports altitude rate (m/s) per PWM so
    you can pick DRONE_HOVER_THROTTLE = the PWM closest to 0 m/s rate.

    Each step:
      1. Reset to CH3=1000 for 1s to let the drone settle on the ground.
      2. Hold target PWM for hold_s seconds.
      3. Sample altitude at start and end; rate = (end - start) / hold_s.
    """
    logger.info(
        "=== Hover sweep: %d-%d PWM, step %d, hold %.1fs ===",
        pwm_low,
        pwm_high,
        step,
        hold_s,
    )
    results = []
    for pwm in range(pwm_low, pwm_high + 1, step):
        # Settle back to ground so each step starts from the same state
        hold(rc_sock, make_channels(1000, armed=True), 1.0)
        pos_start = query_pose()
        alt_start = pos_start[2] if pos_start else None

        hold(rc_sock, make_channels(pwm, armed=True), hold_s)
        pos_end = query_pose()
        alt_end = pos_end[2] if pos_end else None

        if alt_start is not None and alt_end is not None:
            rate = (alt_end - alt_start) / hold_s
            results.append((pwm, alt_start, alt_end, rate))
            print(
                f"  pwm={pwm} alt {alt_start:.3f}→{alt_end:.3f}m  rate={rate:+.3f} m/s"
            )
        else:
            results.append((pwm, alt_start, alt_end, None))
            print(f"  pwm={pwm} alt ?→?  rate=?")

    # Find PWM with smallest absolute rate (closest to true hover)
    valid = [r for r in results if r[3] is not None]
    if valid:
        best = min(valid, key=lambda r: abs(r[3]))
        logger.info("--- Hover sweep summary ---")
        logger.info("Best hover candidate: PWM=%d (rate=%+.3f m/s)", best[0], best[3])
        logger.info(
            "Set DRONE_HOVER_THROTTLE = %d in your drone_config_XXmm.py", best[0]
        )
    else:
        logger.info("No valid samples — check pose subscription.")


def run_thrust_ramp(rc_sock):
    """Ramp throttle 1000→2000, measure altitude at each step."""
    logger.info("=== Thrust ramp test ===")
    logger.info("Ramping throttle from 1000 to 2000...")
    throttle_levels = list(range(1000, 2001, 50))

    results = []
    for pwm in throttle_levels:
        hold(rc_sock, make_channels(pwm, armed=True), 0.5)
        try:
            pos = query_pose()
            if pos:
                alt = pos[2]
                results.append((pwm, alt))
                print(f"throttle={pwm}, altitude={alt:.3f}m")
            else:
                results.append((pwm, None))
                print(f"throttle={pwm}, altitude=?")
        except Exception as e:
            results.append((pwm, None))
            print(f"throttle={pwm}, error: {e}")

    logger.info("--- Thrust ramp summary ---")
    hover_pwm = None
    for pwm, alt in results:
        if alt is not None and alt > 0.2:
            if hover_pwm is None:
                hover_pwm = pwm
            print(f"  {pwm} PWM -> {alt:.3f}m")
    if hover_pwm:
        logger.info("Approximate hover at ~%d PWM", hover_pwm)
    else:
        logger.info("No clear hover detected.")


def run_damper_test(
    rc_sock, hover_pwm=1495, climb_pwm=1600, climb_s=2.0, sample_s=5.0, dt=0.25
):
    """Validate vertical velocity damper.

    Steps:
      A. Climb to ~2m at climb_pwm for climb_s seconds.
      B. Cut to hover_pwm (altitude stick in deadband), sample altitude for sample_s.
      C. Assert |vz| < 0.1 m/s within 1s and altitude drift < 0.5m over sample_s.

    This test requires DroneGymEnv's altitude hold enabled, or it will just
    coast upward after throttle cut.
    """
    logger.info("=== Damper test: hover=%d climb=%d ===", hover_pwm, climb_pwm)

    # --- Phase A: climb ---
    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    pos = query_pose()
    alt_a = pos[2] if pos else None
    logger.info(
        "A: altitude after climb = %s", f"{alt_a:.3f}m" if alt_a is not None else "?"
    )

    # --- Phase B: hover with damper active ---
    logger.info("B: holding at hover PWM=%d, sampling %.1fs...", hover_pwm, sample_s)

    samples = []
    n_samples = int(sample_s / dt)
    for i in range(n_samples):
        hold(rc_sock, make_channels(hover_pwm, armed=True), dt)
        pos = query_pose()
        alt = pos[2] if pos else None
        samples.append(((i + 1) * dt, alt))

    # --- Check results ---
    alt_start = samples[0][1] if samples else None
    alt_end = samples[-1][1] if samples else None

    if alt_start is None or alt_end is None:
        logger.warning("No pose samples — cannot validate damper")
        return

    # Compute velocity by linear regression over all samples
    times = [s[0] for s in samples if s[1] is not None]
    alts = [s[1] for s in samples if s[1] is not None]
    if len(times) >= 2:
        # Simple average velocity
        vz_avg = (alts[-1] - alts[0]) / (times[-1] - times[0])
    else:
        vz_avg = None

    # Check first 1s window for rapid settling
    early = [(t, a) for t, a in samples if t <= 1.0 and a is not None]
    if len(early) >= 2:
        vz_early = (early[-1][1] - early[0][1]) / (early[-1][0] - early[0][0])
    else:
        vz_early = None

    drift = abs(alt_end - alt_start)

    logger.info("--- Damper test summary ---")
    logger.info(
        "Start: %s, End: %s, Drift: %s",
        f"{alt_start:.3f}m" if alt_start is not None else "?",
        f"{alt_end:.3f}m" if alt_end is not None else "?",
        f"{drift:.3f}m",
    )
    if vz_early is not None:
        logger.info(
            "Early vz (0-1s): %+.3f m/s %s",
            vz_early,
            "(settled)" if abs(vz_early) < 0.1 else "(NOT settled)",
        )
    if vz_avg is not None:
        logger.info(
            "Avg vz (%.1fs): %+.3f m/s %s",
            sample_s,
            vz_avg,
            "(holding)" if abs(vz_avg) < 0.1 else "(drifting)",
        )

    # Assertions
    if vz_early is not None and abs(vz_early) >= 0.1:
        logger.warning("FAIL: early vz %+.3f m/s >= 0.1 m/s (not settled)", vz_early)
    if drift >= 0.5:
        logger.warning("FAIL: drift %+.3f m >= 0.5m over %.1fs", drift, sample_s)
    if vz_early is not None and abs(vz_early) < 0.1 and drift < 0.5:
        logger.info("PASS: damper holding altitude within tolerance")


def run_pitch_climb_test(
    rc_sock, hover_pwm=1490, climb_pwm=1550, climb_s=3.0, pitch_s=5.0,
    pitch_pwm=2000, dt=0.25,
):
    """Quantify the "forward throttle → rapid climb" issue.

    In Angle mode at hover throttle, pitching forward should NOT cause a
    climb — vertical thrust component decreases as cos(angle), so altitude
    should drop slightly while the drone moves forward. If altitude climbs
    when CH2 is pushed forward at hover PWM, it indicates either:
      - BetaFlight is over-compensating motor commands during attitude hold
      - The mixer is producing excess net thrust under pitch demand
      - Drag/LiftDrag asymmetry produces lift at forward velocity

    Phases:
      A. Climb to ~2m at climb_pwm (gives clean test starting altitude)
      B. Baseline: hover_pwm + pitch centered for pitch_s, sample alt
      C. Pitch forward: hover_pwm + pitch_pwm for pitch_s, sample alt
    """
    logger.info(
        "=== Pitch-climb test: hover=%d climb=%d pitch=%d pitch_s=%.1f ===",
        hover_pwm, climb_pwm, pitch_pwm, pitch_s,
    )

    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    pos = query_pose()
    alt_a = pos[2] if pos else None
    logger.info("A: altitude after climb = %s",
                f"{alt_a:.3f}m" if alt_a is not None else "?")

    logger.info("B: baseline hover (CH3=%d, pitch centered), sampling %.1fs",
                hover_pwm, pitch_s)
    pos = query_pose()
    b_start = pos[2] if pos else None
    b_samples = _sample_alt_over(
        rc_sock, make_channels(hover_pwm, armed=True, pitch_pwm=1500), pitch_s, dt=dt,
    )
    _print_samples("B-pitch-center", b_samples)
    b_end = b_samples[-1][1] if b_samples else None

    logger.info("Re-climbing at CH3=%d for %.1fs to reset altitude",
                climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)

    logger.info("C: pitch forward (CH3=%d, CH2=%d), sampling %.1fs",
                hover_pwm, pitch_pwm, pitch_s)
    pos = query_pose()
    c_start = pos[2] if pos else None
    c_samples = _sample_alt_over(
        rc_sock,
        make_channels(hover_pwm, armed=True, pitch_pwm=pitch_pwm),
        pitch_s, dt=dt,
    )
    _print_samples("C-pitch-fwd", c_samples)
    c_end = c_samples[-1][1] if c_samples else None

    def delta(s, e):
        return e - s if s is not None and e is not None else None
    b_d = delta(b_start, b_end)
    c_d = delta(c_start, c_end)

    logger.info("--- Pitch-climb summary ---")
    logger.info(
        "B baseline (pitch centered): start=%s end=%s delta=%s",
        f"{b_start:.3f}m" if b_start is not None else "?",
        f"{b_end:.3f}m" if b_end is not None else "?",
        f"{b_d:+.3f}m" if b_d is not None else "?",
    )
    logger.info(
        "C pitch forward:             start=%s end=%s delta=%s",
        f"{c_start:.3f}m" if c_start is not None else "?",
        f"{c_end:.3f}m" if c_end is not None else "?",
        f"{c_d:+.3f}m" if c_d is not None else "?",
    )
    if b_d is not None and c_d is not None:
        bleed = c_d - b_d
        if bleed > 1.0:
            verdict = f"PITCH BLEED: pitching adds {bleed:+.2f}m climb vs baseline"
        elif bleed < -1.0:
            verdict = f"pitching reduces altitude {bleed:+.2f}m vs baseline (expected)"
        else:
            verdict = f"clean ({bleed:+.2f}m vs baseline)"
        logger.info("Verdict: %s", verdict)


def run_lateral_coast_test(
    rc_sock,
    hover_pwm=1490,
    climb_pwm=1550,
    climb_s=3.0,
    roll_pwm=2000,
    accel_s=2.0,
    coast_s=6.0,
    dt=0.25,
):
    """Quantify lateral coasting after roll is leveled.

    Phases:
      A. Climb to altitude.
      B. Apply sideways roll at hover throttle to build XY velocity.
      C. Return roll to center and sample horizontal speed / distance.

    With no translational drag, XY speed will decay slowly or not at all.
    A realistic body-drag model should show a clear speed half-life after
    leveling, even without active position hold.
    """
    logger.info(
        "=== Lateral-coast test: hover=%d climb=%d roll=%d accel=%.1fs coast=%.1fs ===",
        hover_pwm,
        climb_pwm,
        roll_pwm,
        accel_s,
        coast_s,
    )

    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    pos = query_pose()
    logger.info(
        "A: position after climb = %s",
        f"x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:+.2f}m" if pos else "?",
    )

    logger.info(
        "B: rolling sideways at CH1=%d, CH3=%d for %.1fs",
        roll_pwm,
        hover_pwm,
        accel_s,
    )
    hold(
        rc_sock,
        make_channels(hover_pwm, armed=True, roll_pwm=roll_pwm),
        accel_s,
    )
    pos0 = query_pose()
    if pos0 is None:
        logger.warning("No pose sample before coast — abort.")
        return

    logger.info(
        "C: roll centered, sampling XY coast for %.1fs at %.1fHz",
        coast_s,
        1.0 / dt,
    )
    samples = []
    prev_pos = pos0
    prev_t = time.time()
    t_start = prev_t
    n = int(coast_s / dt)
    for _ in range(n):
        hold(rc_sock, make_channels(hover_pwm, armed=True, roll_pwm=1500), dt)
        pos = query_pose()
        now = time.time()
        if pos is None:
            samples.append((now - t_start, None, None, None, None))
            continue
        elapsed = now - t_start
        step_dt = now - prev_t
        vx = (pos[0] - prev_pos[0]) / step_dt if step_dt > 0 else 0.0
        vy = (pos[1] - prev_pos[1]) / step_dt if step_dt > 0 else 0.0
        speed_xy = (vx * vx + vy * vy) ** 0.5
        drift_xy = (
            (pos[0] - pos0[0]) * (pos[0] - pos0[0])
            + (pos[1] - pos0[1]) * (pos[1] - pos0[1])
        ) ** 0.5
        samples.append((elapsed, pos, speed_xy, drift_xy, pos[2] - pos0[2]))
        prev_pos = pos
        prev_t = now

    for t, pos, speed_xy, drift_xy, dz in samples:
        if pos is None:
            print(f"  coast t={t:.2f}s pos=? speed_xy=? drift_xy=?")
        else:
            print(
                f"  coast t={t:.2f}s x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:+.2f}m"
                f" speed_xy={speed_xy:.3f}m/s drift_xy={drift_xy:.3f}m dz={dz:+.3f}m"
            )

    valid = [s for s in samples if s[1] is not None]
    if len(valid) < 2:
        logger.warning("Not enough pose samples to summarize coast.")
        return

    speeds = [s[2] for s in valid]
    peak = max(speeds)
    final = speeds[-1]
    final_drift = valid[-1][3]
    half_threshold = peak * 0.5
    half_time = next((s[0] for s in valid if s[2] <= half_threshold), None)

    logger.info("--- Lateral-coast summary ---")
    logger.info("Peak sampled XY speed after leveling: %.3f m/s", peak)
    logger.info("Final XY speed after %.1fs: %.3f m/s", valid[-1][0], final)
    logger.info("XY drift while leveled: %.3f m", final_drift)
    if half_time is None:
        logger.info("Speed half-life: > %.1fs (did not drop below 50%%)", coast_s)
    else:
        logger.info("Speed half-life: %.2fs", half_time)


def run_attitude_test(
    rc_sock, hover_pwm=1490, climb_pwm=1550, climb_s=3.0, sample_s=5.0,
    yaw_pwm=1530, dt=0.10,
):
    """Quantify "L/R shake" during yaw — sample roll/pitch oscillations.

    Bring the drone to altitude, command yaw with a moderate stick (default
    yaw_pwm=1530, matching DRONE_YAW_PWM_CAP=30). Sample roll/pitch attitude
    every dt seconds and report the std-dev / peak-to-peak amplitude — those
    quantify the wobble.

    Phases:
      A. Climb to altitude
      B. Yaw centered at hover, sample attitude — baseline noise
      C. Yaw applied at hover, sample attitude — wobble during yaw
    """
    logger.info(
        "=== Attitude test: hover=%d climb=%d yaw=%d sample_s=%.1f dt=%.2f ===",
        hover_pwm, climb_pwm, yaw_pwm, sample_s, dt,
    )

    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    att = query_attitude_deg()
    logger.info(
        "A: attitude after climb = %s",
        f"r={att[0]:+.1f}° p={att[1]:+.1f}° y={att[2]:+.1f}°" if att else "?",
    )

    def _sample_attitude(channels, dur):
        n = int(dur / dt)
        samples = []
        for i in range(n):
            hold(rc_sock, channels, dt)
            att = query_attitude_deg()
            samples.append(((i + 1) * dt, att))
        return samples

    def _stats(samples, idx):
        vals = [s[1][idx] for s in samples if s[1] is not None]
        if len(vals) < 2:
            return None
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        ptp = max(vals) - min(vals)
        return mean, std, ptp

    logger.info("B: baseline (CH4 centered) sampling %.1fs at %.1fHz",
                sample_s, 1.0 / dt)
    b = _sample_attitude(make_channels(hover_pwm, armed=True, yaw_pwm=1500), sample_s)
    b_roll = _stats(b, 0)
    b_pitch = _stats(b, 1)
    if b_roll:
        logger.info("B baseline roll:  mean=%+.2f° std=%.2f° p2p=%.2f°", *b_roll)
    if b_pitch:
        logger.info("B baseline pitch: mean=%+.2f° std=%.2f° p2p=%.2f°", *b_pitch)

    logger.info("Re-climbing at CH3=%d for %.1fs to reset altitude",
                climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)

    logger.info("C: yaw=%d at hover, sampling %.1fs at %.1fHz",
                yaw_pwm, sample_s, 1.0 / dt)
    c = _sample_attitude(make_channels(hover_pwm, armed=True, yaw_pwm=yaw_pwm), sample_s)
    c_roll = _stats(c, 0)
    c_pitch = _stats(c, 1)
    c_yaw = _stats(c, 2)
    if c_roll:
        logger.info("C yaw-on roll:    mean=%+.2f° std=%.2f° p2p=%.2f°", *c_roll)
    if c_pitch:
        logger.info("C yaw-on pitch:   mean=%+.2f° std=%.2f° p2p=%.2f°", *c_pitch)
    if c_yaw:
        # yaw_rate = total yaw delta / sample_s
        ys = [s[1][2] for s in c if s[1] is not None]
        if len(ys) >= 2:
            # Unwrap simple jumps across ±180°
            unwrap = [ys[0]]
            for v in ys[1:]:
                d = v - unwrap[-1]
                if d > 180:
                    v -= 360
                elif d < -180:
                    v += 360
                unwrap.append(v)
            yaw_rate = (unwrap[-1] - unwrap[0]) / sample_s
            logger.info("C yaw rate: %+.1f°/s over %.1fs", yaw_rate, sample_s)

    logger.info("--- Attitude test summary ---")
    if b_roll and c_roll:
        logger.info("Roll  shake increase (std): %.2f° → %.2f° (%+.2f°)",
                    b_roll[1], c_roll[1], c_roll[1] - b_roll[1])
    if b_pitch and c_pitch:
        logger.info("Pitch shake increase (std): %.2f° → %.2f° (%+.2f°)",
                    b_pitch[1], c_pitch[1], c_pitch[1] - b_pitch[1])


def run_inflight_hover_sweep(
    rc_sock, climb_pwm=1600, climb_s=3.0,
    pwm_low=1470, pwm_high=1500, step=2, hold_s=1.5, settle_s=0.5,
):
    """Find true in-flight hover PWM.

    The ground hover sweep can't see true hover because the drone won't lift
    off until thrust > weight + LiftDrag-at-zero-velocity. Once airborne,
    that drag is gone and a much lower PWM can sustain altitude.

    Strategy:
      A. Climb hard to ~5m so we have headroom.
      B. For each PWM (high → low), hold for hold_s, sample vz over the
         second half (skip transient). High → low so each step starts
         with downward momentum bias, helping us see decel from each PWM.

    The PWM with vz closest to 0 is true hover. With vz vs PWM in hand,
    we can also estimate the slope (m/s per PWM) for damper tuning.
    """
    logger.info(
        "=== Inflight hover sweep: %d-%d PWM step %d, hold %.1fs ===",
        pwm_low, pwm_high, step, hold_s,
    )

    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    pos = query_pose()
    logger.info("A: altitude after climb = %s",
                f"{pos[2]:.2f}m" if pos else "?")

    results = []
    # Sweep high → low so we catch the drone before it falls back to ground
    pwms = list(range(pwm_high, pwm_low - 1, -step))
    for pwm in pwms:
        # Hold the PWM for full duration; sample vz over the back half
        # (front half is transient as motors spool to new RPM)
        ch = make_channels(pwm, armed=True)
        # Settle phase
        hold(rc_sock, ch, settle_s)
        pos_a = query_pose()
        t_a = time.time()
        # Measurement window
        hold(rc_sock, ch, hold_s - settle_s)
        pos_b = query_pose()
        t_b = time.time()
        if pos_a and pos_b and (t_b - t_a) > 0:
            vz = (pos_b[2] - pos_a[2]) / (t_b - t_a)
            alt_b = pos_b[2]
            results.append((pwm, alt_b, vz))
            print(f"  pwm={pwm} alt={alt_b:6.2f}m  vz={vz:+.3f} m/s")
        else:
            results.append((pwm, None, None))
            print(f"  pwm={pwm} alt=?  vz=?")
        # Bail if drone has fallen too low to be in-flight (< 0.5m)
        if pos_b and pos_b[2] < 0.5:
            logger.info("Drone too low (%.2fm), stopping sweep", pos_b[2])
            break

    min_inflight_alt = 0.25
    valid = [r for r in results if r[2] is not None and r[1] is not None]
    inflight = [r for r in valid if r[1] > min_inflight_alt]
    if not inflight:
        logger.warning("No valid samples — abort.")
        return

    # Find true hover: PWM where vz crosses zero (or closest to zero)
    best = min(inflight, key=lambda r: abs(r[2]))
    logger.info("--- Inflight hover sweep summary ---")
    logger.info("Closest-to-hover: PWM=%d (vz=%+.3f m/s, alt=%.2fm)",
                best[0], best[2], best[1])

    # Linear interpolation: use adjacent in-flight samples only. This avoids
    # using ground-contact samples where z is clamped and apparent vz is false.
    crossing = None
    ordered = sorted(inflight, key=lambda r: r[0])
    for low, high in zip(ordered, ordered[1:]):
        if low[2] == 0:
            crossing = (low, low)
            break
        if (low[2] < 0 < high[2]) or (high[2] < 0 < low[2]):
            crossing = (low, high)
            break

    if crossing:
        n, p = crossing
        if n[2] > p[2]:
            n, p = p, n
        if p[0] != n[0]:
            slope = (p[2] - n[2]) / (p[0] - n[0])  # m/s per PWM
            zero_pwm = n[0] - n[2] / slope
            logger.info("Zero-vz interpolated: PWM=%.1f", zero_pwm)
            logger.info("Slope: %.4f (m/s) per PWM near hover", slope)
            logger.info("Suggested DRONE_HOVER_THROTTLE = %d", round(zero_pwm))
        else:
            logger.info("Zero-vz sampled directly: PWM=%d", n[0])
            logger.info("Suggested DRONE_HOVER_THROTTLE = %d", n[0])
    else:
        logger.info(
            "No vz zero-crossing found in sweep range — "
            "true hover is %s than swept range",
            "lower" if all(r[2] > 0 for r in inflight) else "higher",
        )


def run_damper_sim_test(
    rc_sock, hover_pwm=1490, k_pwm=30.0, climb_pwm=1600, climb_s=3.0,
    sample_s=8.0, dt=0.05, deadband=0.05, pitch_pwm=1500, pitch_after_s=None,
    yaw_pwm=1500, yaw_after_s=None, yaw_ff=0.0, yaw_pwm_cap=30,
):
    """Emulate drone_gym.py's _PoseTracker damper directly.

    Each tick:
      vz = (pos_z - prev_pos_z) / dt
      bias = clamp(-k_pwm * vz, -200, +200)
      ch3  = clamp(hover_pwm + bias, 1000, 2000)

    If this holds altitude steady (< 0.5m drift over sample_s), the damper
    logic is sound and the integration in drone_gym.py just needs hover_pwm
    correctly calibrated.
    """
    logger.info(
        "=== Damper-sim test: hover=%d k_pwm=%.1f sample=%.1fs dt=%.2fs ===",
        hover_pwm, k_pwm, sample_s, dt,
    )

    logger.info("A: climbing at CH3=%d for %.1fs", climb_pwm, climb_s)
    hold(rc_sock, make_channels(climb_pwm, armed=True), climb_s)
    pos = query_pose()
    alt_a = pos[2] if pos else None
    logger.info("A: altitude after climb = %s",
                f"{alt_a:.2f}m" if alt_a is not None else "?")

    n_ticks = int(sample_s / dt)
    prev_z = None
    prev_t = None
    samples = []  # (t, alt, vz, ch3, yaw_deg)
    log_skip = max(1, int(0.5 / dt))  # log every 0.5s

    target_alt = alt_a
    t_start = time.time()
    for i in range(n_ticks):
        pos = query_pose()
        now = time.time()
        if pos is None:
            ch3 = hover_pwm
            yaw_deg = None
        else:
            z = pos[2]
            if prev_z is None:
                vz = 0.0
            else:
                _dt = now - prev_t
                vz = (z - prev_z) / _dt if _dt > 0 else 0.0
            prev_z = z
            prev_t = now
            bias = -k_pwm * vz
            bias = max(-200.0, min(200.0, bias))
            # Mirror drone_gym.py yaw→throttle feed-forward (step + linear).
            cur_yaw_for_ff = (
                yaw_pwm if (yaw_after_s is not None and (now - t_start) >= yaw_after_s)
                else 1500
            )
            steer_eq = abs(cur_yaw_for_ff - 1500) / max(1, yaw_pwm_cap)
            steer_eq = min(1.0, steer_eq)
            yaw_ff_bias = 0.0
            if yaw_ff > 0 and steer_eq > 0.01:
                yaw_ff_bias = -yaw_ff
            yaw_ff_bias = max(-200.0, min(0.0, yaw_ff_bias))
            ch3 = int(max(1000, min(2000, hover_pwm + bias + yaw_ff_bias)))
            att = query_attitude_deg()
            yaw_deg = att[2] if att else None
            samples.append((now - t_start, z, vz, ch3, yaw_deg))

        # Optional pitch input applied after pitch_after_s
        cur_pitch = 1500
        if pitch_after_s is not None and (now - t_start) >= pitch_after_s:
            cur_pitch = pitch_pwm
        # Optional yaw input applied after yaw_after_s
        cur_yaw = 1500
        if yaw_after_s is not None and (now - t_start) >= yaw_after_s:
            cur_yaw = yaw_pwm
        send_rc(rc_sock, make_channels(
            ch3, armed=True, pitch_pwm=cur_pitch, yaw_pwm=cur_yaw,
        ))
        time.sleep(dt)
        if i % log_skip == 0 and samples:
            t, alt, vz, ch3v, yawv = samples[-1]
            yaw_str = f"{yawv:+.1f}" if yawv is not None else "?"
            logger.info(
                "t=%.2fs alt=%.2fm vz=%+.3f m/s ch3=%d yaw=%s°",
                t, alt, vz, ch3v, yaw_str,
            )

    if not samples:
        logger.warning("No samples collected.")
        return

    alts = [s[1] for s in samples]
    vzs = [s[2] for s in samples]
    ch3s = [s[3] for s in samples]
    drift = alts[-1] - alts[0]
    # Skip first 1s to ignore spin-down transient when computing settled stats
    settled = [s for s in samples if s[0] >= 1.0]
    if settled:
        s_alts = [s[1] for s in settled]
        s_vzs = [s[2] for s in settled]
        s_ch3 = [s[3] for s in settled]
        s_drift = s_alts[-1] - s_alts[0]
        s_alt_ptp = max(s_alts) - min(s_alts)
        s_vz_mean = sum(s_vzs) / len(s_vzs)
        s_ch3_mean = sum(s_ch3) / len(s_ch3)
        logger.info("--- Damper-sim summary ---")
        logger.info("Total drift over %.1fs: %+.3fm", sample_s, drift)
        logger.info(
            "After 1s settle: drift=%+.3fm  alt_p2p=%.3fm  vz_mean=%+.3fm/s  ch3_mean=%.1f",
            s_drift, s_alt_ptp, s_vz_mean, s_ch3_mean,
        )
        if abs(s_drift) < 0.5 and abs(s_vz_mean) < 0.1:
            logger.info("PASS: damper holds altitude (drift<0.5m, vz<0.1m/s)")
        else:
            logger.warning(
                "FAIL: damper not holding (drift=%+.3fm, vz=%+.3fm/s). "
                "Try increasing k_pwm (currently %.1f).",
                s_drift, s_vz_mean, k_pwm,
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["thrust", "yaw", "yaw-airborne", "hover", "both", "damper",
                 "pitch-climb", "lateral-coast", "attitude", "inflight-hover",
                 "damper-sim"],
        default="both",
        help="Which test(s) to run (default: both)",
    )
    parser.add_argument("--hover-low", type=int, default=1450)
    parser.add_argument("--hover-high", type=int, default=1550)
    parser.add_argument("--hover-step", type=int, default=5)
    parser.add_argument(
        "--yaw-throttle",
        type=int,
        default=1000,
        help="CH3 PWM during yaw test (default 1000 = airmode check; use hover PWM to test motor-mixer asymmetry)",
    )
    parser.add_argument(
        "--yaw-pwm",
        type=int,
        default=2000,
        help="CH4 PWM during yaw test (default 2000 = full right)",
    )
    parser.add_argument(
        "--airborne-hover",
        type=int,
        default=1495,
        help="Hover PWM for yaw-airborne test",
    )
    parser.add_argument(
        "--airborne-climb",
        type=int,
        default=1550,
        help="Climb PWM for yaw-airborne test (just above hover)",
    )
    parser.add_argument("--airborne-climb-s", type=float, default=2.0)
    parser.add_argument("--airborne-phase-s", type=float, default=3.0)
    parser.add_argument(
        "--damper-k",
        type=float,
        default=30.0,
        help="Damper k_pwm (PWM per m/s). Default 30 matches drone_gym.py.",
    )
    parser.add_argument(
        "--damper-pitch-pwm",
        type=int,
        default=1500,
        help="If --damper-pitch-after is set, pitch CH2 to this PWM after that time.",
    )
    parser.add_argument(
        "--damper-pitch-after",
        type=float,
        default=None,
        help="Seconds into damper-sim after which pitch is applied (default: never).",
    )
    parser.add_argument(
        "--damper-sample-s",
        type=float,
        default=8.0,
        help="Damper-sim total sample duration.",
    )
    parser.add_argument(
        "--damper-yaw-pwm",
        type=int,
        default=1500,
        help="If --damper-yaw-after is set, yaw CH4 to this PWM after that time.",
    )
    parser.add_argument(
        "--damper-yaw-after",
        type=float,
        default=None,
        help="Seconds into damper-sim after which yaw is applied (default: never).",
    )
    parser.add_argument(
        "--damper-yaw-ff",
        type=float,
        default=0.0,
        help="Yaw→throttle feed-forward (PWM at full equivalent steer). 0 = off.",
    )
    parser.add_argument(
        "--damper-yaw-cap",
        type=int,
        default=30,
        help="DRONE_YAW_PWM_CAP equivalent — used to convert yaw_pwm to steer_eq.",
    )
    parser.add_argument(
        "--lateral-roll-pwm",
        type=int,
        default=2000,
        help="CH1 roll PWM during lateral-coast acceleration phase.",
    )
    parser.add_argument(
        "--lateral-accel-s",
        type=float,
        default=2.0,
        help="Seconds to hold sideways roll before leveling.",
    )
    parser.add_argument(
        "--lateral-coast-s",
        type=float,
        default=6.0,
        help="Seconds to sample after returning roll to center.",
    )
    parser.add_argument(
        "--acro",
        action="store_true",
        help="Set CH6 = 1000 (Acro mode) instead of 2000 (Angle mode).",
    )
    args = parser.parse_args()

    global FLIGHT_MODE_ANGLE
    FLIGHT_MODE_ANGLE = not args.acro
    logger.info("Flight mode: %s", "Angle" if FLIGHT_MODE_ANGLE else "Acro")

    logger.info("Opening RC socket...")
    rc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    logger.info("Subscribing to pose topic...")
    _init_pose_subscriber()

    logger.info("Waiting for BetaFlight to come up...")
    time.sleep(2)

    # Pump idle disarm packets for a few seconds before arming. BF can sit
    # with ARMING_DISABLED_RX_FAILSAFE and ARMING_DISABLED_ANGLE set for
    # several seconds after (re)boot — until RC packets flow and the attitude
    # estimator converges from Gazebo IMU. 1s isn't always enough.
    logger.info("Pumping idle packets for 6s to clear failsafe/attitude...")
    hold(rc_sock, make_channels(1000, armed=False), 6.0)

    arm_sequence(rc_sock)

    if args.mode in ("yaw", "both"):
        run_yaw_test(rc_sock, throttle_pwm=args.yaw_throttle, yaw_pwm=args.yaw_pwm)

    if args.mode == "yaw-airborne":
        run_yaw_airborne_test(
            rc_sock,
            hover_pwm=args.airborne_hover,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
            phase_s=args.airborne_phase_s,
            yaw_pwm=args.yaw_pwm,
        )

    if args.mode == "hover":
        run_hover_sweep(rc_sock, args.hover_low, args.hover_high, args.hover_step)

    if args.mode == "damper":
        run_damper_test(
            rc_sock,
            hover_pwm=args.airborne_hover,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
        )

    if args.mode == "pitch-climb":
        run_pitch_climb_test(
            rc_sock,
            hover_pwm=args.airborne_hover,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
        )

    if args.mode == "lateral-coast":
        run_lateral_coast_test(
            rc_sock,
            hover_pwm=args.airborne_hover,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
            roll_pwm=args.lateral_roll_pwm,
            accel_s=args.lateral_accel_s,
            coast_s=args.lateral_coast_s,
        )

    if args.mode == "attitude":
        run_attitude_test(
            rc_sock,
            hover_pwm=args.airborne_hover,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
            sample_s=args.airborne_phase_s,
            yaw_pwm=args.yaw_pwm,
        )

    if args.mode == "inflight-hover":
        run_inflight_hover_sweep(
            rc_sock,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
            pwm_low=args.hover_low,
            pwm_high=args.hover_high,
            step=args.hover_step,
        )

    if args.mode == "damper-sim":
        run_damper_sim_test(
            rc_sock,
            hover_pwm=args.airborne_hover,
            climb_pwm=args.airborne_climb,
            climb_s=args.airborne_climb_s,
            k_pwm=args.damper_k,
            sample_s=args.damper_sample_s,
            pitch_pwm=args.damper_pitch_pwm,
            pitch_after_s=args.damper_pitch_after,
            yaw_pwm=args.damper_yaw_pwm,
            yaw_after_s=args.damper_yaw_after,
            yaw_ff=args.damper_yaw_ff,
            yaw_pwm_cap=args.damper_yaw_cap,
        )

    if args.mode in ("thrust", "both"):
        run_thrust_ramp(rc_sock)

    logger.info("Disarming...")
    hold(rc_sock, make_channels(1000, armed=False), 0.5)
    rc_sock.close()


if __name__ == "__main__":
    main()
