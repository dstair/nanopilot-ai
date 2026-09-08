"""Sweep DRONE_PITCH_THROTTLE_FEEDFORWARD to minimize climb during forward flight.

The RL policy holds pitch nearly continuously, and the velocity-only damper
can only cancel a constant thrust bias at a nonzero climb rate
(vz_steady = ff*pitch/K). Too much ff -> climbs out of the gate band; too
little -> sinks as the tilt eats vertical thrust. We want ~0 drift at the
pitch the policy actually flies.

The feed-forward is read live off the env, so one stack serves the whole
sweep. Between measurements the drone is re-centered into an altitude band
so each sample starts from comparable conditions.
"""
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "nanopilot"))
os.environ.setdefault("GZ_WORLD", "slalom_gates_80mm")

import donkeycar as dk
from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from rl_env import POS, VEL

BAND_LOW, BAND_HIGH = 1.2, 2.2
MEASURE_S = 15.0
FF_VALUES = [0.0, 2.0, 4.0, 6.0, 8.0]
PITCHES = [0.5, 1.0]

cfg = dk.load_config(
    config_path=str(pathlib.Path(__file__).resolve().parent.parent / "nanopilot" / "config.py"),
    myconfig="drone_config_80mm.py",
)
print(f"HOVER={cfg.DRONE_HOVER_THROTTLE} K={cfg.DRONE_ALTITUDE_HOLD_K} "
      f"PITCH_FF(config)={cfg.DRONE_PITCH_THROTTLE_FEEDFORWARD}")

_wait_for_tcp("127.0.0.1", 5761, 10, "BetaFlight MSP")
env = build_drone_env(
    cfg, airframe="80mm", gz_world="slalom_gates_80mm",
    record_position=True, record_attitude=True, record_velocity=True, record_imu=True,
)
# This sweep sets the PITCH STICK directly to measure tilt-vs-climb, so it is a
# direct-angle-mode tool by construction. Under DRONE_VELOCITY_CONTROL the same
# argument would be a speed setpoint and the sweep would measure nothing.
env.velocity_control = False
threading.Thread(target=env.update, daemon=True).start()
_wait_for_ready(env, 30)


def z():
    return env.run_threaded(0.0, 0.0, 0.0, 0.0)[POS][2]


def takeoff():
    climb = 0.2
    t0 = time.time()
    while time.time() - t0 < 15:
        out = env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=climb)
        if out[VEL][2] > 1.0:
            return
        climb = min(0.6, climb + 0.01)
        time.sleep(0.05)


def recenter(timeout=30.0):
    """Bring altitude back into the band using explicit climb/descend."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        alt = z()
        if alt < 0.3:
            takeoff()
            continue
        if BAND_LOW <= alt <= BAND_HIGH:
            # settle with damper
            for _ in range(40):
                env.run_threaded(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)
            if BAND_LOW - 0.3 <= z() <= BAND_HIGH + 0.3:
                return True
            continue
        cmd = 0.25 if alt < BAND_LOW else -0.25
        env.run_threaded(0.0, 0.0, 0.0, cmd)
        time.sleep(0.05)
    return False


def measure(pitch, dur=MEASURE_S):
    t0 = time.time()
    z0 = z()
    while time.time() - t0 < dur:
        env.run_threaded(0.0, pitch, 0.0, 0.0)
        time.sleep(0.05)
    z1 = z()
    return (z1 - z0) / dur, z0, z1


takeoff()
recenter()
print(f"\nMeasuring {MEASURE_S:.0f}s per sample, band {BAND_LOW}-{BAND_HIGH}m\n")

results = {}
for ff in FF_VALUES:
    env.pitch_throttle_feedforward = ff
    for pitch in PITCHES:
        ok = recenter()
        rate, z0, z1 = measure(pitch)
        results[(ff, pitch)] = rate
        print(f"  ff={ff:4.1f} pitch={pitch:.1f}: z {z0:5.2f} -> {z1:5.2f}  "
              f"climb {rate:+.3f} m/s  (60s: {rate*60:+5.1f}m)"
              f"{'' if ok else '  [recenter timeout]'}")

print("\n=== summary: climb rate (m/s) ===")
print("   ff  " + "".join(f"  pitch{p:.1f}" for p in PITCHES))
for ff in FF_VALUES:
    row = "".join(f"  {results[(ff,p)]:+7.3f}" for p in PITCHES)
    print(f"  {ff:4.1f}{row}")

best = min(FF_VALUES, key=lambda f: sum(abs(results[(f, p)]) for p in PITCHES))
print(f"\nBest ff (min |climb| summed over pitches): {best}")
for p in PITCHES:
    print(f"  at pitch {p}: {results[(best,p)]:+.3f} m/s "
          f"({results[(best,p)]*60:+.1f}m over a 60s episode)")

env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
time.sleep(0.5)
env.shutdown()
