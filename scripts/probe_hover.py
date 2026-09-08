"""Does the 80mm airframe hold altitude at neutral once airborne?

Lifts off, then holds neutral sticks (damper engaged) for 30s, logging z.
"""
import os
import sys
import threading
import time

sys.path.insert(0, "/Users/Dan/dev/nanopilot-ai/nanopilot")
os.environ.setdefault("GZ_WORLD", "slalom_gates_80mm")

import donkeycar as dk
from autonomous_collect import _wait_for_ready, _wait_for_tcp
from drone_env import build_drone_env
from rl_env import POS, VEL

cfg = dk.load_config(
    config_path="/Users/Dan/dev/nanopilot-ai/nanopilot/config.py",
    myconfig="drone_config_80mm.py",
)
print(f"HOVER_THROTTLE={cfg.DRONE_HOVER_THROTTLE} "
      f"RANGE={cfg.DRONE_THROTTLE_RANGE} "
      f"DAMPER_K={cfg.DRONE_ALTITUDE_HOLD_K} "
      f"DEADBAND={cfg.DRONE_ALTITUDE_HOLD_DEADBAND}")

_wait_for_tcp("127.0.0.1", 5761, 10, "BetaFlight MSP")
env = build_drone_env(
    cfg, airframe="80mm", gz_world="slalom_gates_80mm",
    record_position=True, record_attitude=True, record_velocity=True, record_imu=True,
)
threading.Thread(target=env.update, daemon=True).start()
_wait_for_ready(env, 30)
print(f"armed={env.bf_armed}")

# Ramp up until it lifts off, then release to neutral.
climb = 0.2
t0 = time.time()
while time.time() - t0 < 15:
    out = env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=climb)
    if out[VEL][2] > 1.0:
        print(f"liftoff at z={out[POS][2]:.2f} climb_cmd={climb:.2f}")
        break
    climb = min(0.6, climb + 0.01)
    time.sleep(0.05)

print("\n--- holding NEUTRAL for 30s (damper engaged) ---")
t0 = time.time()
i = 0
while time.time() - t0 < 30:
    out = env.run_threaded(0.0, 0.0, 0.0, 0.0)
    if i % 20 == 0:
        print(f"  t={time.time()-t0:5.1f}s z={out[POS][2]:7.2f} vz={out[VEL][2]:6.2f} "
              f"thr_pwm={env.last_throttle_pwm}")
    i += 1
    time.sleep(0.05)

env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=-1.0)
time.sleep(0.5)
env.shutdown()
