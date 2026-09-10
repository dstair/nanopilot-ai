"""Episode driver for the takeoff -> line-follow -> land task.

`LineChaseEnv` is `GateChaseEnv` with one block swapped. Everything that makes
the gate env worth reusing is task-agnostic and stays: the eight-attempt
teleport-and-takeoff, the `direct_thrust` climb ramp and climb arrest, the
fixed-rate control loop, the telemetry/image capture `rl_collect` writes tub
rows from, and the crash/flyaway/truncation handling. Only "have we crossed
the next gate" becomes "have we touched down, and was it a landing".

The pad is passed to the parent AS A ONE-ENTRY GATE LIST. That is not a hack
to satisfy a signature: on this task the pad genuinely is the single target,
so `_gate_dist` becomes distance to the pad and the parent's observation
becomes pad-relative, both of which are the right thing.

**Touchdown is classified by `eval_flight.classify_touchdown`, the same
function the scorer uses.** Deliberately: a collector that kept episodes the
evaluator would call crashes would poison the tub with exactly the labels the
student must not learn, and two definitions of "a landing" would drift apart.

`crash_z` is disabled here (-1.0) rather than left at the parent's 0.12.
A textbook landing passes 0.12 on the way down -- measured 2026-09-09, z<0.20
at 5.4s with touchdown at 8.6s -- so the parent's crash test would fire on
every successful landing before the touchdown test could see it. A genuine
fall is still caught, by the impact speed in the classifier, and a tumble by
the tilt test, which is untouched.
"""

import math
from collections import deque

from eval_flight import TOUCHDOWN_VZ_WINDOW_S, classify_touchdown
from rl_env import POS, VEL, GateChaseEnv


class LineChaseEnv(GateChaseEnv):
    """Fly the guidance line to the pad and land on it, scored from ground truth."""

    def __init__(
        self,
        drone_env,
        pad_xy,
        *,
        pad_radius=1.0,
        touchdown_z=0.06,
        touchdown_vz=0.15,
        touchdown_speed=0.5,
        land_bonus=20.0,
        **kwargs,
    ):
        if pad_xy is None:
            raise ValueError("LineChaseEnv needs a landing pad")
        kwargs.setdefault("crash_z", -1.0)  # see the module docstring
        pad_gate = [{
            "name": "landing_pad",
            "x": float(pad_xy[0]), "y": float(pad_xy[1]), "z": 0.0, "yaw": 0.0,
        }]
        super().__init__(drone_env, pad_gate, **kwargs)
        self.pad_xy = (float(pad_xy[0]), float(pad_xy[1]))
        self.pad_radius = float(pad_radius)
        self.touchdown_z = float(touchdown_z)
        self.touchdown_vz = float(touchdown_vz)
        self.touchdown_speed = float(touchdown_speed)
        self.land_bonus = float(land_bonus)
        # Rolling window of (t, x, y, z, vz, speed_xy), the tuple shape
        # classify_touchdown expects, covering the last TOUCHDOWN_VZ_WINDOW_S.
        # The impact speed has to be read from BEFORE contact: velocity is
        # low-passed with tau=0.1 and the ground stops the drone in one frame,
        # so the sample at contact makes every impact look gentle.
        self._recent = deque()
        self._t = 0.0
        self.last_touchdown = None

    def reset(self, seed=None, options=None):
        self._recent.clear()
        self._t = 0.0
        self.last_touchdown = None
        return super().reset(seed=seed, options=options)

    def _task_step(self, pos, outputs):
        reward = 0.0
        terminated = False
        info = {}

        vel = outputs[VEL]
        self._t += self.period
        self._recent.append(
            (self._t, pos[0], pos[1], pos[2], vel[2], math.hypot(vel[0], vel[1]))
        )
        while self._recent and self._recent[0][0] < self._t - TOUCHDOWN_VZ_WINDOW_S:
            self._recent.popleft()

        # Progress toward the pad, same shaping as the gate env's.
        dist = self._gate_dist(self.gates[0], pos)
        if self._prev_dist is not None:
            reward += self.progress_scale * (self._prev_dist - dist)
        self._prev_dist = dist

        if pos[2] < self.touchdown_z:
            result, touchdown = classify_touchdown(
                self._recent, pos, self.pad_xy, self.pad_radius,
                self.touchdown_vz, self.touchdown_speed,
            )
            touchdown["touchdown_s"] = round(self._t, 2)
            self.last_touchdown = touchdown
            terminated = True
            info["episode_result"] = result
            if result == "completed":
                reward += self.land_bonus
            else:
                reward -= self.crash_penalty
        return reward, terminated, info

    def _task_info(self):
        info = {"landed": self.last_touchdown is not None}
        if self.last_touchdown:
            info.update(self.last_touchdown)
        return info

    def _task_summary(self):
        if not self.last_touchdown:
            return "no touchdown"
        return (
            f"touchdown vz {self.last_touchdown['touchdown_vz_ms']:+.3f} m/s "
            f"{self.last_touchdown['touchdown_pad_offset_m']:.3f}m from pad"
        )
