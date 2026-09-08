"""Scripted gate-flying expert over the velocity-setpoint interface.

This replaces the PPO teacher. With DRONE_VELOCITY_CONTROL the env closes the
inner loop, so "fly through the next gate" is a short function of the
privileged state rather than something that needs 400k steps of exploration to
discover. That buys three things the RL teacher never had:

  * exact labels, with no checkpoint-picking and no reward shaping
  * a DAgger labeller that costs one function call per visited state
  * a failure mode that localises -- if the expert misses, either the cascade
    is mistuned (visible in probe_velocity) or the approach law is wrong

Control law:

  Pure pursuit along the GATE CENTERLINE, the line through the gate center
  along its normal. The carrot is the drone's own along-track position pushed
  forward by a lookahead, with zero lateral offset, so the drone converges onto
  the normal before it arrives instead of cutting the corner. Lateral misses
  were the wall for both RL teachers (v5: 10 of 10 misses were lateral), and
  this is aimed squarely at them.

  Yaw turns the nose toward the carrot. That matters more for the STUDENT than
  for the expert: it keeps the gate inside the camera frame, so the distilled
  vision policy is learning a task whose target it can actually see.

  Height is a proportional trim toward the gate height, and its gain is a
  STABILITY constraint, not an accuracy one. The command shifts the setpoint of
  a hold that is already a position regulator, so Kz wraps a second proportional
  loop around a second-order plant: the closed loop becomes w' = w*sqrt(1+Kz),
  zeta' = zeta/sqrt(1+Kz). Worse, the trim saturates at |error| > TRIM_RANGE/Kz,
  and a saturated proportional loop is a relay, which limit-cycles.

  Measured 2026-09-06 at Kz=2.0: z oscillated 1.65 -> 2.48 -> 1.38 -> 3.01 ->
  1.11 -> 3.08 with the command pinned at -1.00/+0.80, i.e. a GROWING bang-bang
  limit cycle that flew the drone over the 3.0m gate ceiling. Kz=0.5 gives
  zeta' = 0.67 and only saturates beyond 2m of error. Accuracy is not the
  tradeoff it looks like: the settling point is (setpoint + Kz*gate_z)/(1+Kz),
  and episodes anchor the setpoint at the gate height, so the residual is ~0.
"""

import math

import numpy as np

# Sign of the yaw stick, MEASURED with dense UNWRAPPED sampling on 2026-09-06:
# steer +0.25/+0.50/+1.00 gave -36.8/-120.9/-423.0 deg/s, and -0.50 gave
# +119.9. So a positive stick DECREASES the Gazebo/ENU yaw and turns the drone
# RIGHT, and steering toward a target on the right is POSITIVE.
#
# Worth knowing how this was got wrong first: an earlier probe sampled yaw only
# at the START and END of a 2s leg and read "+0.2 -> +119.4 degrees" as +119.
# The true motion was -240.8. Both fit the endpoints, so the measurement was
# aliased and the sign came out backwards. The resulting expert drove yaw AWAY
# from the target until bearing 180 became a stable equilibrium: the drone flew
# the whole course facing backwards. Always unwrap, and sample faster than half
# a revolution.
YAW_STICK_SIGN = 1.0


class GateExpert:
    """Emits velocity setpoints that fly the gates in order.

    Outputs are in DonkeyCar key order and units, ready to hand to
    DroneGymEnv.run_threaded: [steering, throttle, roll, altitude].
    """

    def __init__(
        self,
        gates,
        *,
        max_forward_speed=2.0,
        max_lateral_speed=1.0,
        altitude_trim_range=1.0,
        cruise_speed=1.5,
        lookahead=3.0,
        yaw_gain=1.5,
        max_yaw_rate_rad=None,
        height_gain=0.5,
        align_slowdown=False,
        min_speed_frac=0.25,
    ):
        if not gates:
            raise ValueError("GateExpert needs at least one gate")
        self.gates = gates
        self.max_forward_speed = float(max_forward_speed)
        self.max_lateral_speed = float(max_lateral_speed)
        self.altitude_trim_range = float(max(1e-6, altitude_trim_range))
        self.cruise_speed = float(cruise_speed)
        self.lookahead = float(lookahead)
        # Loop gain of the heading controller, in 1/s: the commanded yaw RATE is
        # yaw_gain * bearing. Normalising by the true full-stick rate below makes
        # this a real time constant (1/yaw_gain) rather than an arbitrary number.
        self.yaw_gain = float(yaw_gain)
        # Yaw rate at full stick, used to convert the desired rate into a stick.
        self.max_yaw_rate_rad = float(max_yaw_rate_rad or math.radians(120.0))
        self.height_gain = float(height_gain)
        self.align_slowdown = bool(align_slowdown)
        self.min_speed_frac = float(min_speed_frac)

    @classmethod
    def from_env(cls, drone_env, gates, **kwargs):
        """Build with the speed/trim scaling the env actually applies."""
        return cls(
            gates,
            max_forward_speed=getattr(drone_env, "max_forward_speed", 2.0),
            max_lateral_speed=getattr(drone_env, "max_lateral_speed", 1.0),
            altitude_trim_range=getattr(drone_env, "altitude_trim_range", 1.0),
            max_yaw_rate_rad=math.radians(getattr(drone_env, "max_yaw_rate", 120.0)),
            **kwargs,
        )

    def act(self, pos, yaw_rad, gate_idx):
        """Return [steering, throttle, roll, altitude] for the current state.

        :param pos: (x, y, z) world position
        :param yaw_rad: drone yaw in radians
        :param gate_idx: index of the gate being chased
        """
        gate = self.gates[min(gate_idx, len(self.gates) - 1)]
        gc, gs = math.cos(gate["yaw"]), math.sin(gate["yaw"])

        # Drone in gate-local coords: lx along the gate normal (negative on the
        # approach side), ly lateral. Same convention as GateChaseEnv.
        dx, dy = pos[0] - gate["x"], pos[1] - gate["y"]
        lx = gc * dx + gs * dy
        ly = -gs * dx + gc * dy

        # Pure-pursuit carrot: forward along the centerline, zero lateral.
        # Driving toward it converges ly to 0 before the gate plane is reached.
        cx_local = lx + self.lookahead
        carrot_x = gate["x"] + gc * cx_local
        carrot_y = gate["y"] + gs * cx_local

        # Carrot in the drone-heading frame (forward, right).
        wx, wy = carrot_x - pos[0], carrot_y - pos[1]
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        fwd = wx * cy + wy * sy
        right = -(-wx * sy + wy * cy)

        dist = math.hypot(fwd, right)
        if dist < 1e-6:
            return np.zeros(4, dtype=np.float32)

        bearing = math.atan2(right, fwd)

        # NOTE: no cos(bearing) slowdown by default. That is a car-like
        # (nonholonomic) reflex and this vehicle is holonomic -- the velocity
        # setpoint is expressed in the HEADING frame, so a misaligned nose still
        # produces motion in the right world direction via the lateral axis.
        # Measured 2026-09-06: with the slowdown on, cruise collapsed to
        # 0.375 m/s whenever the bearing was large, and an episode took 849
        # steps (42s) to reach one gate.
        speed = self.cruise_speed
        if self.align_slowdown:
            speed *= max(self.min_speed_frac, math.cos(bearing))

        v_fwd = speed * (fwd / dist)
        v_right = speed * (right / dist)

        throttle = v_fwd / self.max_forward_speed
        roll = v_right / self.max_lateral_speed
        # Desired yaw rate (rad/s) -> stick, via the full-stick rate.
        steering = YAW_STICK_SIGN * (self.yaw_gain * bearing) / self.max_yaw_rate_rad
        altitude = self.height_gain * (gate["z"] - pos[2]) / self.altitude_trim_range

        return np.clip(
            np.array([steering, throttle, roll, altitude], dtype=np.float32), -1.0, 1.0
        )

    def act_from_telemetry(self, telemetry, gate_idx):
        """Convenience for GateChaseEnv.last_telemetry (pos, att, vel, imu)."""
        pos = telemetry[0:3]
        yaw_deg = telemetry[5]
        return self.act(pos, math.radians(yaw_deg), gate_idx)
