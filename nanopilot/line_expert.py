"""Scripted takeoff-line-land expert over the velocity-setpoint interface.

Same shape and the same reasoning as `gate_expert.GateExpert`, against a
different task: fly the guidance line to the pad, then put the drone down on
it. Like that one it is a PURE FUNCTION of privileged state, so it needs no
training, no checkpoint to pick, and it can label any state DAgger visits with
one call.

Control law, in three parts:

  Horizontal -- pure pursuit along the LINE from the start to the pad. The
  carrot is the drone's own along-track position pushed forward by a lookahead
  and clamped at the pad, at zero lateral offset, so the drone converges onto
  the line before it arrives rather than cutting across. Identical in spirit to
  the gate expert's centerline pursuit, and it is what keeps the guidance line
  under the camera for the student that gets distilled from this.

  Speed -- a proportional POSITION loop on distance to the pad,
  `v = min(cruise, approach_gain * d)`, so the drone eases in and stops over
  the pad instead of arriving at cruise and needing to be caught. Converges
  exponentially with time constant `1/approach_gain`.

  Height -- a reference that ramps from the cruise height down to a target
  BELOW GROUND as the drone closes on the pad, fed to the env through the same
  feed-forward-plus-proportional law the gate expert uses:

      altitude = ((z_ref - setpoint) + Kz * (z_ref - z)) / TRIM_RANGE

Three things about the height law are load-bearing:

  * **The descent is a function of horizontal distance, not of time.** That
    makes the whole expert memoryless -- there is no phase variable, no timer
    and no latch, so `act()` is reproducible from state alone and DAgger can
    label a drone that has drifted back off the pad with "climb and go round"
    rather than with whatever the last phase happened to be.

  * **The ramp SATURATES at `land_radius`, it does not taper to the target.**
    A reference that only reaches the ground target as d -> 0 never lands: the
    hold settles ~0.10m above whatever it is asked for (the plant's residual
    thrust bias over Kp, measured 2026-09-09), so a drone hovering 0.3m off the
    pad centre would sit at z ~ 0.2m forever and the episode would score NEVER
    TOUCHED DOWN. Below `land_radius` the reference is pinned at
    `ground_target` and the drone is driven into the ground.

  * **`ground_target` is BELOW ground (-0.1m) on purpose.** Descent speed is
    set entirely by how far below ground the target sits, because the axis
    commands a height and not a rate -- measured 2026-09-09, a full-down
    command from a 1.5m cruise touches down in 8.6s at 0.02 m/s with the target
    at -0.10m, and in 4.0s at 0.18 m/s with it at -0.50m. Aiming at exactly 0.0
    would approach the ground asymptotically and never quite arrive.

**`descend_radius` is a GLIDE SLOPE, and it is sized by what the camera can
see, not by what the plant can do.** A small radius (2m) makes the drone fly
level to the pad and then descend vertically on the spot. That lands perfectly
well -- 10/10 at 0.075m from the pad centre -- but it is bad DATA: the camera
is forward-facing with a 140 deg horizontal FOV, which is +-64 deg vertically
at 4:3, so a drone hovering 0.3m from the pad at 1.5m sees it at 78.7 deg
depression, i.e. OUT OF FRAME. The pad only reappears below ~0.6m. That is the
2026-09-06b "the target is outside the camera at the moment of the turn"
failure, pointed at the pad instead of a gate.

Descending along the approach instead keeps the pad in frame essentially all
the way down. Measured over one episode each, at trim 1.6:

    radius   duration   pad offset   pad out of frame   alt label in the ramp
      2m      20.9s       0.075m          16.7%                 6.1%
      8m      18.5s       0.048m           3.3%                22.7%
     12m      17.4s       0.165m           2.3%                37.8%

8m is the default because it wins on every axis at once: faster than level
flight, the tightest touchdown of the three, the pad visible for all but the
last few centimetres, and an altitude label that spends a fifth of the episode
on a continuous ramp rather than being a near-binary switch between 0 and -1.
12m descends so early that the drone arrives low and still moving, which costs
touchdown accuracy (0.165m).

Reaching a target below ground needs the trim range to cover it:
`DRONE_ALTITUDE_TRIM_RANGE >= cruise_alt - ground_target`, i.e. >= 1.6 for a
1.5m cruise. At the older 1.0 the command simply clips at -1 and the drone
hovers at `setpoint - 1.0` = 0.5m, which the land scorer reports as NEVER
TOUCHED DOWN rather than as a landing. That is the intended failure: it is
self-diagnosing rather than silent.
"""

import math

import numpy as np

from gate_expert import YAW_STICK_SIGN


class LineExpert:
    """Emits velocity setpoints that fly a line to a pad and land on it.

    Outputs are in DonkeyCar key order and units, ready to hand to
    DroneGymEnv.run_threaded: [steering, throttle, roll, altitude].
    """

    def __init__(
        self,
        start_xy,
        pad_xy,
        *,
        max_forward_speed=2.0,
        max_lateral_speed=1.0,
        altitude_trim_range=1.0,
        cruise_speed=1.5,
        cruise_alt=None,
        lookahead=3.0,
        yaw_gain=1.5,
        max_yaw_rate_rad=None,
        height_gain=0.75,
        approach_gain=0.6,
        descend_radius=8.0,
        land_radius=0.5,
        ground_target=-0.1,
        yaw_hold_radius=1.5,
        altitude_setpoint_fn=None,
    ):
        if pad_xy is None:
            raise ValueError("LineExpert needs a landing pad")
        self.start_xy = (float(start_xy[0]), float(start_xy[1]))
        self.pad_xy = (float(pad_xy[0]), float(pad_xy[1]))
        self.max_forward_speed = float(max_forward_speed)
        self.max_lateral_speed = float(max_lateral_speed)
        self.altitude_trim_range = float(max(1e-6, altitude_trim_range))
        self.cruise_speed = float(cruise_speed)
        # None means "hold whatever height the takeoff froze the setpoint at",
        # which makes the cruise command ~0 the way a flat gate course does.
        self.cruise_alt = None if cruise_alt is None else float(cruise_alt)
        self.lookahead = float(lookahead)
        self.yaw_gain = float(yaw_gain)
        self.max_yaw_rate_rad = float(max_yaw_rate_rad or math.radians(120.0))
        self.height_gain = float(height_gain)
        # 1/s. The commanded speed is approach_gain * distance, capped at
        # cruise, so 1/approach_gain is the approach time constant.
        self.approach_gain = float(approach_gain)
        self.descend_radius = float(descend_radius)
        self.land_radius = float(land_radius)
        self.ground_target = float(ground_target)
        # Inside this radius the heading command is frozen. The bearing to a
        # carrot the drone is sitting on top of is numerical noise, and acting
        # on it spins the drone while it is trying to settle.
        self.yaw_hold_radius = float(yaw_hold_radius)
        self.altitude_setpoint_fn = altitude_setpoint_fn
        self.last_debug = {}

    @classmethod
    def from_env(cls, drone_env, start_xy, pad_xy, **kwargs):
        """Build with the speed/trim scaling the env actually applies."""
        return cls(
            start_xy,
            pad_xy,
            max_forward_speed=getattr(drone_env, "max_forward_speed", 2.0),
            max_lateral_speed=getattr(drone_env, "max_lateral_speed", 1.0),
            altitude_trim_range=getattr(drone_env, "altitude_trim_range", 1.0),
            max_yaw_rate_rad=math.radians(getattr(drone_env, "max_yaw_rate", 120.0)),
            altitude_setpoint_fn=getattr(drone_env, "get_altitude_setpoint", None),
            **kwargs,
        )

    def _height_ref(self, d_pad, cruise_alt):
        """Course height at the drone's own distance from the pad.

        Cruise height beyond `descend_radius`, `ground_target` inside
        `land_radius`, linear between. See the module docstring for why the
        inner end saturates instead of tapering to the target.
        """
        span = max(1e-6, self.descend_radius - self.land_radius)
        f = (d_pad - self.land_radius) / span
        f = min(1.0, max(0.0, f))
        return self.ground_target + f * (cruise_alt - self.ground_target)

    def act(self, pos, yaw_rad):
        """Return [steering, throttle, roll, altitude] for the current state.

        :param pos: (x, y, z) world position
        :param yaw_rad: drone yaw in radians
        """
        px, py = float(pos[0]), float(pos[1])
        sx, sy = self.start_xy
        gx, gy = self.pad_xy

        # Along-track frame of the guidance line.
        lx, ly = gx - sx, gy - sy
        length = math.hypot(lx, ly)
        if length < 1e-6:
            ux, uy, length = 1.0, 0.0, 1e-6
        else:
            ux, uy = lx / length, ly / length
        s = (px - sx) * ux + (py - sy) * uy
        cross = -(px - sx) * uy + (py - sy) * ux

        # Pure-pursuit carrot: forward along the line at zero lateral offset,
        # clamped at the pad so the drone is never aimed past it.
        carrot_s = min(length, max(0.0, s) + self.lookahead)
        carrot_x = sx + ux * carrot_s
        carrot_y = sy + uy * carrot_s

        # Carrot in the drone-heading frame (forward, right). Same convention
        # as GateExpert: right is negated so a target to the drone's right is
        # positive, matching YAW_STICK_SIGN.
        wx, wy = carrot_x - px, carrot_y - py
        cy, sy_ = math.cos(yaw_rad), math.sin(yaw_rad)
        fwd = wx * cy + wy * sy_
        right = -(-wx * sy_ + wy * cy)
        dist = math.hypot(fwd, right)

        d_pad = math.hypot(gx - px, gy - py)

        # Proportional position loop on the pad, capped at cruise. Note this
        # uses distance to the PAD, not to the carrot: the carrot is a heading
        # device and is a fixed lookahead ahead for most of the run, so pacing
        # off it would never slow the drone down.
        speed = min(self.cruise_speed, self.approach_gain * d_pad)
        if dist < 1e-6:
            v_fwd = v_right = 0.0
            bearing = 0.0
        else:
            v_fwd = speed * (fwd / dist)
            v_right = speed * (right / dist)
            bearing = math.atan2(right, fwd)

        throttle = v_fwd / self.max_forward_speed
        roll = v_right / self.max_lateral_speed
        if d_pad < self.yaw_hold_radius:
            steering = 0.0
        else:
            steering = YAW_STICK_SIGN * (self.yaw_gain * bearing) / self.max_yaw_rate_rad

        setpoint = self.altitude_setpoint_fn() if self.altitude_setpoint_fn else None
        cruise_alt = self.cruise_alt
        if cruise_alt is None:
            cruise_alt = float(setpoint) if setpoint is not None else 1.5
        z_ref = self._height_ref(d_pad, cruise_alt)
        height_ff = 0.0 if setpoint is None else (z_ref - float(setpoint))
        altitude = (
            height_ff + self.height_gain * (z_ref - float(pos[2]))
        ) / self.altitude_trim_range

        # PRE-clip altitude is recorded for the same reason GateExpert records
        # it: once np.clip has run there is no way to tell "asked for 0.9" from
        # "asked for 1.8", and on this task a clipped command is the signature
        # of a trim range too small to reach the ground.
        self.last_debug = {
            "d_pad": d_pad,
            "along_track": s,
            "cross_track": cross,
            "bearing_deg": math.degrees(bearing),
            "speed_cmd": speed,
            "z": float(pos[2]),
            "z_ref": z_ref,
            "setpoint": None if setpoint is None else float(setpoint),
            "alt_raw": float(altitude),
            "alt_clipped": float(min(1.0, max(-1.0, altitude))),
        }

        return np.clip(
            np.array([steering, throttle, roll, altitude], dtype=np.float32), -1.0, 1.0
        )

    def act_from_telemetry(self, telemetry):
        """Convenience for GateChaseEnv.last_telemetry (pos, att, vel, imu)."""
        pos = telemetry[0:3]
        yaw_deg = telemetry[5]
        return self.act(pos, math.radians(yaw_deg))
