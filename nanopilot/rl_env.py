"""State-based Gymnasium environment for RL gate flying.

GateChaseEnv wraps a running DroneGymEnv (BetaFlight SITL + Gazebo must
already be up) and exposes a low-dimensional privileged-state observation
built from Gazebo ground truth — no camera. This is the "teacher" side of the
teacher/student plan: a small MLP policy can learn gate flying from state in
a few hundred thousand real-time steps, then rl_collect.py flies the trained
policy to generate tubs for the existing vision behavior-cloning pipeline.

By default the policy controls three axes — [steering(yaw), throttle(pitch),
roll] — and altitude is left to the airframe's vertical damper, which only
engages while the altitude stick sits inside its deadband. Giving the policy
the altitude axis is available via control_altitude=True but is much harder
to learn: the axis is direct motor throttle around hover PWM, so exploration
climbs out of the world, and any nonzero command disengages the damper that
would otherwise arrest the climb. All the gates sit at one height anyway.

Observation (19 floats by default, mildly scaled; VecNormalize does the rest):
    [0:3]   target gate center, drone-heading frame (x/10, y/10, z/5)
    [3:5]   sin/cos of gate normal yaw relative to drone heading
    [5:8]   velocity, drone-heading frame (/5)
    [8:10]  roll, pitch (radians)
    [10:13] gyro rates (rad/s)
    [13:n]  previous action (3, or 4 with control_altitude)
    [n:n+3] following gate center, drone-heading frame (zeros if none)
"""

import logging
import math
import time

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# DroneGymEnv.run_threaded outputs[] layout with record_position,
# record_attitude, record_velocity, record_imu all True (same as eval_flight).
POS = slice(7, 10)
ATT = slice(10, 13)
VEL = slice(13, 16)
IMU = slice(16, 22)
TELEMETRY = slice(7, 22)  # pos + att + vel + imu, matches tub schema order


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GateChaseEnv(gym.Env):
    """Fly through the world's gates in order, scored from ground truth."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        drone_env,
        gates,
        *,
        rate_hz=20.0,
        max_episode_steps=1200,
        control_altitude=False,
        altitude_scale=0.3,
        start_alt=1.5,
        spawn=(0.0, 0.0, 0.5),
        spawn_jitter_xy=(1.0, 1.5),
        spawn_jitter_yaw=0.4,
        # Neutral throttle does not lift the drone off the ground (measured:
        # 5s at hover PWM, no movement), so takeoff needs real climb command.
        climb_altitude=0.6,
        takeoff_timeout=15.0,
        settle_vz=1.0,
        liftoff_vz=1.0,
        min_start_alt=0.8,
        reset_timeout=30.0,
        settle_s=1.0,
        crash_z=0.12,
        crash_tilt_deg=70.0,
        flyaway_z=12.0,
        bounds_xy=30.0,
        gate_half_width=0.9,
        gate_z_min=0.2,
        gate_z_max=3.0,
        progress_scale=2.0,
        pass_bonus=10.0,
        miss_penalty=3.0,
        complete_bonus=10.0,
        crash_penalty=10.0,
        time_penalty=0.02,
        action_rate_penalty=0.05,
        gyro_penalty=0.01,
    ):
        super().__init__()
        if not gates:
            raise ValueError("GateChaseEnv needs at least one gate")
        self._env = drone_env
        self.gates = gates
        self.period = 1.0 / float(rate_hz)
        self.max_episode_steps = int(max_episode_steps)
        # Altitude is normally not a policy axis at all: it is direct motor
        # throttle around hover PWM, and any nonzero command also disengages
        # the vertical damper, so the policy can start a climb it has no
        # authority to stop. Left at 0.0 the damper holds height for free.
        self.control_altitude = bool(control_altitude)
        self.altitude_scale = float(altitude_scale)
        self.n_actions = 4 if self.control_altitude else 3
        self.start_alt = float(start_alt)
        self.spawn = spawn
        self.spawn_jitter_xy = spawn_jitter_xy
        self.spawn_jitter_yaw = float(spawn_jitter_yaw)
        self.climb_altitude = float(climb_altitude)
        self.takeoff_timeout = float(takeoff_timeout)
        self.settle_vz = float(settle_vz)
        self.liftoff_vz = float(liftoff_vz)
        self.min_start_alt = float(min_start_alt)
        self.reset_timeout = float(reset_timeout)
        self.settle_s = float(settle_s)
        self.crash_z = float(crash_z)
        self.crash_tilt_deg = float(crash_tilt_deg)
        self.flyaway_z = float(flyaway_z)
        self.bounds_xy = float(bounds_xy)
        self.gate_half_width = float(gate_half_width)
        self.gate_z_min = float(gate_z_min)
        self.gate_z_max = float(gate_z_max)
        self.progress_scale = float(progress_scale)
        self.pass_bonus = float(pass_bonus)
        self.miss_penalty = float(miss_penalty)
        self.complete_bonus = float(complete_bonus)
        self.crash_penalty = float(crash_penalty)
        self.time_penalty = float(time_penalty)
        self.action_rate_penalty = float(action_rate_penalty)
        self.gyro_penalty = float(gyro_penalty)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
        # 16 fixed features + the previous action.
        self.obs_dim = 16 + self.n_actions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Per-episode state
        self._gate_idx = 0
        self._prev_gate_lx = None
        self._prev_dist = None
        self._prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self._steps = 0
        self._next_tick = None
        self._gates_passed = 0
        # Always four wide: the commands actually sent to the drone.
        self.last_applied_action = np.zeros(4, dtype=np.float32)

        # Latest raw outputs, exposed so rl_collect can write tub rows without
        # a second run_threaded call.
        self.last_image = None
        self.last_telemetry = None

    # ------------------------------------------------------------------
    # Gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generous retry count: a transient reset hiccup must not kill a
        # multi-hour training run.
        attempts = 8
        for attempt in range(attempts):
            try:
                self._teleport_to_spawn()
                self._settle(self.settle_s)
                if self._takeoff():
                    break
                logger.warning("Takeoff attempt %d timed out; re-resetting", attempt + 1)
            except RuntimeError as e:
                if not self._env.bf_armed:
                    # Retrying is pointless and just delays the supervisor's
                    # stack rebuild, which is the only thing that fixes this.
                    raise RuntimeError(
                        "BetaFlight lost arming; rebuild the stack"
                    ) from e
                logger.warning("Reset attempt %d failed: %s", attempt + 1, e)
        else:
            raise RuntimeError(f"Could not reset + take off after {attempts} attempts")

        self._gate_idx = 0
        self._gates_passed = 0
        self._prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self._steps = 0
        self._next_tick = time.time() + self.period

        outputs = self._env.run_threaded(0.0, 0.0, 0.0, 0.0)
        self._record_outputs(outputs)
        pos = outputs[POS]
        self._prev_gate_lx = self._gate_local_x(self.gates[self._gate_idx], pos)
        self._prev_dist = self._gate_dist(self.gates[self._gate_idx], pos)
        obs = self._build_obs(outputs)
        return obs, {"gates_passed": 0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        # What actually reaches the drone. rl_collect records this, not the
        # raw policy output, so distilled tubs carry the commands that were
        # really flown. Altitude stays 0.0 unless the policy owns that axis,
        # which keeps the vertical damper engaged.
        applied = np.zeros(4, dtype=np.float32)
        applied[:3] = action[:3]
        if self.control_altitude:
            applied[3] = action[3] * self.altitude_scale
        self.last_applied_action = applied
        self._env.run_threaded(*applied.tolist())
        # Real-time env: let the action act for one control period, then
        # sample the resulting state.
        now = time.time()
        sleep_s = self._next_tick - now
        if sleep_s > 0:
            time.sleep(sleep_s)
            self._next_tick += self.period
        else:
            # Fell behind (e.g. PPO update pause); don't try to catch up.
            self._next_tick = now + self.period
        outputs = self._env.run_threaded(*applied.tolist())
        self._record_outputs(outputs)
        pos = outputs[POS]
        att = outputs[ATT]
        imu = outputs[IMU]

        reward = -self.time_penalty
        reward -= self.action_rate_penalty * float(
            np.square(action - self._prev_action).sum()
        )
        reward -= self.gyro_penalty * float(
            math.sqrt(imu[3] ** 2 + imu[4] ** 2 + imu[5] ** 2)
        )
        terminated = False
        info = {}

        # --- gate progress / passing ---
        gate = self.gates[self._gate_idx]
        dist = self._gate_dist(gate, pos)
        reward += self.progress_scale * (self._prev_dist - dist)
        self._prev_dist = dist

        lx, ly, z = self._gate_local(gate, pos)
        crossed = (
            self._prev_gate_lx is not None
            and self._prev_gate_lx < 0.0 <= lx
        )
        if crossed:
            through = (
                abs(ly) <= self.gate_half_width
                and self.gate_z_min <= z <= self.gate_z_max
            )
            if through:
                reward += self.pass_bonus
                self._gates_passed += 1
            else:
                reward -= self.miss_penalty
            # Advance to the next gate either way so the drone never has to
            # loop back behind a missed gate.
            self._gate_idx += 1
            if self._gate_idx >= len(self.gates):
                terminated = True
                if through and self._gates_passed == len(self.gates):
                    reward += self.complete_bonus
                info["episode_result"] = "completed"
                self._gate_idx = len(self.gates) - 1  # keep obs valid
            else:
                self._prev_dist = self._gate_dist(self.gates[self._gate_idx], pos)
                self._prev_gate_lx = self._gate_local_x(self.gates[self._gate_idx], pos)
        else:
            self._prev_gate_lx = lx

        # --- crash / flyaway ---
        if not terminated:
            if pos[2] < self.crash_z or (
                abs(att[0]) > self.crash_tilt_deg or abs(att[1]) > self.crash_tilt_deg
            ):
                reward -= self.crash_penalty
                terminated = True
                info["episode_result"] = "crashed"
            elif pos[2] > self.flyaway_z or math.hypot(pos[0], pos[1]) > self.bounds_xy:
                reward -= self.crash_penalty
                terminated = True
                info["episode_result"] = "flyaway"

        self._steps += 1
        truncated = not terminated and self._steps >= self.max_episode_steps
        if terminated or truncated:
            self.hold_hover()
            info.setdefault("episode_result", "timeout")
            logger.info(
                "Episode end: %s, gates %d/%d, %d steps",
                info["episode_result"], self._gates_passed, len(self.gates), self._steps,
            )
        info["gates_passed"] = self._gates_passed

        self._prev_action = action
        obs = self._build_obs(outputs)
        return obs, float(reward), terminated, truncated, info

    def hold_hover(self):
        """Neutral sticks; altitude damper holds height. Used between rollouts."""
        self._env.run_threaded(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Geometry / observation

    def _gate_local(self, gate, pos):
        c, s = math.cos(gate["yaw"]), math.sin(gate["yaw"])
        dx, dy = pos[0] - gate["x"], pos[1] - gate["y"]
        return c * dx + s * dy, -s * dx + c * dy, pos[2]

    def _gate_local_x(self, gate, pos):
        return self._gate_local(gate, pos)[0]

    def _gate_dist(self, gate, pos):
        return math.sqrt(
            (pos[0] - gate["x"]) ** 2
            + (pos[1] - gate["y"]) ** 2
            + (pos[2] - gate["z"]) ** 2
        )

    def _build_obs(self, outputs):
        pos = outputs[POS]
        att = outputs[ATT]
        vel = outputs[VEL]
        imu = outputs[IMU]
        roll = math.radians(att[0])
        pitch = math.radians(att[1])
        yaw = math.radians(att[2])
        cy, sy = math.cos(yaw), math.sin(yaw)

        def to_heading(wx, wy):
            return wx * cy + wy * sy, -wx * sy + wy * cy

        gate = self.gates[self._gate_idx]
        gx, gy = to_heading(gate["x"] - pos[0], gate["y"] - pos[1])
        gz = gate["z"] - pos[2]
        rel_yaw = _wrap_angle(gate["yaw"] - yaw)
        vx, vy = to_heading(vel[0], vel[1])

        if self._gate_idx + 1 < len(self.gates):
            nxt = self.gates[self._gate_idx + 1]
            nx, ny = to_heading(nxt["x"] - pos[0], nxt["y"] - pos[1])
            nz = nxt["z"] - pos[2]
        else:
            nx = ny = nz = 0.0

        return np.array(
            [
                gx / 10.0,
                gy / 10.0,
                gz / 5.0,
                math.sin(rel_yaw),
                math.cos(rel_yaw),
                vx / 5.0,
                vy / 5.0,
                vel[2] / 5.0,
                roll,
                pitch,
                imu[3],
                imu[4],
                imu[5],
                *self._prev_action.tolist(),
                nx / 10.0,
                ny / 10.0,
                nz / 5.0,
            ],
            dtype=np.float32,
        )

    def _record_outputs(self, outputs):
        self.last_image = outputs[0]
        self.last_telemetry = list(outputs[TELEMETRY])

    # ------------------------------------------------------------------
    # Reset helpers (mirrors eval_flight's robust reset/takeoff)

    def _teleport_to_spawn(self):
        """Teleport to a jittered spawn and wait until the pose sticks AND the
        drone is calm.

        gz set_pose does not zero velocity, so after a fast flyaway the drone
        keeps its momentum through the teleport and instantly leaves bounds
        again. Re-pin the pose whenever it strays and let the altitude damper
        plus drag bleed the speed off before declaring the reset done.
        """
        x0, y0, z0 = self.spawn
        jx, jy = self.spawn_jitter_xy
        x = x0 + self.np_random.uniform(-jx, jx)
        y = y0 + self.np_random.uniform(-jy, jy)
        yaw = self.np_random.uniform(-self.spawn_jitter_yaw, self.spawn_jitter_yaw)
        self._env.spawn_pose = (x, y, z0, yaw)

        # The damper stays on: it is what arrests the drone after a teleport.
        # Teleport spikes are handled in DroneGymEnv._do_reset, which clears
        # the velocity estimators' history.
        self._wait_until_settled(x, y, z0, yaw)

    def _wait_until_settled(self, x, y, z0, yaw):
        """Re-pin the pose until the drone is at (x, y) and has stopped moving."""
        start = time.time()
        deadline = start + self.reset_timeout
        next_request = 0.0
        next_log = start + 3.0
        # Settled = small displacement over a 0.5s window. Telemetry velocity
        # is a raw per-message finite difference, too noisy to threshold.
        anchor_t = None
        anchor_pos = None

        while time.time() < deadline:
            # Neutral altitude, NOT low throttle: the damper only engages
            # inside the altitude deadband, so commanding a descent switches
            # off the very thing that arrests the drone after a teleport.
            outputs = self._env.run_threaded(0.0, 0.0, 0.0, 0.0)
            pos = tuple(outputs[POS])
            if time.time() >= next_log:
                vel = outputs[VEL]
                logger.info(
                    "Reset wait: pos=(%.2f, %.2f, %.2f) vel=(%.1f, %.1f, %.1f) "
                    "target=(%.2f, %.2f, %.2f) thr=%s",
                    pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                    x, y, z0, self._env.last_throttle_pwm,
                )
                next_log = time.time() + 3.0

            near = math.hypot(pos[0] - x, pos[1] - y) < 0.75 and -0.5 < pos[2] < 5.0
            if not near:
                anchor_t = None
                if time.time() >= next_request:
                    self._env.run_threaded(0.0, 0.0, 0.0, 0.0, web_reset=True)
                    # Requests go to a persistent worker and are ~free, so
                    # re-pin often enough that drift can't outrun the reset.
                    next_request = time.time() + 0.4
            elif anchor_t is None:
                anchor_t, anchor_pos = time.time(), pos
            elif time.time() - anchor_t >= 0.5:
                disp = math.sqrt(sum((pos[i] - anchor_pos[i]) ** 2 for i in range(3)))
                # ~2 m/s. The drone never goes truly still: at neutral it hovers
                # on the damper and bobs between 2-4m. Demanding real stillness
                # made every reset run its full timeout. All the episode start
                # needs is that it is near the spawn and not moving violently.
                if disp < 1.0:
                    logger.info(
                        "Reset done in %.1fs: spawn=(%.2f, %.2f) yaw=%.2f",
                        time.time() - start, x, y, yaw,
                    )
                    return
                anchor_t, anchor_pos = time.time(), pos
            time.sleep(0.1)
        raise RuntimeError("Timed out waiting for calm pose reset at spawn")

    def _settle(self, seconds):
        start = time.time()
        while time.time() - start < seconds:
            self._env.run_threaded(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.02)

    def _takeoff(self):
        """Climb to start_alt, then let the climb bleed off before handing over.

        A fixed settle is not enough: the climb command is well outside the
        damper deadband, so the drone crosses start_alt still accelerating and
        the episode used to begin mid-rocket — it would leave the ceiling in
        ~2s no matter what the policy did.
        """
        deadline = time.time() + self.takeoff_timeout
        # Throttle authority is only about +/-100 PWM and the band between
        # "will not leave the ground" and "rockets away" sits inside it, so
        # ramp up rather than guessing a constant. Crucially, cut the climb as
        # soon as the drone is rising: powering all the way to start_alt makes
        # it cross at ~8 m/s and coast to 9m even with the damper saturated.
        # Once airborne the damper carries it to its own hover equilibrium.
        climb = 0.2
        while time.time() < deadline:
            outputs = self._env.run_threaded(0.0, 0.0, 0.0, 0.0, direct_thrust=climb)
            if outputs[VEL][2] > self.liftoff_vz or outputs[POS][2] >= self.start_alt:
                return self._arrest_climb(deadline)
            climb = min(self.climb_altitude, climb + 0.01)
            time.sleep(0.05)

        z = self._env.run_threaded(0.0, 0.0, 0.0, 0.0)[POS][2]
        if not self._env.bf_armed:
            # No amount of throttle helps a disarmed board; let the caller's
            # retries fall through fast so the supervisor rebuilds the stack.
            raise RuntimeError("BetaFlight is not armed; cannot take off")
        logger.warning("Takeoff timed out at z=%.2f (target %.2f)", z, self.start_alt)
        return False

    def _arrest_climb(self, deadline):
        """Hold neutral (damper engaged) until vertical motion is small.

        Succeeds at whatever height the damper settles on, not at start_alt:
        the gates are 3m tall openings, so anything airborne is a usable
        episode start, and insisting on an exact altitude just fails resets.
        """
        calm_polls = 0
        while time.time() < deadline:
            outputs = self._env.run_threaded(0.0, 0.0, 0.0, 0.0)
            vz = outputs[VEL][2]
            z = outputs[POS][2]
            if abs(vz) < self.settle_vz and z >= self.min_start_alt:
                calm_polls += 1
                if calm_polls >= 4:
                    # With a PD altitude hold, anchor the setpoint on start_alt
                    # so every episode begins at the same height instead of
                    # wherever the climb happened to stop. Without the position
                    # term this is a no-op and the old behaviour stands.
                    if getattr(self._env, "altitude_hold_kp", 0.0) > 0.0:
                        self._env.set_altitude_setpoint(self.start_alt)
                        for _ in range(int(4.0 / 0.05)):
                            o = self._env.run_threaded(0.0, 0.0, 0.0, 0.0)
                            if (abs(o[POS][2] - self.start_alt) < 0.15
                                    and abs(o[VEL][2]) < 0.15):
                                break
                            time.sleep(0.05)
                        z = self._env.run_threaded(0.0, 0.0, 0.0, 0.0)[POS][2]
                    logger.info("Takeoff settled at z=%.2f", z)
                    return True
            else:
                calm_polls = 0
            if z > self.flyaway_z * 0.75:
                logger.warning("Climb not arrested by z=%.1f; re-resetting", z)
                return False
            time.sleep(0.05)
        logger.warning("Timed out arresting climb after takeoff")
        return False
