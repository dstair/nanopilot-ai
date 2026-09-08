"""
DroneGymEnv - DonkeyCar part wrapping BetaFlight SITL + Gazebo.

This part connects to a BetaFlight SITL instance via UDP RC channel packets
and accepts yaw, pitch, roll, and altitude inputs.

The semantic mapping is (BetaFlight Angle mode):
    steering [-1, 1]  ->  yaw rate
    throttle [-1, 1]  ->  forward pitch (tilt angle)
    roll [-1, 1]      ->  lateral roll (bank angle)
    altitude [-1, 1]  ->  motor throttle (direct power, no PID)

Camera sources (set DRONE_CAMERA_SOURCE in your drone_config_XXmm.py):
    "gz_transport"  Native macOS: subscribe to Gazebo Harmonic camera topic
                    via gz-transport subprocess + shared memory.
    "rtsp"          Docker mode: read RTSP stream from Gazebo Classic in container.

Requires:
    - BetaFlight SITL + Gazebo running, UDP 9004 for RC input
    - opencv-python-headless
    - gz-transport mode: gz-python (brew install gz-harmonic)
"""

import collections
import logging
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import gymnasium as gym
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np

try:
    from nanopilot.gz_telemetry import PoseTracker, init_pose_subscriber, init_sim_telemetry
except ImportError:
    from gz_telemetry import PoseTracker, init_pose_subscriber, init_sim_telemetry
from utils.betaflight import MSPClient, pack_rc_frame

logger = logging.getLogger(__name__)


_GZ_CAMERA_TOPIC_DEFAULT = (
    "/world/drone_course_65mm/model/betaloop_drone_cam_65mm"
    "/link/camera_link/sensor/camera/image"
)
# Yaw rate delivered per PWM microsecond of CH4 deflection, calibrated AT the
# 120 deg/s operating point. Re-measured 2026-09-06 with dense, UNWRAPPED yaw
# sampling (scripts/probe_velocity.py "yaw rate" leg):
#
#   100 PWM ->  36.8 deg/s      200 PWM -> 120.9 deg/s      400 PWM -> 423 deg/s
#
# The response is strongly expo, so no single constant is right everywhere; this
# one is chosen so DRONE_MAX_YAW_RATE=120 yields a ~200 PWM cap and full stick
# really is ~120 deg/s. The previous value (0.30) came from an old measurement
# and produced a 400 PWM cap, i.e. 423 deg/s at full stick -- 3.5x what the
# config asked for. That mis-scaling made every yaw command far more aggressive
# than intended, including the RL teachers'.
_YAW_DEG_PER_SEC_PER_PWM = 0.60


class DroneGymEnv(gym.Env):
    """
    DonkeyCar part that interfaces with BetaFlight SITL + Gazebo via UDP RC.

    Supports two camera sources (controlled by the camera_source parameter):
        "gz_transport"  - native macOS: gz-transport subscription (Gazebo Harmonic)
        "rtsp"          - Docker mode: OpenCV VideoCapture on RTSP stream

    This is a threaded part: update() runs the BetaFlight control loop in a
    background thread, while run_threaded() is called by the Vehicle loop to
    exchange yaw/pitch/roll/altitude commands and camera images.

    Usage in manage.py:
        gym = DroneGymEnv(cfg)
        V.add(gym, inputs=['steering', 'throttle', 'roll', 'altitude'],
              outputs=['cam/image_array', ...], threaded=True)
    """

    def __init__(
        self,
        rc_host="127.0.0.1",
        rc_port=9004,
        camera_source="gz_transport",
        gz_camera_topic=_GZ_CAMERA_TOPIC_DEFAULT,
        rtsp_url="rtsp://127.0.0.1:8554/live",
        max_pitch_angle=25.0,
        max_roll_angle=None,
        max_yaw_rate=90.0,
        hover_throttle=1500,
        throttle_range=300,
        throttle_scale=1.0,
        arm_channel=4,
        mode_channel=5,
        image_w=160,
        image_h=120,
        simulated_delay_ms=0,
        measure_loop_delay=False,
        loop_delay_log_interval=100,
        input_sensitivity=1.0,
        yaw_pwm_cap=None,
        yaw_throttle_feedforward=0.0,
        pitch_throttle_feedforward=0.0,
        altitude_hold_k=30.0,
        altitude_hold_kp=0.0,
        altitude_trim_range=0.0,
        altitude_hold_deadband=0.05,
        altitude_hold_enabled=True,
        velocity_control=False,
        max_forward_speed=2.0,
        max_lateral_speed=1.0,
        velocity_kp_fwd=0.6,
        velocity_kff_fwd=0.35,
        velocity_kp_lat=1.2,
        velocity_kff_lat=0.8,
        velocity_tau=0.1,
        angle_mode=True,
        record_position=False,
        record_attitude=False,
        record_velocity=False,
        record_imu=False,
        gz_world=None,
        gz_model_name=None,
        gz_imu_topic=None,
    ):
        super().__init__()
        self.rc_host = rc_host
        self.rc_port = rc_port
        self.camera_source = camera_source
        self.gz_camera_topic = gz_camera_topic
        self.msp = MSPClient(host=rc_host)
        self.rtsp_url = rtsp_url
        self.max_pitch_angle = float(max_pitch_angle)
        self.max_roll_angle = (
            self.max_pitch_angle if max_roll_angle is None else float(max_roll_angle)
        )
        self.max_yaw_rate = float(max_yaw_rate)
        self.hover_throttle = hover_throttle
        self.throttle_range = throttle_range
        self.throttle_scale = float(max(0.1, throttle_scale))
        self.arm_channel = arm_channel
        self.mode_channel = mode_channel
        self.image_w = image_w
        self.image_h = image_h
        self.simulated_delay_ms = simulated_delay_ms
        self.measure_loop_delay = measure_loop_delay
        self.loop_delay_log_interval = loop_delay_log_interval
        self.input_sensitivity = float(max(0.0, min(1.0, input_sensitivity)))
        self.altitude_hold_k = float(altitude_hold_k)
        # Position (proportional) gain, PWM per metre of height error. 0 keeps
        # the historical pure-velocity-damper behaviour.
        self.altitude_hold_kp = float(altitude_hold_kp)
        # How far the in-deadband altitude command may shift the held height,
        # in metres per unit of the (curve-shaped) command.
        self.altitude_trim_range = float(altitude_trim_range)
        self.altitude_hold_deadband = float(altitude_hold_deadband)
        # Height the hold regulates about; None until hold mode is first entered.
        self._alt_setpoint = None
        self.altitude_hold_enabled = bool(altitude_hold_enabled)

        # ---- Velocity control mode ----
        # When enabled the pitch/roll inputs stop being ANGLE commands and
        # become SPEED SETPOINTS in the drone-heading frame, closed by a P+FF
        # cascade against measured velocity. The altitude input likewise becomes
        # a LINEAR height trim instead of a curve-shaped thrust command.
        #
        # Why: every distillation failure in this project traced back to the
        # learned policy having to produce a precisely calibrated THRUST/tilt.
        # A 5 PWM altitude bias (2.5% of full scale, invisible in val_loss) was
        # fatal because the plant integrated it. With setpoints, a small command
        # error is a slightly wrong speed that the next frame corrects.
        self.velocity_control = bool(velocity_control)
        self.max_forward_speed = float(max_forward_speed)
        self.max_lateral_speed = float(max_lateral_speed)
        # Gains are PER AXIS because the airframe's authority is: pitch is
        # limited to 20 degrees and roll to 10, so an identical roll stick buys
        # half the acceleration. Measured open-loop 2026-09-06: pitch stick 0.30
        # -> 0.73 m/s and 0.50 -> 1.53, roll 0.60 -> 0.72 and 1.00 -> 1.52.
        # KFF is stick per m/s at cruise (the inverse of those slopes); KP sets
        # the loop bandwidth, ~2 rad/s on both axes given d(accel)/d(stick) of
        # 3.57 m/s^2 on pitch and 1.79 on roll.
        self.velocity_kp_fwd = float(velocity_kp_fwd)
        self.velocity_kff_fwd = float(velocity_kff_fwd)
        self.velocity_kp_lat = float(velocity_kp_lat)
        self.velocity_kff_lat = float(velocity_kff_lat)
        self.velocity_tau = float(velocity_tau)
        self._velocity_warned = False
        # Explicit thrust command from run_threaded(direct_thrust=...). Bypasses
        # the altitude hold entirely; takeoff/landing ramps need raw thrust and
        # must keep working unchanged in velocity mode.
        self._direct_thrust = None
        # Stick values the cascade produced, exposed for probes and logging.
        self.last_pitch_cmd = 0.0
        self.last_roll_cmd = 0.0
        self.angle_mode = bool(angle_mode)
        # Max CH4 (yaw) deflection in PWM microseconds from center (1500).
        # When no explicit PWM cap is provided, derive it from the desired yaw
        # rate using the 80mm sim measurement: CH4 +/-293us produced ~87 deg/s.
        # The value is clamped to BF's full stick range.
        if yaw_pwm_cap is None:
            yaw_pwm_cap = self.max_yaw_rate / _YAW_DEG_PER_SEC_PER_PWM
        self.yaw_pwm_cap = int(max(0, min(500, round(yaw_pwm_cap))))
        # Yaw-induced thrust feed-forward: yaw input creates net upward thrust
        # via motor-mixer ω² asymmetry. Subtract this many PWM from CH3 at
        # |steer|=1 (linear in |steer|). Tune via test_thrust damper-sim.
        self.yaw_throttle_feedforward = float(max(0.0, yaw_throttle_feedforward))
        # Pitch tilt reduces vertical thrust when altitude is neutral. Add
        # this many PWM at |pitch input|=1 to compensate forward/back flight.
        self.pitch_throttle_feedforward = float(max(0.0, pitch_throttle_feedforward))
        self.record_position = record_position
        self.record_attitude = record_attitude
        self.record_velocity = record_velocity
        self.record_imu = record_imu
        self.gz_world = gz_world or os.environ.get("GZ_WORLD", "drone_course_65mm")
        self.gz_model_name = gz_model_name or "betaloop_drone_cam_80mm"
        self.gz_imu_topic = gz_imu_topic

        # Shared state between threads
        self.steering = 0.0
        self.throttle = 0.0
        self.roll = 0.0
        self.altitude = 0.0  # altitude control input [-1, 1] → motor throttle
        # Arm signal from user/arm. None = legacy auto-arm (always armed after
        # boot disarm phase). True/False = explicit arm control (e.g. RT trigger).
        self.user_arm = None
        self._prev_user_arm = None
        self._explicit_arm_started_at = None
        self._reset_requested = False
        # Pose used by _do_reset teleports: (x, y, z, yaw). RL episode resets
        # overwrite this to randomize spawn; everything else keeps the default.
        self.spawn_pose = (0.0, 0.0, 0.5, 0.0)
        self.last_roll_pwm = 1500
        self.last_pitch_pwm = 1500
        self.last_yaw_pwm = 1500
        self.last_throttle_pwm = 1000
        self.last_arm_pwm = 1000
        self.last_mode_pwm = 2000 if self.angle_mode else 1000
        self.bf_armed = False
        self.bf_arming_flags = None
        self.bf_arming_disable_flags = []
        self.bf_active_modes = []
        self.frame = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        self.position = (0.0, 0.0, 0.0)
        self.attitude = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.imu = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.running = True
        self._rc_sock = None
        self._cap = None  # used by RTSP mode
        self._shm = None  # shared memory for gz-transport subprocess
        self._camera_proc = None  # gz_camera_worker subprocess
        self._last_seq = 0  # sequence counter for new-frame detection
        self._frame_size = image_h * image_w * 3
        self._frame_skip = 0
        self._logged_frame_stats = False

        # Simulated camera delay buffer: deque of (timestamp_ms, frame)
        self._delay_buffer = collections.deque()

        # Loop delay measurement
        self._loop_delays = collections.deque(maxlen=loop_delay_log_interval)
        self._last_loop_time = None
        self._loop_count = 0

        # Altitude hold (vertical velocity damper)
        self._pose_tracker = None
        if self.altitude_hold_enabled:
            self._pose_tracker = PoseTracker(
                k_pwm=self.altitude_hold_k,
                deadband=self.altitude_hold_deadband,
                enabled=self.altitude_hold_enabled,
                vel_tau=self.velocity_tau,
            )

        # RL Space Definitions
        # Actions: [steering, throttle, roll, altitude] all in [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # Observation: Image + IMU (simplified for this example)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(image_h, image_w, 3), dtype=np.uint8),
            "imu": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })

        logger.info(
            "Flight mode request: %s (CH%d/AUX%d PWM=%d)",
            "ANGLE" if self.angle_mode else "ACRO",
            self.mode_channel + 1,
            self.mode_channel - 3,
            self.last_mode_pwm,
        )

    def _start_camera(self):
        """Open RTSP stream from Gazebo (Docker mode)."""
        logger.info("Opening RTSP camera stream: %s", self.rtsp_url)

        self._cap = cv2.VideoCapture(self.rtsp_url)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info("RTSP stream opened successfully")
            return

        # Fallback: try GStreamer pipeline
        gst = (
            f"rtspsrc location={self.rtsp_url} latency=0 "
            "! rtph264depay ! h264parse ! avdec_h264 "
            "! videoconvert ! appsink drop=true max-buffers=1 sync=false"
        )
        logger.info("Trying GStreamer pipeline: %s", gst)
        self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if self._cap.isOpened():
            logger.info("GStreamer pipeline opened successfully")
            return

        logger.warning("Could not open RTSP stream. Images will be blank.")

    def _capture_frame(self):
        """Capture a single frame from the RTSP stream and resize (Docker mode)."""
        if self._cap is None or not self._cap.isOpened():
            return

        ret, raw_frame = self._cap.read()
        if ret and raw_frame is not None:
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.image_w, self.image_h))
            self.frame = resized

    def _start_gz_camera(self):
        """Launch gz_camera_worker subprocess for gz-transport camera capture."""
        logger.info(
            "Starting gz-transport camera subprocess for topic: %s",
            self.gz_camera_topic,
        )

        shm_size = 1 + self._frame_size
        self._shm = SharedMemory(create=True, size=shm_size)
        self._shm.buf[0] = 0
        logger.info("Created shared memory '%s' (%d bytes)", self._shm.name, shm_size)

        worker_path = str(Path(__file__).parent / "gz_camera_worker.py")

        self._camera_proc = subprocess.Popen(
            [
                "uv",
                "run",
                "--env-file",
                ".env",
                "python",
                worker_path,
                self.gz_camera_topic,
                str(self.image_w),
                str(self.image_h),
                self._shm.name,
            ],
            env=os.environ.copy(),
        )
        logger.info("Camera worker started (PID %d)", self._camera_proc.pid)

    def _read_gz_frame(self):
        """Read the latest frame from shared memory if a new one is available."""
        if self._shm is None:
            return
        seq = self._shm.buf[0]
        if seq != self._last_seq:
            self._last_seq = seq
            frame_bytes = bytes(self._shm.buf[1 : 1 + self._frame_size])
            self.frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                self.image_h, self.image_w, 3
            )
            if not self._logged_frame_stats:
                logger.info(
                    "Camera frame stats: min=%d max=%d mean=%.1f",
                    int(self.frame.min()),
                    int(self.frame.max()),
                    float(self.frame.mean()),
                )
                if self.frame.max() == 0:
                    logger.warning(
                        "Camera frames are arriving but are all black. "
                        "Check Gazebo sensor rendering and camera pose/topic."
                    )
                self._logged_frame_stats = True
            if self._frame_skip == 0:
                logger.info(
                    "Frame updated (seq=%d, shape=%s, dtype=%s)",
                    seq,
                    self.frame.shape,
                    self.frame.dtype,
                )
            self._frame_skip = (self._frame_skip + 1) % 30

    def _send_rc(self, channels):
        """Send RC channel packet to BetaFlight SITL.

        Packet format: 8-byte timestamp (double) + 16 × 2-byte channels (uint16)
        = 40 bytes total.
        """
        try:
            self._rc_sock.sendto(pack_rc_frame(channels), (self.rc_host, self.rc_port))
        except OSError as e:
            logger.warning("RC send failed: %s", e)

    def _map_controls_to_rc(self):
        """Convert yaw/pitch/roll/altitude [-1,1] to 16 RC PWM channels [1000-2000].

        BetaFlight Angle mode channel mapping:
            CH1 (roll):     1500 + roll * 500 * sensitivity * roll/pitch angle ratio
            CH2 (pitch):    1500 + throttle * 500 * sensitivity (forward tilt)
            CH3 (throttle): bipolar — altitude in [-1,1] → hover_throttle ± throttle_range.
                            altitude=0 → hover PWM (drone holds altitude in sim where
                            thrust is deterministic); altitude=+1 → hover+range (climb);
                            altitude=-1 → hover-range (descend).
            CH4 (yaw):      1500 + steering * yaw_pwm_cap (capped independently
                            of input_sensitivity to limit yaw-induced climb)
            CH5 (AUX1):     2000 = armed, 1000 = disarmed
            CH6 (AUX2):     2000 = angle mode active
            CH7-16:         1000 (unused)
        """
        channels = [1000] * 16

        # Sensitivity scales pitch stick deflection. Roll can be tuned
        # independently as an angle ratio while preserving existing behavior
        # when max_roll_angle == max_pitch_angle.
        pitch_deflection = 500 * self.input_sensitivity
        roll_angle_ratio = self.max_roll_angle / max(1.0, self.max_pitch_angle)
        roll_deflection = pitch_deflection * roll_angle_ratio

        # Velocity cascade: turn speed setpoints into angle sticks.
        #
        # throttle -> forward speed setpoint, roll -> rightward speed setpoint,
        # both as a fraction of the configured max. A proportional term closes
        # the loop on measured velocity and a feed-forward term supplies the
        # tilt that holds cruise speed against drag, so the P term only has to
        # correct the residual.
        pitch_cmd = self.throttle
        roll_cmd = self.roll
        if self.velocity_control:
            tracker = self._pose_tracker
            if tracker is None or not tracker.enabled:
                # No velocity feedback available: the cascade would be pure
                # open-loop tilt. Fall back to the direct angle mapping, which
                # is at least a known-bounded interpretation of the same input.
                if not self._velocity_warned:
                    logger.error(
                        "velocity_control requested but no pose tracker; "
                        "falling back to direct angle control"
                    )
                    self._velocity_warned = True
            else:
                vel = tracker.get_velocity()
                yaw = tracker.get_yaw()
                cy, sy = math.cos(yaw), math.sin(yaw)
                # Drone-heading frame, matching GateChaseEnv._build_obs. Its y
                # is LEFT-positive; the roll stick is right-positive, so flip.
                v_fwd = vel[0] * cy + vel[1] * sy
                v_right = -(-vel[0] * sy + vel[1] * cy)
                vx_cmd = max(-1.0, min(1.0, self.throttle)) * self.max_forward_speed
                vy_cmd = max(-1.0, min(1.0, self.roll)) * self.max_lateral_speed
                pitch_cmd = (
                    self.velocity_kp_fwd * (vx_cmd - v_fwd)
                    + self.velocity_kff_fwd * vx_cmd
                )
                roll_cmd = (
                    self.velocity_kp_lat * (vy_cmd - v_right)
                    + self.velocity_kff_lat * vy_cmd
                )
        pitch_cmd = max(-1.0, min(1.0, pitch_cmd))
        roll_cmd = max(-1.0, min(1.0, roll_cmd))
        self.last_pitch_cmd = pitch_cmd
        self.last_roll_cmd = roll_cmd

        # CH1: roll (lateral bank)
        channels[0] = int(max(1000, min(2000, 1500 + roll_cmd * roll_deflection)))

        # CH2: pitch (forward tilt)
        channels[1] = int(max(1000, min(2000, 1500 + pitch_cmd * pitch_deflection)))

        # CH3: motor throttle — bipolar around hover.
        alt = max(-1.0, min(1.0, self.altitude))
        alt_scaled = pow(abs(alt), self.throttle_scale) * (1 if alt >= 0 else -1)

        # Altitude hold. Two modes, split by the command deadband:
        #
        #   |alt| >= deadband : an explicit climb/descent. Direct thrust, and the
        #                       held height follows the drone (no hold applied).
        #   |alt| <  deadband : hold mode. With altitude_hold_kp > 0 this is a PD
        #                       regulator about a FROZEN setpoint, and the small
        #                       command becomes a bounded position trim instead
        #                       of raw thrust.
        #
        # Why the position term matters: a pure velocity damper (-K*vz) leaves the
        # plant with no height regulation at all, so ANY steady thrust bias B
        # settles at vz = B/K and integrates into the floor. A distilled BC
        # student was measured at a -5 PWM bias (2.5% of full scale, invisible in
        # val_loss) which sank it 1.4m in 30s, reliably fatal. With a position
        # term the same bias produces a bounded offset of B/Kp instead.
        alt_hold_bias = 0
        alt_thrust = alt_scaled * self.throttle_range
        tracker = self._pose_tracker
        direct = self._direct_thrust
        if direct is not None:
            # Explicit thrust command (takeoff / landing ramps). Bypasses the
            # hold exactly the way an over-deadband stick used to, including the
            # power curve, so the calibrated takeoff ramp behaves identically.
            d = max(-1.0, min(1.0, float(direct)))
            d_scaled = pow(abs(d), self.throttle_scale) * (1 if d >= 0 else -1)
            alt_thrust = d_scaled * self.throttle_range
            alt_hold_bias = 0
            if tracker is not None and tracker.enabled:
                self._alt_setpoint = tracker.latest[2]
        elif (
            self.velocity_control
            and tracker is not None
            and tracker.enabled
            and self.altitude_hold_kp > 0.0
        ):
            # Velocity mode: the altitude input is a LINEAR height trim over the
            # whole axis. No deadband and no power curve, which is the point --
            # sign(a)*|a|**0.2 is near-vertical at zero (alt=-0.0001 was already
            # -15.8 PWM), so tiny regression errors became large thrust errors.
            # Here the same error is a proportional, bounded height offset.
            z = tracker.latest[2]
            if self._alt_setpoint is None:
                self._alt_setpoint = z
            z_target = self._alt_setpoint + alt * self.altitude_trim_range
            alt_hold_bias = int(
                -self.altitude_hold_kp * (z - z_target)
                - tracker.k_pwm * tracker.get_vz()
            )
            alt_thrust = 0.0
        elif tracker is not None and tracker.enabled:
            in_hold = abs(alt) < self.altitude_hold_deadband
            if not in_hold:
                # Commanded climb/descent: let it fly, and re-anchor the hold to
                # wherever it ends up.
                self._alt_setpoint = tracker.latest[2]
            elif self.altitude_hold_kp > 0.0:
                z = tracker.latest[2]
                if self._alt_setpoint is None:
                    self._alt_setpoint = z
                # The in-deadband command trims the target height rather than
                # adding thrust, so a constant command cannot integrate.
                z_target = self._alt_setpoint + alt_scaled * self.altitude_trim_range
                alt_hold_bias = int(
                    -self.altitude_hold_kp * (z - z_target)
                    - tracker.k_pwm * tracker.get_vz()
                )
                alt_thrust = 0.0
            else:
                # Legacy behaviour: velocity damper only, command adds thrust.
                alt_hold_bias = int(-tracker.k_pwm * tracker.get_vz())
        alt_hold_bias = max(-200, min(200, alt_hold_bias))

        steer = max(-1.0, min(1.0, self.steering))

        # Yaw→throttle feed-forward (always subtractive — both yaw directions
        # add upward thrust via mixer ω² asymmetry). Empirically the excess
        # thrust is near-constant across yaw magnitude (BF yaw PID I-term winds
        # up to similar motor differential regardless of stick magnitude — see
        # test_thrust damper-sim 2026-04-28), so a flat step works best.
        yaw_ff_bias = 0
        if self.yaw_throttle_feedforward > 0 and abs(steer) > 0.01:
            yaw_ff_bias = -int(self.yaw_throttle_feedforward)
        yaw_ff_bias = max(-200, min(0, yaw_ff_bias))

        # Keyed off the commanded TILT, not self.throttle, which is a speed
        # setpoint in velocity mode. In velocity mode the hold is always active
        # (no deadband on the trim axis), so the tilt compensation always applies.
        pitch_ff_bias = 0
        ff_active = self.velocity_control or abs(alt) < self.altitude_hold_deadband
        if self.pitch_throttle_feedforward > 0 and direct is None and ff_active:
            pitch_ff_bias = int(self.pitch_throttle_feedforward * abs(pitch_cmd))
        pitch_ff_bias = max(0, min(200, pitch_ff_bias))

        throttle_pwm = int(
            max(
                1000,
                min(
                    2000,
                    self.hover_throttle
                    + alt_thrust
                    + alt_hold_bias
                    + yaw_ff_bias
                    + pitch_ff_bias,
                ),
            )
        )
        if self.user_arm is True and self._explicit_arm_started_at is not None:
            if time.time() - self._explicit_arm_started_at < 1.0:
                throttle_pwm = 1000
        channels[2] = throttle_pwm

        # CH4: yaw — capped separately from pitch sensitivity. At hover
        # throttle, yaw deflection adds net thrust via ω² mixer asymmetry;
        # a small cap keeps the yaw-induced climb manageable.
        channels[3] = int(max(1000, min(2000, 1500 + steer * self.yaw_pwm_cap)))

        # CH5 (AUX1): armed. user_arm=None preserves legacy "always armed
        # after boot disarm phase" behavior; explicit False disarms.
        channels[self.arm_channel] = 1000 if self.user_arm is False else 2000

        # CH6 (AUX2): 2000 = Angle (self-leveling), 1000 = Acro (rate mode)
        channels[self.mode_channel] = 2000 if self.angle_mode else 1000

        # Snapshot for telemetry / UI display
        self.last_roll_pwm = channels[0]
        self.last_pitch_pwm = channels[1]
        self.last_throttle_pwm = channels[2]
        self.last_yaw_pwm = channels[3]
        self.last_arm_pwm = channels[self.arm_channel]
        self.last_mode_pwm = channels[self.mode_channel]

        return channels

    def _query_arming_flags(self, timeout=0.5):
        """Query MSP_STATUS_EX and return BF's arming-disable + active-mode state.
        Now delegated to MSPClient for repository-wide reuse.
        """
        return self.msp.query_status(timeout=timeout)

    def _update_bf_status(self, flags, active_arming, active_modes):
        if flags is None and not active_modes:
            return
        if flags is not None:
            self.bf_arming_flags = int(flags)
            self.bf_arming_disable_flags = list(active_arming or [])
        if active_modes is not None:
            self.bf_active_modes = list(active_modes or [])
            self.bf_armed = "ARM" in self.bf_active_modes

    def _betaflight_loop(self):
        """Main control loop: arm BetaFlight and send RC commands at 50Hz.

        Plain synchronous loop — no asyncio needed.
        """
        self._rc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info("RC UDP socket ready → %s:%d", self.rc_host, self.rc_port)

        # Reusable channel templates. The disarm packet holds the sticks
        # centered with throttle low and AUX1 low so BF clears any latched
        # FAILSAFE / ARM_SWITCH disable bits before we toggle AUX1 high.
        disarm_channels = [1000] * 16
        disarm_channels[0] = 1500  # roll centered
        disarm_channels[1] = 1500  # pitch centered
        disarm_channels[2] = 1000  # throttle low
        disarm_channels[3] = 1500  # yaw centered
        disarm_channels[self.mode_channel] = 2000 if self.angle_mode else 1000

        arm_channels = list(disarm_channels)
        arm_channels[self.arm_channel] = 2000  # AUX1 armed

        # Disarm + arm sequence with arming-flags verification and retry.
        # The FAILSAFE bit takes a long time to clear after BF spent the
        # pre-launch window with no RC, so we hold disarm for 3s on the first
        # attempt; later attempts fall back to 1s since by then RX is healthy.
        max_attempts = 3
        armed_ok = False
        for attempt in range(1, max_attempts + 1):
            disarm_secs = 3.0 if attempt == 1 else 1.0
            disarm_iters = int(disarm_secs / 0.02)
            logger.info(
                "Arm attempt %d/%d: disarm %.1fs, then arm 1s...",
                attempt,
                max_attempts,
                disarm_secs,
            )
            for _ in range(disarm_iters):
                if not self.running:
                    return
                self._send_rc(disarm_channels)
                time.sleep(0.02)
            for _ in range(50):
                if not self.running:
                    return
                self._send_rc(arm_channels)
                time.sleep(0.02)

            flags, active_arming, mode_bits, active_modes = (
                self._query_arming_flags()
            )
            self._update_bf_status(flags, active_arming, active_modes)
            if flags is None:
                logger.warning(
                    "Arm attempt %d: MSP query failed; assuming BF unreachable",
                    attempt,
                )
                continue
            if flags == 0:
                logger.info(
                    "Arm attempt %d: BF armed (arming_flags=0, modes=%s, "
                    "requested=%s)",
                    attempt,
                    active_modes or "(none)",
                    "ANGLE" if self.angle_mode else "ACRO",
                )
                if self.angle_mode and "ANGLE" not in (active_modes or []):
                    logger.warning(
                        "ANGLE mode requested but not active in BF "
                        "(mode_bits=0x%08x). Check aux bindings in eeprom — "
                        "see scripts/start.sh CLI block.",
                        mode_bits or 0,
                    )
                if not self.angle_mode and "ANGLE" in (active_modes or []):
                    logger.warning(
                        "ACRO mode requested but BF still reports ANGLE active "
                        "(mode_bits=0x%08x, CH%d PWM=%d). Check aux bindings "
                        "and confirm no other ANGLE range is active.",
                        mode_bits or 0,
                        self.mode_channel + 1,
                        arm_channels[self.mode_channel],
                    )
                armed_ok = True
                break
            logger.warning(
                "Arm attempt %d: arming_flags=0x%08x %s — retrying",
                attempt,
                flags,
                active_arming,
            )

        if not armed_ok:
            logger.error(
                "Failed to arm BetaFlight after %d attempts. "
                "Entering control loop anyway — motors will stay at 0 until "
                "the disable flags clear.",
                max_attempts,
            )
        logger.info("Entering control loop at 50Hz")

        # Phase 3: Control loop
        loop_count = 0
        while self.running:
            if self._reset_requested:
                self._reset_requested = False
                self._do_reset()
            channels = self._map_controls_to_rc()
            self._send_rc(channels)

            # Log control values every ~2s
            loop_count += 1
            if loop_count % 50 == 0:
                flags, active_arming, _mode_bits, active_modes = (
                    self._query_arming_flags(timeout=0.05)
                )
                self._update_bf_status(flags, active_arming, active_modes)
            if loop_count % 100 == 0:
                logger.info(
                    "rc: yaw=%.2f pitch=%.2f roll=%.2f alt=%.2f mode=%s → "
                    "roll_pwm=%d pitch=%d yaw=%d throttle=%d ch%d=%d ch%d=%d",
                    self.steering,
                    self.throttle,
                    self.roll,
                    self.altitude,
                    "ANGLE" if self.angle_mode else "ACRO",
                    channels[0],
                    channels[1],
                    channels[3],
                    channels[2],
                    self.arm_channel + 1,
                    channels[self.arm_channel],
                    self.mode_channel + 1,
                    channels[self.mode_channel],
                )

            # RTSP mode: poll for frames every other iteration
            if self.camera_source == "rtsp":
                self._frame_skip += 1
                if self._frame_skip % 2 == 0:
                    self._capture_frame()

            time.sleep(0.02)  # 50 Hz

        # Phase 4: Disarm on exit
        logger.info("Disarming BetaFlight...")
        disarm_channels[self.arm_channel] = 1000
        disarm_channels[2] = 1000
        for _ in range(25):
            self._send_rc(disarm_channels)
            time.sleep(0.02)

        self._rc_sock.close()
        self._rc_sock = None
        logger.info("BetaFlight disarmed and socket closed")

    def update(self):
        """
        Background thread entry point.
        Starts camera, then runs the BetaFlight control loop.
        """
        needs_telemetry = (
            self.record_position
            or self.record_attitude
            or self.record_velocity
            or self.record_imu
        )
        if needs_telemetry:
            self._sim_telemetry = init_sim_telemetry(
                self.gz_world,
                self.gz_model_name,
                imu_topic=self.gz_imu_topic,
            )
        else:
            self._sim_telemetry = None

        if self.altitude_hold_enabled:
            logger.info("Subscribing to pose topic for altitude hold...")
            if not init_pose_subscriber(self.gz_world):
                logger.warning("Pose subscription failed, altitude hold disabled")
                self._pose_tracker = None

        if self.camera_source == "rtsp":
            self._start_camera()
        else:
            self._start_gz_camera()

        try:
            self._betaflight_loop()
        except Exception as e:
            logger.error("BetaFlight loop error: %s", e)

    def set_altitude_setpoint(self, z):
        """Anchor the altitude hold at a specific height (metres).

        Takeoff routines need a REPEATABLE episode start height. Left alone the
        hold freezes wherever the climb happens to be when the command re-enters
        the deadband, and climb momentum makes that vary by ~1m run to run.
        Requires DRONE_ALTITUDE_HOLD_KP > 0; a no-op otherwise.
        """
        self._alt_setpoint = float(z)

    def get_altitude_setpoint(self):
        return self._alt_setpoint

    def _do_reset(self):
        """Teleport the drone back to spawn_pose via a gz_reset_worker subprocess.

        The set_pose_vector request is made from a fresh helper process rather
        than in-process: gz-transport service requests from a process with
        torch loaded time out reliably (same conflict that puts the camera in
        gz_camera_worker), and enough timed-out in-process requests wedge the
        server's service handler for everyone. Fire-and-forget; callers verify
        the pose via telemetry and re-request if it didn't stick.
        """
        try:
            x, y, z, yaw = self.spawn_pose
            proc = self._gz_reset_worker()
            if proc is None:
                return
            proc.stdin.write(f"{float(x)} {float(y)} {float(z)} {float(yaw)}\n")
            proc.stdin.flush()
            # Drop the velocity estimators' previous sample: otherwise the
            # position jump reads as a huge phantom velocity and the altitude
            # damper reacts to a movement that never happened.
            if self._pose_tracker is not None:
                self._pose_tracker.reset_history()
            if getattr(self, "_sim_telemetry", None) is not None:
                self._sim_telemetry.reset_history()
            # Forget the held height too: a setpoint from the previous episode
            # would make the hold fight the new spawn altitude.
            self._alt_setpoint = None
            logger.info(
                "Reset requested: %s -> (%.2f, %.2f, %.2f, yaw=%.2f)",
                self.gz_model_name, x, y, z, yaw,
            )
        except Exception as e:
            logger.error("Reset pose error: %s", e)

    def _gz_reset_worker(self):
        """Return the persistent reset worker, starting/restarting as needed."""
        proc = getattr(self, "_gz_reset_proc", None)
        if proc is not None and proc.poll() is None:
            return proc
        if proc is not None:
            logger.warning("Reset worker exited (rc=%s); restarting", proc.poll())

        worker = str(Path(__file__).parent / "gz_reset_worker.py")
        log_path = Path(__file__).parent.parent / "logs" / "gz_reset_worker.log"
        log_path.parent.mkdir(exist_ok=True)
        self._gz_reset_log = open(log_path, "a")
        self._gz_reset_proc = subprocess.Popen(
            [
                "uv", "run", "--env-file", ".env", "python", worker,
                self.gz_world, self.gz_model_name, "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=self._gz_reset_log,
            stderr=self._gz_reset_log,
            text=True,
            env=os.environ.copy(),
        )
        logger.info("Reset worker started (PID %d)", self._gz_reset_proc.pid)
        return self._gz_reset_proc

    def set_altitude_hold(self, enabled):
        """Enable/disable the vertical damper at runtime.

        Callers that teleport the drone should switch this off first: the
        damper reacts to the resulting fall with near-max throttle, which
        turns a reset into a runaway climb.
        """
        if self._pose_tracker is not None:
            self._pose_tracker.enabled = bool(enabled)

    def update_flight_config(self, config_update):
        """Apply runtime-editable flight config values from the web UI."""
        if not isinstance(config_update, dict):
            return

        changed = []
        if "max_pitch_angle" in config_update:
            value = float(config_update["max_pitch_angle"])
            self.max_pitch_angle = max(1.0, min(80.0, value))
            changed.append(f"max_pitch_angle={self.max_pitch_angle:.1f}")
        if "max_roll_angle" in config_update:
            value = float(config_update["max_roll_angle"])
            self.max_roll_angle = max(1.0, min(80.0, value))
            changed.append(f"max_roll_angle={self.max_roll_angle:.1f}")
        if "max_yaw_rate" in config_update:
            value = float(config_update["max_yaw_rate"])
            self.max_yaw_rate = max(1.0, min(360.0, value))
            self.yaw_pwm_cap = int(
                max(0, min(500, round(self.max_yaw_rate / _YAW_DEG_PER_SEC_PER_PWM)))
            )
            changed.append(
                f"max_yaw_rate={self.max_yaw_rate:.1f} yaw_pwm_cap={self.yaw_pwm_cap}"
            )

        if changed:
            logger.info("Updated flight config: %s", ", ".join(changed))

    def run_threaded(
        self,
        steering,
        throttle,
        roll,
        altitude,
        user_arm=None,
        user_reset=None,
        web_reset=None,
        config_update=None,
        direct_thrust=None,
    ):
        """
        Called by the DonkeyCar Vehicle loop each frame.

        :param steering: normalized steering [-1, 1], mapped to yaw rate
        :param throttle: normalized throttle [-1, 1], mapped to forward pitch
        :param roll: normalized roll [-1, 1], mapped to lateral bank
        :param altitude: normalized altitude [-1, 1], mapped to motor throttle
        :param user_arm: optional bool. None = legacy auto-arm; True/False =
                         explicit arm switch (e.g. Xbox RT deadman).
        :param user_reset: optional bool. True = one-shot pose reset from LT.
        :param web_reset: optional bool. True = one-shot pose reset from web UI.
        :param config_update: optional runtime flight config update from web UI.
        :param direct_thrust: optional raw thrust command in [-1, 1], bypassing
                              the altitude hold. Takeoff and landing ramps need
                              real thrust; in velocity mode the `altitude`
                              argument is a height trim and cannot provide it.
        :return: camera image array + optional telemetry values
        """
        self.update_flight_config(config_update)

        # Loop delay measurement
        if self.measure_loop_delay:
            now = time.time()
            if self._last_loop_time is not None:
                delta_ms = (now - self._last_loop_time) * 1000.0
                self._loop_delays.append(delta_ms)
                self._loop_count += 1
                if self._loop_count % self.loop_delay_log_interval == 0:
                    delays = list(self._loop_delays)
                    if delays:
                        avg = sum(delays) / len(delays)
                        lo = min(delays)
                        hi = max(delays)
                        logger.info(
                            "loop delay: avg=%.1fms min=%.1fms max=%.1fms", avg, lo, hi
                        )
            self._last_loop_time = now

        # Set control inputs
        if steering is None:
            steering = 0.0
        if throttle is None:
            throttle = 0.0
        if roll is None:
            roll = 0.0
        if altitude is None:
            altitude = 0.0

        self.steering = float(steering)
        self.throttle = float(throttle)
        self.roll = float(roll)
        self.altitude = float(altitude)
        self._direct_thrust = None if direct_thrust is None else float(direct_thrust)
        self.user_arm = user_arm if isinstance(user_arm, bool) else None
        if self.user_arm is True and self._prev_user_arm is not True:
            self._explicit_arm_started_at = time.time()
            logger.info("Explicit arm requested; holding throttle low for 1.0s")
        if self.user_arm is not True:
            self._explicit_arm_started_at = None
        self._prev_user_arm = self.user_arm

        # One-shot reset from Xbox LT (handled in background loop).
        if user_reset is True or web_reset is True:
            self._reset_requested = True

        # Read camera frame
        if self.camera_source == "gz_transport":
            self._read_gz_frame()

        # Update pose tracker for altitude hold damper
        if self._pose_tracker is not None:
            self._pose_tracker.update()

        if getattr(self, "_sim_telemetry", None) is not None:
            self.position = self._sim_telemetry.position
            self.attitude = self._sim_telemetry.attitude
            self.velocity = self._sim_telemetry.velocity
            self.imu = self._sim_telemetry.imu

        # Simulated camera delay
        current_frame = self.frame
        if self.simulated_delay_ms > 0:
            now_ms = time.time() * 1000.0
            self._delay_buffer.append((now_ms, current_frame))
            # Find the frame that is delay_ms old
            target_time = now_ms - self.simulated_delay_ms
            output_frame = self._delay_buffer[0][1]  # oldest as fallback
            while (
                len(self._delay_buffer) > 1 and self._delay_buffer[1][0] <= target_time
            ):
                self._delay_buffer.popleft()
            output_frame = self._delay_buffer[0][1]
        else:
            output_frame = current_frame

        outputs = [
            output_frame,
            self.last_roll_pwm,
            self.last_pitch_pwm,
            self.last_yaw_pwm,
            self.last_throttle_pwm,
            self.last_arm_pwm,
            self.last_mode_pwm,
        ]

        if self.record_position:
            outputs += [self.position[0], self.position[1], self.position[2]]
        if self.record_attitude:
            outputs += [self.attitude[0], self.attitude[1], self.attitude[2]]
        if self.record_velocity:
            outputs += [self.velocity[0], self.velocity[1], self.velocity[2]]
        if self.record_imu:
            outputs += [
                self.imu[0],
                self.imu[1],
                self.imu[2],
                self.imu[3],
                self.imu[4],
                self.imu[5],
            ]

        outputs += [
            self.bf_armed,
            self.bf_arming_flags,
            ",".join(self.bf_arming_disable_flags),
            ",".join(self.bf_active_modes),
        ]

        return outputs

    def shutdown(self):
        """Stop the background thread and clean up."""
        logger.info("Shutting down DroneGymEnv...")
        self.running = False
        time.sleep(1.0)
        if self._cap is not None:
            self._cap.release()
        if self._camera_proc is not None:
            logger.info("Terminating camera worker (PID %d)...", self._camera_proc.pid)
            self._camera_proc.terminate()
            try:
                self._camera_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._camera_proc.kill()
            self._camera_proc = None
        reset_proc = getattr(self, "_gz_reset_proc", None)
        if reset_proc is not None:
            logger.info("Terminating reset worker (PID %d)...", reset_proc.pid)
            try:
                reset_proc.stdin.close()
            except Exception:
                pass
            reset_proc.terminate()
            try:
                reset_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                reset_proc.kill()
            self._gz_reset_proc = None
            log = getattr(self, "_gz_reset_log", None)
            if log is not None:
                log.close()
                self._gz_reset_log = None
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                # The camera worker/resource_tracker can win this cleanup race
                pass
    