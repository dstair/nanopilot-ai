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
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np

from gz_telemetry import PoseTracker, init_pose_subscriber, init_sim_telemetry
from utils.betaflight import MSPClient, pack_rc_frame

logger = logging.getLogger(__name__)


_GZ_CAMERA_TOPIC_DEFAULT = (
    "/world/drone_course_65mm/model/betaloop_drone_cam_65mm"
    "/link/camera_link/sensor/camera/image"
)
_YAW_DEG_PER_SEC_PER_PWM = 0.30


class DroneGymEnv:
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
        altitude_hold_deadband=0.05,
        altitude_hold_enabled=True,
        angle_mode=True,
        record_position=False,
        record_attitude=False,
        record_velocity=False,
        record_imu=False,
        gz_world=None,
        gz_model_name=None,
        gz_imu_topic=None,
    ):
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
        self.altitude_hold_deadband = float(altitude_hold_deadband)
        self.altitude_hold_enabled = bool(altitude_hold_enabled)
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
            )

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

        # CH1: roll (lateral bank)
        roll = max(-1.0, min(1.0, self.roll))
        channels[0] = int(max(1000, min(2000, 1500 + roll * roll_deflection)))

        # CH2: pitch (forward tilt from throttle input)
        channels[1] = int(max(1000, min(2000, 1500 + self.throttle * pitch_deflection)))

        # CH3: motor throttle — bipolar around hover.
        alt = max(-1.0, min(1.0, self.altitude))
        alt_scaled = pow(abs(alt), self.throttle_scale) * (1 if alt >= 0 else -1)

        # Altitude hold damper: when altitude stick is in deadband, bias throttle
        # by -k * vz to counteract vertical drift.
        alt_hold_bias = 0
        if self._pose_tracker is not None and self._pose_tracker.enabled:
            if abs(alt) < self.altitude_hold_deadband:
                vz = self._pose_tracker.get_vz()
                alt_hold_bias = int(-self._pose_tracker.k_pwm * vz)
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

        pitch_ff_bias = 0
        if self.pitch_throttle_feedforward > 0 and abs(alt) < self.altitude_hold_deadband:
            pitch_ff_bias = int(self.pitch_throttle_feedforward * abs(self.throttle))
        pitch_ff_bias = max(0, min(200, pitch_ff_bias))

        throttle_pwm = int(
            max(
                1000,
                min(
                    2000,
                    self.hover_throttle
                    + alt_scaled * self.throttle_range
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
            if not init_pose_subscriber():
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

    def _do_reset(self):
        """Teleport the drone back to spawn position via gz-transport set_pose."""
        try:
            from gz.transport13 import Node
            from gz.msgs10.pose_v_pb2 import Pose_V

            node = Node()
            pose_msg = Pose_V()
            p = pose_msg.pose.add()
            p.name = self.gz_model_name
            p.position.x = 0.0
            p.position.y = 0.0
            p.position.z = 0.5
            p.orientation.w = 1.0
            result, _reply = node.request(
                f"/world/{self.gz_world}/set_pose",
                pose_msg,
                1000,
            )
            if result:
                logger.info("Drone pose reset to origin (world=%s model=%s)",
                            self.gz_world, self.gz_model_name)
            else:
                logger.warning("Reset pose request failed for %s/%s",
                               self.gz_world, self.gz_model_name)
        except Exception as e:
            logger.error("Reset pose error: %s", e)

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
        config_update=None,
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
        :param config_update: optional runtime flight config update from web UI.
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
        self.user_arm = user_arm if isinstance(user_arm, bool) else None
        if self.user_arm is True and self._prev_user_arm is not True:
            self._explicit_arm_started_at = time.time()
            logger.info("Explicit arm requested; holding throttle low for 1.0s")
        if self.user_arm is not True:
            self._explicit_arm_started_at = None
        self._prev_user_arm = self.user_arm

        # One-shot reset from Xbox LT (handled in background loop).
        if user_reset is True:
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
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                # The camera worker/resource_tracker can win this cleanup race
                # on process exit. The segment is already gone, which is fine.
                pass
            self._shm = None
