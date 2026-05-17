# """
# DRONE CONFIG
#
# Configuration overrides for flying a drone in BetaFlight SITL + Gazebo simulator.
# Use with: python drone_manage.py drive --myconfig=drone_config_80mm.py
#   (BETAFPV Pavo Pico II O4 profile, 79g with selected battery, 80mm wheelbase)
#   https://betafpv.com/products/pavo-pico-ii-brushless-whoop-quadcopter
# The drone uses the same DonkeyCar pipeline as a car, with a different
# semantic mapping (BetaFlight Angle mode):
#   steering [-1, 1]  ->  yaw rate
#   throttle [-1, 1]  ->  forward pitch (tilt angle)
#   altitude [-1, 1]  ->  motor throttle (bipolar, 0 = hover PWM,
#                         ±1 = hover ± DRONE_THROTTLE_RANGE)
# """

# ---- Camera ----
# DroneGymEnv provides images directly, so use MOCK camera in DonkeyCar
CAMERA_TYPE = "MOCK"

# Enable FPV web server (shows camera in browser at /drive)
USE_FPV = True

# Image resolution for the CNN pipeline (overrides DonkeyCar's 160x120 default).
# The Gazebo camera model renders 640x480; use that natively in the pipeline.
IMAGE_W = 640
IMAGE_H = 480

# ---- Drive Train ----
# No physical actuators; DroneGymEnv sends RC commands via UDP
DONKEY_GYM = False
DRIVE_TRAIN_TYPE = "MOCK"

# ---- Controller ----
# Disable physical joystick — web UI only
USE_JOYSTICK_AS_DEFAULT = False

# ---- Drone Simulator ----
USE_DRONE_SIM = True

# ---- BetaFlight SITL ----
BETAFLIGHT_RC_HOST = "127.0.0.1"
BETAFLIGHT_RC_PORT = 9004
BETAFLIGHT_ARM_CHANNEL = 4  # AUX1 (0-indexed)
BETAFLIGHT_MODE_CHANNEL = 5  # AUX2

# ---- Flight Control Mapping (Angle mode) ----
DRONE_MAX_PITCH_ANGLE = 45.0  # max pitch degrees (throttle input maps to pitch)
DRONE_MAX_ROLL_ANGLE = 45.0  # max roll degrees (roll input maps to lateral bank)
DRONE_HOVER_THROTTLE = 1475  # Calibrated at 79g AUW with test_thrust damper-sim.
DRONE_THROTTLE_RANGE = (
    100  # altitude=±1 maps to hover ± range (clamped to [1000, 2000])
)
DRONE_THROTTLE_SCALE = 0.2
DRONE_THROTTLE_STEP_SIZE = (
    0.1  # keyboard step per keypress (deflection, snaps back to 0 on release)
)

# Max yaw rate target for full yaw input. 120 deg/s is a 360-degree turn in
# about 3 seconds. DroneGymEnv derives the CH4 PWM deflection from this unless
# DRONE_YAW_PWM_CAP is set below.
DRONE_MAX_YAW_RATE = 120.0


DRONE_ANGLE_MODE=True # this appears to always set to Acro mode regardless of True/FLse
DRONE_YAW_THROTTLE_FEEDFORWARD = 0.0

# Input sensitivity multiplier [0.0–1.0]: scales stick deflection sent to
# BetaFlight. 1.0 = full deflection (±500 PWM from center on pitch/yaw);
# 0.3 = gentler, easier-to-fly commands.
DRONE_INPUT_SENSITIVITY = 0.4

# Optional CH4 yaw deflection cap in PWM microseconds from center (1500).
# Leave as None so DRONE_MAX_YAW_RATE controls turn speed. Set an integer here
# only when you want a hard safety cap regardless of the configured yaw rate.
DRONE_YAW_PWM_CAP = None

# ---- Altitude Hold (Vertical Velocity Damper) ----
# Proportional gain (PWM per m/s): -k_pwm * vz added to throttle when
# altitude stick is in deadband. 45 passes the 80mm damper-sim check.
DRONE_ALTITUDE_HOLD_K = 45.0

# Deadband around altitude=0 where damper is active (in normalized [-1,1] units).
# Stick outside this range bypasses damper so climb/descend commands dominate.
DRONE_ALTITUDE_HOLD_DEADBAND = 0.1

# Enable vertical velocity damper. Set False to disable and use raw throttle.
DRONE_ALTITUDE_HOLD_ENABLED = True

# ---- Camera Source ----
# "gz_transport" - native macOS: Gazebo Harmonic via gz-transport (GPU-accelerated)
#                  Discover topic: gz topic -l | grep camera
# "rtsp"         - Docker mode: RTSP stream from Gazebo Classic in container
DRONE_CAMERA_SOURCE = "gz_transport"

# gz-transport camera topic (native macOS mode).
# Must match the world name in your launch script. Run `gz topic -l | grep camera`
# to confirm the topic on your setup.
DRONE_GZ_CAMERA_TOPIC = "/world/baylands_80mm/model/betaloop_drone_cam_80mm/link/camera_link/sensor/camera/image"

# RTSP camera stream URL (Docker mode only -- used when DRONE_CAMERA_SOURCE = "rtsp")
# DRONE_RTSP_URL = "rtsp://127.0.0.1:8554/live"

# ---- Simulated Camera Delay ----
SIMULATED_DELAY_MS = 0  # 0=off; e.g. 150 for 150ms lag

# ---- Loop Timing ----
MEASURE_LOOP_DELAY = True
LOOP_DELAY_LOG_INTERVAL = 100  # log stats every N iterations

# ---- Telemetry Recording ----
# Record additional drone telemetry in tubs alongside images
DRONE_RECORD_POSITION = True
DRONE_RECORD_ATTITUDE = True
DRONE_RECORD_VELOCITY = True
DRONE_RECORD_IMU = True

# ---- Vehicle Loop ----
DRIVE_LOOP_HZ = 30

# ---- Data Storage ----
# Always create a new tub (drone schema differs from car schema)
AUTO_CREATE_NEW_TUB = True
