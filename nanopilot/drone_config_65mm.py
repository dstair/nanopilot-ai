# """
# DRONE CONFIG
#
# Configuration overrides for flying a drone in BetaFlight SITL + Gazebo simulator.
# Use with: python drone_manage.py drive --myconfig=drone_config_65mm.py
#   (BetaFPV Air65 profile, ~31g AUW, 65mm wheelbase)
#   https://betafpv.com/products/air65-brushless-whoop-quadcopter
#
# The drone uses the same DonkeyCar pipeline as a car, with a different
# semantic mapping (set by DRONE_ANGLE_MODE below):
#   steering [-1, 1]  ->  yaw (rate in Acro, rate→heading in Angle)
#   throttle [-1, 1]  ->  forward pitch (rate in Acro, bank angle in Angle)
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

# ---- Flight Control Mapping ----
# True  = Angle mode (CH6 high) — BetaFlight self-levels; stick → bank angle.
# False = Acro mode  (CH6 low)  — raw rate command; no self-leveling. Often
#         smoother under yaw because the angle controller's roll/pitch loops
#         aren't fighting transient tilt from yaw torque. Drone won't auto-
#         level if disturbed — must be tested in sim before real flight.
DRONE_ANGLE_MODE = True

DRONE_MAX_PITCH_ANGLE = 45.0  # max pitch degrees (throttle input maps to pitch)
DRONE_MAX_ROLL_ANGLE = 45.0  # max roll degrees (roll input maps to lateral bank)
DRONE_HOVER_THROTTLE = 1490  # Recalibrate this using --mode=inflight-hover
DRONE_THROTTLE_RANGE = 100  # altitude=±1 maps to hover ± range (clamped to [1000, 2000])
# Apply quadratic scaling to altitude input: at high TWR, more throttle gives
# disproportionate thrust. scale**2 = 1 gives linear; lower values gentler slope.
DRONE_THROTTLE_SCALE = 0.1
DRONE_THROTTLE_STEP_SIZE = (
    0.2  # keyboard step per keypress (deflection, snaps back to 0 on release)
)

# Max yaw rate scaling (steering input maps to yaw)
DRONE_MAX_YAW_RATE = 180.0

# Input sensitivity multiplier [0.0–1.0]: scales stick deflection sent to
# BetaFlight. 1.0 = full deflection (±500 PWM from center on pitch/yaw);
# 0.3 = gentler, easier-to-fly commands.
DRONE_INPUT_SENSITIVITY = 1.0

# CH4 yaw deflection cap in PWM microseconds from center (1500). Yaw input at
# hover PWM produces net upward thrust via motor-mixer ω² asymmetry — larger
# deflections make the drone climb on every turn. Keep this small (20–40).
DRONE_YAW_PWM_CAP = 150

# Yaw→throttle feed-forward (PWM step). When |steering| > ~0.01, CH3 is biased
# down by exactly this many PWM (capped at -200) to cancel the upward thrust
# from the motor mixer's ω² asymmetry in Angle mode. Empirically the excess
# thrust is near-constant across yaw magnitude (test_thrust damper-sim
# 2026-04-28: both yaw=1499 and yaw=1530 needed ~50–60 PWM to hold), hence a
# flat step. Damper soaks up the residual. Tune via:
#   ./scripts/test_thrust.sh --mode=damper-sim --airborne-hover=1494 \
#       --damper-yaw-after=3 --damper-yaw-pwm=<...> --damper-yaw-ff=<...>
#
# In Acro mode (DRONE_ANGLE_MODE = False) the angle controller is out of the
# loop and the upward bleed is small/absent — leaving FF on at 60 just drops
# CH3 with nothing to cancel, and the drone descends on every yaw input.
# Keep this 0 while in Acro; retune separately if you go back to Angle.
DRONE_YAW_THROTTLE_FEEDFORWARD = -50.0

# ---- Altitude Hold (Vertical Velocity Damper) ----
# Proportional gain (PWM per m/s): -k_pwm * vz added to throttle when
# altitude stick is in deadband. Start with k=30 (1 m/s climb gets -30 PWM).
DRONE_ALTITUDE_HOLD_K = 30.0

# Deadband around altitude=0 where damper is active (in normalized [-1,1] units).
# Stick outside this range bypasses damper so climb/descend commands dominate.
DRONE_ALTITUDE_HOLD_DEADBAND = 0.05

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
DRONE_GZ_CAMERA_TOPIC = "/world/drone_course_65mm/model/betaloop_drone_cam_65mm/link/camera_link/sensor/camera/image"

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
