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
# The Gazebo camera model renders 640x480; use that natively for capture/recording.
IMAGE_W = 640
IMAGE_H = 480

# Resolution the CNN actually trains and runs at. Decoupled from the capture
# resolution above: full 640x480 frames make the data loader the bottleneck and
# add no useful detail for flying a sim course. torch_train downscales tub
# frames to this, and TorchPilot downscales live camera frames to match.
TRAIN_IMAGE_W = 160
TRAIN_IMAGE_H = 120

# Per-axis training loss weights on [yaw, pitch, roll, altitude].
#
# RL-teacher tubs (rl_collect.py) carry altitude labels that are small but REAL:
# std ~0.020 against 0.23-0.69 on the other three axes. Under uniform weights
# that axis contributes almost nothing to the Huber loss, so the model predicts
# a near-constant value (measured: pred std 0.0069, corr 0.25 on tub_15) and the
# student loses the teacher's active vertical trim, oscillating into the floor.
# 25 is roughly the inverse-variance ratio needed to make the axis matter.
#
# NOTE: this is the opposite of what human-piloted tubs want, where altitude
# genuinely is ~0 and up-weighting it starves the axes with real visual signal.
# See the comment in torch_train.py where these weights are applied.
# With TRAIN_ALTITUDE_PWM_SPACE the altitude target already has std ~0.40,
# comparable to pitch (0.44) and roll (0.69), so it no longer needs a weight to
# be visible to the loss. Keep uniform; revisit only if altitude regresses.
TRAIN_LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Early-stopping patience threshold, RELATIVE to the current best. The stock
# MIN_DELTA is an absolute 0.0005, which is sane at the ~0.01 losses of
# human-piloted tubs but larger than the ENTIRE remaining loss of an
# expert-distilled student (~0.0003) -- there, no improvement can ever clear it
# and training stops on real gains. A relative threshold behaves the same at
# any loss magnitude. Best-weight saving no longer uses either value: it saves
# on any true improvement (see torch_train.py).
MIN_DELTA_REL = 0.01   # need a 1% improvement to reset the patience counter
MIN_DELTA_ABS = 0.0    # optional absolute floor; 0 = purely relative

# Early-stopping patience. The stock 5 is too tight for expert-distilled runs:
# validation loss on these tubs swings by ~10x epoch to epoch (train loss falls
# smoothly the whole time), so whether training stops early is decided by
# whether an EARLY epoch happened to draw a lucky val value. Measured on
# data/tub_22_26-09-07 (2026-09-07): epoch 2 drew 0.001186, nothing beat it for
# 5 epochs, and the run stopped at epoch 7 and deployed the epoch-2 weights
# while train_loss was still falling (0.00144 -> 0.00040). The same trajectory
# on the smaller slalom tub drew UNLUCKY early values instead, so patience kept
# resetting and it ran 30 epochs to 0.000087. Same code, opposite outcome, from
# val noise alone. 15 gives the loss room to settle before patience judges it.
EARLY_STOP_PATIENCE = 15

# Patience is judged on the MEDIAN of the last N validation losses, never on a
# single epoch (see torch_train.py). Verified by replaying the real observed
# loss sequences through the new logic:
#
#   sequence                     window=1 pat=5 (old)     window=3 pat=15 (now)
#   tub_22 S-curve, 46 epochs    stop @36, best 0.000053  stop @42, best 0.000052
#   tub_21 slalom, 30 epochs     stop @30, best 0.000087  stop @30, best 0.000087
#   the lucky-epoch-2 prefix     stop @7,  best 0.001186  stop @42, best 0.000052
#
# The last row is the bug: one lucky early draw ended the run 23x short.
#
# NOTE the median LAGS, so it needs MORE patience, not less. Replaying the same
# sequences at window=3 patience=5 stops at epoch 26 (best 0.000068) on tub_22
# and epoch 19 (best 0.000164) on the slalom -- i.e. smoothing with the stock
# patience is WORSE than no smoothing. The two settings must move together.
EARLY_STOP_SMOOTH_WINDOW = 3

# Train the altitude axis in throttle-PWM space (see TubDataset). The raw
# altitude command reaches the motors through sign(a)*|a|**DRONE_THROTTLE_SCALE,
# which is near-vertical at zero -- alt=-0.0001 is already -15.8 PWM. Measured
# on tub_15: raw labels std 0.020 but the thrust they encode spans std 39.8 PWM,
# so a 0.008 raw error (~1e-5 of Huber loss) is a 9.3 PWM sustained thrust bias.
# That bias is what flew the distilled student into the floor.
# TURNED OFF 2026-09-06 with DRONE_VELOCITY_CONTROL. This flag existed because
# the altitude command reached the motors through sign(a)*|a|**0.2, a curve so
# steep at zero that a 0.008 raw error was 9.3 PWM of sustained thrust -- fatal,
# because the plant integrated it. Under velocity control the altitude axis is a
# LINEAR height trim: the same 0.008 error is 8mm of height offset. The curve is
# no longer in the path, so pre-applying it to the labels would distort the loss
# to fix a problem that no longer exists. Turn it back on ONLY if
# DRONE_VELOCITY_CONTROL goes back off.
TRAIN_ALTITUDE_PWM_SPACE = False

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
DRONE_MAX_PITCH_ANGLE = 20.0  # max pitch degrees (throttle input maps to pitch)
DRONE_MAX_ROLL_ANGLE = 10.0  # max roll degrees (roll input maps to lateral bank)
DRONE_HOVER_THROTTLE = 1480  # Re-verified 2026-08-23 after taming BF pitch/roll
# PIDs. Static hover (thrust = weight at level attitude) sits at ~1480; the old
# flyaway at 1475 was NOT a bias error but the BF pitch PID limit-cycling at
# full motor split, whose omega^2 rectification added phantom collective
# thrust. Active maneuvering still adds a little rectified thrust, so expect
# slight climb during stick activity.
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
# Retuned 8.0 -> 2.0 on 2026-08-23. The velocity-only damper cancels a
# constant thrust bias only at a nonzero climb rate (vz = ff*pitch/K), so an
# over-large feed-forward makes any sustained pitch climb: at ff=8 the drone
# gained ~4m per 60s of half-pitch flight, which put the RL policy above the
# gate band (0.2-3.0m) within ~25s. Swept 0/2/4/6/8 airborne with the damper
# engaged; 2.0 minimizes |drift| and, unlike higher values, is not
# systematically upward (+2.0m/60s at half pitch, -0.9m at full pitch).
DRONE_PITCH_THROTTLE_FEEDFORWARD = 2.0

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
# altitude stick is in deadband. Raised 45 -> 90 on 2026-08-23: with the tamed
# BF pitch/roll PIDs the thrust-vs-PWM slope is ~0.01 m/s^2 per PWM, so K=45
# gave only ~0.45/s of closed-loop damping — too soft to hold residual thrust
# scatter (~±0.5 m/s^2).
DRONE_ALTITUDE_HOLD_K = 90.0

# Position gain for the altitude hold, PWM per metre of height error (0 = the
# old pure-velocity damper). A velocity-only damper gives the plant NO height
# regulation: a steady thrust bias B settles at vz = B/K and sinks forever. A
# distilled BC student's -5 PWM bias (2.5% of full scale) sank it into the floor
# in ~30s every episode. With this term the same bias is a bounded B/Kp offset.
#
# Sizing: thrust slope near hover is ~0.01 m/s^2 per PWM, so the closed loop is
# w^2 = 0.01*Kp and 2*zeta*w = 0.01*K = 0.9. Kp=30 -> w=0.55 rad/s, zeta=0.82
# (well damped), and a 5 PWM bias becomes a 0.17m offset.
DRONE_ALTITUDE_HOLD_KP = 30.0

# Metres the in-deadband altitude command may shift the held height. Inside the
# 0.1 deadband the shaped command spans +-0.63, so this is +-0.63m of authority
# at 1.0. The command trims the target height rather than adding thrust, which
# is what stops a constant command from integrating.
DRONE_ALTITUDE_TRIM_RANGE = 1.0

# ---- Velocity Control Mode ----
# Changes what the learned policy has to output. With this ON, pitch/roll are
# SPEED SETPOINTS in the drone-heading frame and altitude is a LINEAR height
# trim; the env closes the inner loop. With it OFF the inputs are raw angle and
# thrust commands, as they were before 2026-09-06.
#
# Why: every distillation failure traced to the policy needing a precisely
# calibrated thrust. A 5 PWM altitude bias (2.5% of full scale, invisible in
# val_loss) sank the student into the floor because the plant integrated it.
# Under setpoints, the same error is a slightly wrong speed the next frame
# corrects. Models and tubs are NOT comparable across this flag.
DRONE_VELOCITY_CONTROL = True

# Speed at full stick. Gates sit 5m apart with 2m lateral offsets, so 1.5 m/s
# cruise gives ~3.5s between gates and a ~12s course.
DRONE_MAX_FORWARD_SPEED = 2.0  # m/s at throttle = 1
DRONE_MAX_LATERAL_SPEED = 1.0  # m/s at roll = 1

# Cascade gains, stick units per m/s, PER AXIS. Roll is capped at 10 degrees and
# pitch at 20, so the same roll stick buys half the acceleration and the axes
# cannot share gains. Measured open-loop with scripts/probe_velocity.py and a
# stick sweep on 2026-09-06:
#
#   pitch stick 0.15/0.30/0.50/0.80 -> 0.26/0.73/1.53/2.99 m/s forward
#   roll  stick 0.30/0.60/1.00      -> 0.26/0.72/1.52 m/s right
#
# KFF is the inverse slope at cruise (stick needed to HOLD the speed against
# drag); KP trims the residual and sets the bandwidth, ~2 rad/s on both axes.
# No integral term: the outer loop (expert, then the vision student) closes on
# gate geometry, so a small steady-state speed error is harmless.
DRONE_VELOCITY_KP_FWD = 0.6
DRONE_VELOCITY_KFF_FWD = 0.35
DRONE_VELOCITY_KP_LAT = 1.2
DRONE_VELOCITY_KFF_LAT = 0.8
# Low-pass on the pose-differenced velocity estimate, seconds.
DRONE_VELOCITY_TAU = 0.1

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
