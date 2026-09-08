"""Shared tub schema constants for Nanopilot data."""

import math
import shutil

# Velocity normalization, shared by TubDataset (training) and TorchPilot
# (inference). GateChaseEnv scales the teacher's velocity observation the same
# way, so the student sees the axis in the same units the teacher acted on.
VELOCITY_SCALE = 5.0


def heading_frame_velocity(vel_x, vel_y, vel_z, yaw_deg):
    """World-frame velocity -> drone-heading frame, scaled by VELOCITY_SCALE.

    Mirrors GateChaseEnv._get_obs()[5:8] exactly. The teacher's altitude control
    is velocity feedback, so a student without this input cannot represent the
    policy it is being trained to clone -- that is what caused the vertical
    oscillation that survived loss-weighting, the PWM-space fix and DAgger.

    Heading frame (not world) so the input is yaw-invariant: the same physical
    sideways drift reads the same regardless of which way the drone is facing.
    vel_z needs no rotation.

    Defined here, and imported by BOTH dataset.py and torch_pilot.py, so the
    training and inference transforms cannot silently drift apart.
    """
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (vel_x * cy + vel_y * sy) / VELOCITY_SCALE,
        (-vel_x * sy + vel_y * cy) / VELOCITY_SCALE,
        vel_z / VELOCITY_SCALE,
    )

BASE_TUB_INPUTS = [
    "cam/image_array",
    "user/angle",
    "user/throttle",
    "user/roll",
    "user/altitude",
    "user/mode",
]
BASE_TUB_TYPES = ["image_array", "float", "float", "float", "float", "str"]

POSITION_KEYS = ["pos/pos_x", "pos/pos_y", "pos/pos_z"]
ATTITUDE_KEYS = ["imu/roll", "imu/pitch", "imu/yaw"]
VELOCITY_KEYS = ["vel/vel_x", "vel/vel_y", "vel/vel_z"]
IMU_KEYS = [
    "imu/acl_x",
    "imu/acl_y",
    "imu/acl_z",
    "imu/gyr_x",
    "imu/gyr_y",
    "imu/gyr_z",
]

DRONE_TUB_INPUTS = (
    BASE_TUB_INPUTS
    + POSITION_KEYS
    + ATTITUDE_KEYS
    + VELOCITY_KEYS
    + IMU_KEYS
)
DRONE_TUB_TYPES = ["image_array"] + ["float"] * 4 + ["str"] + ["float"] * 15


def drone_tub_schema(
    record_position=False,
    record_attitude=False,
    record_velocity=False,
    record_imu=False,
):
    """Return tub inputs/types for the requested drone telemetry fields."""
    inputs = list(BASE_TUB_INPUTS)
    types = list(BASE_TUB_TYPES)
    for enabled, keys in (
        (record_position, POSITION_KEYS),
        (record_attitude, ATTITUDE_KEYS),
        (record_velocity, VELOCITY_KEYS),
        (record_imu, IMU_KEYS),
    ):
        if enabled:
            inputs += keys
            types += ["float"] * len(keys)
    return inputs, types


def prune_if_empty(tub):
    """Delete a freshly auto-created tub directory if the session ended
    without writing any records, so runs where nothing was recorded don't
    leave empty tub_NN skeleton folders behind in data/. Call only on a
    tub that was just created for this run (never a shared/appended-to
    DATA_PATH), after the writer has been shut down."""
    if len(tub) > 0:
        return False
    shutil.rmtree(tub.base_path, ignore_errors=True)
    return True
