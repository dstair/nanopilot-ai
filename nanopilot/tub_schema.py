"""Shared tub schema constants for Nanopilot data."""

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
