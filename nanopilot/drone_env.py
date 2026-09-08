"""Shared DroneGymEnv construction."""

import os

from drone_gym import DroneGymEnv, _GZ_CAMERA_TOPIC_DEFAULT


def build_drone_env(
    cfg,
    *,
    airframe=None,
    record_position=None,
    record_attitude=None,
    record_velocity=None,
    record_imu=None,
    gz_world=None,
    gz_model_name=None,
):
    """Build DroneGymEnv from Nanopilot config with optional overrides."""
    if airframe is not None:
        gz_world = gz_world or os.environ.get("GZ_WORLD", f"drone_course_{airframe}")
        gz_model_name = gz_model_name or f"betaloop_drone_cam_{airframe}"
    else:
        gz_world = (
            gz_world if gz_world is not None
            else os.environ.get("GZ_WORLD")
            or getattr(cfg, "GZ_WORLD", None)
        )
        gz_model_name = (
            gz_model_name if gz_model_name is not None
            else getattr(cfg, "DRONE_GZ_MODEL_NAME", None)
        )
        if gz_world and gz_model_name is None:
            gz_model_name = (
                "betaloop_drone_cam_80mm" if gz_world.endswith("80mm")
                else "betaloop_drone_cam_65mm"
            )
    if gz_world and gz_model_name:
        gz_camera_topic = (
            f"/world/{gz_world}/model/{gz_model_name}"
            "/link/camera_link/sensor/camera/image"
        )
    else:
        gz_camera_topic = getattr(cfg, "DRONE_GZ_CAMERA_TOPIC", _GZ_CAMERA_TOPIC_DEFAULT)
    return DroneGymEnv(
        rc_host=getattr(cfg, "BETAFLIGHT_RC_HOST", "127.0.0.1"),
        rc_port=getattr(cfg, "BETAFLIGHT_RC_PORT", 9004),
        camera_source=getattr(cfg, "DRONE_CAMERA_SOURCE", "gz_transport"),
        gz_camera_topic=gz_camera_topic,
        rtsp_url=getattr(cfg, "DRONE_RTSP_URL", "rtsp://127.0.0.1:8554/live"),
        max_pitch_angle=getattr(cfg, "DRONE_MAX_PITCH_ANGLE", 25.0),
        max_roll_angle=getattr(
            cfg,
            "DRONE_MAX_ROLL_ANGLE",
            getattr(cfg, "DRONE_MAX_PITCH_ANGLE", 25.0),
        ),
        max_yaw_rate=getattr(cfg, "DRONE_MAX_YAW_RATE", 90.0),
        hover_throttle=getattr(cfg, "DRONE_HOVER_THROTTLE", 1500),
        throttle_range=getattr(cfg, "DRONE_THROTTLE_RANGE", 300),
        throttle_scale=getattr(cfg, "DRONE_THROTTLE_SCALE", 1.0),
        arm_channel=getattr(cfg, "BETAFLIGHT_ARM_CHANNEL", 4),
        mode_channel=getattr(cfg, "BETAFLIGHT_MODE_CHANNEL", 5),
        image_w=cfg.IMAGE_W,
        image_h=cfg.IMAGE_H,
        simulated_delay_ms=getattr(cfg, "SIMULATED_DELAY_MS", 0),
        measure_loop_delay=getattr(cfg, "MEASURE_LOOP_DELAY", False),
        loop_delay_log_interval=getattr(cfg, "LOOP_DELAY_LOG_INTERVAL", 100),
        input_sensitivity=getattr(cfg, "DRONE_INPUT_SENSITIVITY", 1.0),
        yaw_pwm_cap=getattr(cfg, "DRONE_YAW_PWM_CAP", None),
        yaw_throttle_feedforward=getattr(cfg, "DRONE_YAW_THROTTLE_FEEDFORWARD", 0.0),
        pitch_throttle_feedforward=getattr(cfg, "DRONE_PITCH_THROTTLE_FEEDFORWARD", 0.0),
        altitude_hold_k=getattr(cfg, "DRONE_ALTITUDE_HOLD_K", 30.0),
        altitude_hold_kp=getattr(cfg, "DRONE_ALTITUDE_HOLD_KP", 0.0),
        altitude_trim_range=getattr(cfg, "DRONE_ALTITUDE_TRIM_RANGE", 0.0),
        altitude_hold_deadband=getattr(cfg, "DRONE_ALTITUDE_HOLD_DEADBAND", 0.05),
        altitude_hold_enabled=getattr(cfg, "DRONE_ALTITUDE_HOLD_ENABLED", True),
        velocity_control=getattr(cfg, "DRONE_VELOCITY_CONTROL", False),
        max_forward_speed=getattr(cfg, "DRONE_MAX_FORWARD_SPEED", 2.0),
        max_lateral_speed=getattr(cfg, "DRONE_MAX_LATERAL_SPEED", 1.0),
        velocity_kp_fwd=getattr(cfg, "DRONE_VELOCITY_KP_FWD", 0.6),
        velocity_kff_fwd=getattr(cfg, "DRONE_VELOCITY_KFF_FWD", 0.35),
        velocity_kp_lat=getattr(cfg, "DRONE_VELOCITY_KP_LAT", 1.2),
        velocity_kff_lat=getattr(cfg, "DRONE_VELOCITY_KFF_LAT", 0.8),
        velocity_tau=getattr(cfg, "DRONE_VELOCITY_TAU", 0.1),
        angle_mode=getattr(cfg, "DRONE_ANGLE_MODE", True),
        record_position=(
            getattr(cfg, "DRONE_RECORD_POSITION", False)
            if record_position is None
            else record_position
        ),
        record_attitude=(
            getattr(cfg, "DRONE_RECORD_ATTITUDE", False)
            if record_attitude is None
            else record_attitude
        ),
        record_velocity=(
            getattr(cfg, "DRONE_RECORD_VELOCITY", False)
            if record_velocity is None
            else record_velocity
        ),
        record_imu=(
            getattr(cfg, "DRONE_RECORD_IMU", False) if record_imu is None else record_imu
        ),
        gz_world=gz_world,
        gz_model_name=gz_model_name,
        gz_imu_topic=getattr(cfg, "DRONE_GZ_IMU_TOPIC", None),
    )
