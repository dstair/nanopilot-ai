"""Gazebo telemetry subscribers used by DroneGymEnv."""

import logging
import math
import os
import time

logger = logging.getLogger(__name__)

_pose_node = None
_sim_telemetry = None


def quat_to_euler_deg(w, x, y, z):
    """Convert quaternion to roll/pitch/yaw degrees."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


class SimTelemetry:
    """Keeps latest Gazebo pose-derived telemetry and raw IMU samples."""

    def __init__(self, world, model_name, imu_topic=None):
        from gz.transport13 import Node
        from gz.msgs10.imu_pb2 import IMU
        from gz.msgs10.pose_v_pb2 import Pose_V

        self.node = Node()
        self.model_name = model_name.lower()
        self.pose_topic = f"/world/{world}/dynamic_pose/info"
        self.imu_topic = imu_topic or (
            f"/world/{world}/model/{model_name}/link/base_link/sensor/imu_sensor/imu"
        )
        self.position = (0.0, 0.0, 0.0)
        self.attitude = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.imu = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._prev_position = None
        self._prev_time = None

        if not self.node.subscribe(Pose_V, self.pose_topic, self._on_pose):
            raise RuntimeError(f"Could not subscribe to {self.pose_topic}")
        if not self.node.subscribe(IMU, self.imu_topic, self._on_imu):
            raise RuntimeError(f"Could not subscribe to {self.imu_topic}")

    def _on_pose(self, msg):
        now = time.time()
        for p in msg.pose:
            if self.model_name in p.name.lower():
                pos = (p.position.x, p.position.y, p.position.z)
                q = p.orientation
                self.position = pos
                self.attitude = quat_to_euler_deg(q.w, q.x, q.y, q.z)
                if self._prev_position is not None and self._prev_time is not None:
                    dt = now - self._prev_time
                    if dt > 0:
                        self.velocity = tuple(
                            (pos[i] - self._prev_position[i]) / dt for i in range(3)
                        )
                self._prev_position = pos
                self._prev_time = now
                return

    def _on_imu(self, msg):
        acl = msg.linear_acceleration
        gyr = msg.angular_velocity
        self.imu = (acl.x, acl.y, acl.z, gyr.x, gyr.y, gyr.z)


def init_sim_telemetry(world, model_name, imu_topic=None):
    """Subscribe once to Gazebo pose and IMU topics."""
    global _sim_telemetry
    if _sim_telemetry is not None:
        return _sim_telemetry
    try:
        _sim_telemetry = SimTelemetry(world, model_name, imu_topic=imu_topic)
    except Exception as e:
        logger.warning("Gazebo telemetry subscription failed: %s", e)
        return None
    logger.info(
        "Subscribed to Gazebo telemetry: pose=%s imu=%s",
        _sim_telemetry.pose_topic,
        _sim_telemetry.imu_topic,
    )
    return _sim_telemetry


def init_pose_subscriber():
    """Subscribe to pose topic for vertical velocity tracking. Called once."""
    global _pose_node
    if _pose_node is not None:
        return True
    try:
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
    except Exception as e:
        logger.warning("gz-transport not available: %s", e)
        return False

    node = Node()

    def on_pose(msg):
        for p in msg.pose:
            if "betaloop" in p.name.lower():
                _pose_node.latest = (p.position.x, p.position.y, p.position.z)
                return

    world = os.environ.get("GZ_WORLD", "drone_course_65mm")
    topic = f"/world/{world}/dynamic_pose/info"
    if not node.subscribe(Pose_V, topic, on_pose):
        logger.warning("Could not subscribe to %s", topic)
        return False

    _pose_node = node
    _pose_node.latest = (0.0, 0.0, 0.0)
    return True


class PoseTracker:
    """Tracks drone pose and computes vertical velocity from pose topic."""

    def __init__(self, k_pwm=30.0, deadband=0.05, enabled=True):
        self.k_pwm = k_pwm
        self.deadband = deadband
        self.enabled = enabled
        self.latest = (0.0, 0.0, 0.0)
        self._prev = None
        self._prev_time = None

    def update(self):
        """Call each frame to update position from subscribed topic."""
        if not self.enabled or _pose_node is None:
            return
        self.latest = _pose_node.latest
        now = time.time()
        if self._prev is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                self.vz = (self.latest[2] - self._prev[2]) / dt
        self._prev = self.latest
        self._prev_time = now

    def get_vz(self):
        """Return vertical velocity in m/s."""
        return getattr(self, "vz", 0.0)
