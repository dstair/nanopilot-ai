"""Gazebo telemetry subscribers used by DroneGymEnv."""

import logging
import math
import os
import time

logger = logging.getLogger(__name__)

_pose_node = None
_sim_telemetry = None

# Velocities are finite differences of pose samples, so they are noisy and a
# teleport produces a meaningless spike. Readings saturate here instead of
# being discarded — see SimTelemetry._on_pose.
MAX_VELOCITY = 20.0


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
                        # Saturate rather than zero: a large reading must still
                        # produce a large correction. Zeroing it would blind
                        # the altitude damper exactly when the drone is moving
                        # fastest, which lets a climb run away unchecked.
                        # Teleport spikes are handled by reset_history().
                        self.velocity = tuple(
                            max(-MAX_VELOCITY, min(MAX_VELOCITY,
                                                   (pos[i] - self._prev_position[i]) / dt))
                            for i in range(3)
                        )
                self._prev_position = pos
                self._prev_time = now
                return

    def reset_history(self):
        """Forget the previous sample so a teleport yields no phantom velocity."""
        self._prev_position = None
        self._prev_time = None
        self.velocity = (0.0, 0.0, 0.0)

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


def init_pose_subscriber(world=None):
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
                # Yaw is needed to rotate world velocity into the drone-heading
                # frame for the velocity cascade. Computed inline rather than
                # via quat_to_euler_deg: this runs on every pose message.
                q = p.orientation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                _pose_node.latest_yaw = math.atan2(siny_cosp, cosy_cosp)
                return

    world = world or os.environ.get("GZ_WORLD", "drone_course_65mm")
    topic = f"/world/{world}/dynamic_pose/info"
    if not node.subscribe(Pose_V, topic, on_pose):
        logger.warning("Could not subscribe to %s", topic)
        return False

    _pose_node = node
    _pose_node.latest = (0.0, 0.0, 0.0)
    _pose_node.latest_yaw = 0.0
    return True


class PoseTracker:
    """Tracks drone pose and computes vertical velocity from pose topic."""

    def __init__(self, k_pwm=30.0, deadband=0.05, enabled=True, vel_tau=0.1):
        self.k_pwm = k_pwm
        self.deadband = deadband
        self.enabled = enabled
        self.latest = (0.0, 0.0, 0.0)
        self.latest_yaw = 0.0
        self._prev = None
        self._prev_time = None
        # Filtered full velocity vector, used by the velocity cascade. Kept
        # entirely separate from the vz above so the verified altitude hold
        # behaviour is bit-identical to before this existed.
        self.vel_tau = float(max(1e-3, vel_tau))
        self._vel = [0.0, 0.0, 0.0]
        self._f_prev = None
        self._f_prev_time = None

    def update(self):
        """Call each frame to update position from subscribed topic."""
        if not self.enabled or _pose_node is None:
            return
        self.latest = _pose_node.latest
        self.latest_yaw = getattr(_pose_node, "latest_yaw", 0.0)
        now = time.time()
        self._update_velocity(now)
        if self._prev is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                vz = (self.latest[2] - self._prev[2]) / dt
                # Saturate, never zero — see SimTelemetry._on_pose.
                self.vz = max(-MAX_VELOCITY, min(MAX_VELOCITY, vz))
        self._prev = self.latest
        self._prev_time = now

    def _update_velocity(self, now):
        """Low-passed world-frame velocity for the velocity cascade.

        Only advances when the pose sample actually CHANGES. update() is called
        from the vehicle loop, which polls faster than Gazebo publishes poses,
        so differencing on every call would read a stale-but-identical position
        as genuine zero motion and drag the estimate toward zero.
        """
        pos = self.latest
        if self._f_prev is not None and self._f_prev_time is not None:
            if pos == self._f_prev:
                return
            dt = now - self._f_prev_time
            if dt > 0:
                alpha = dt / (self.vel_tau + dt)
                for i in range(3):
                    raw = (pos[i] - self._f_prev[i]) / dt
                    # Saturate, never zero — same reasoning as SimTelemetry.
                    raw = max(-MAX_VELOCITY, min(MAX_VELOCITY, raw))
                    self._vel[i] += alpha * (raw - self._vel[i])
        self._f_prev = pos
        self._f_prev_time = now

    def reset_history(self):
        """Forget the previous sample so a teleport yields no phantom vz."""
        self._prev = None
        self._prev_time = None
        self.vz = 0.0
        self._f_prev = None
        self._f_prev_time = None
        self._vel = [0.0, 0.0, 0.0]

    def get_vz(self):
        """Return vertical velocity in m/s."""
        return getattr(self, "vz", 0.0)

    def get_velocity(self):
        """Filtered world-frame velocity (vx, vy, vz) in m/s."""
        return tuple(self._vel)

    def get_yaw(self):
        """Latest yaw in radians."""
        return self.latest_yaw
