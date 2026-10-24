"""Publish a Nav2-only map with the fixed initial free-space assumption."""

from math import isfinite

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from piracer_navigation.initial_free_zone import apply_initial_free_zone


class StartZoneMap(Node):
    def __init__(self):
        super().__init__("piracer_start_zone_map")
        self.declare_parameter("enabled", True)
        self.declare_parameter("radius", 1.5)
        self.declare_parameter("input_topic", "/map")
        self.declare_parameter("output_topic", "/nav2/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("max_tf_age", 0.20)

        self._enabled = bool(self.get_parameter("enabled").value)
        self._radius = float(self.get_parameter("radius").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._max_tf_age = float(self.get_parameter("max_tf_age").value)
        self._latest_map = None
        self._center = None
        self._last_wait_reason = None

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            OccupancyGrid, str(self.get_parameter("output_topic").value), map_qos
        )
        self._subscription = self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("input_topic").value),
            self._on_map,
            map_qos,
        )
        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(0.1, self._retry_initialization)

        if self._radius < 0.0 or not isfinite(self._radius):
            raise ValueError("radius must be finite and non-negative")
        if self._max_tf_age < 0.0 or not isfinite(self._max_tf_age):
            raise ValueError("max_tf_age must be finite and non-negative")

    def _on_map(self, message):
        self._latest_map = message
        if not self._enabled:
            self._publisher.publish(message)
            return
        self._publish_if_ready()

    def _retry_initialization(self):
        if self._enabled and self._center is None and self._latest_map is not None:
            self._publish_if_ready()

    def _publish_if_ready(self):
        message = self._latest_map
        if message.header.frame_id != self._map_frame:
            self._wait_once("latest map is not in the configured map frame")
            return

        if self._center is None and not self._capture_center():
            return

        try:
            output = apply_initial_free_zone(
                message, self._center[0], self._center[1], self._radius
            )
        except ValueError as error:
            self._wait_once(f"latest map is invalid: {error}")
            return
        self._publisher.publish(output)

    def _capture_center(self):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns <= 0:
            self._wait_once("waiting for initialized simulation clock")
            return False
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except TransformException:
            self._wait_once("waiting for map to base_link transform")
            return False

        stamp_ns = (
            transform.header.stamp.sec * 1_000_000_000
            + transform.header.stamp.nanosec
        )
        age_ns = now_ns - stamp_ns
        if age_ns < 0 or age_ns > Duration(seconds=self._max_tf_age).nanoseconds:
            self._wait_once("waiting for a current map to base_link transform")
            return False

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        values = (translation.x, translation.y, translation.z,
                  rotation.x, rotation.y, rotation.z, rotation.w)
        if not all(isfinite(value) for value in values):
            self._wait_once("map to base_link transform contains non-finite values")
            return False
        if sum(value * value for value in values[3:]) == 0.0:
            self._wait_once("map to base_link transform has a zero quaternion")
            return False

        self._center = (translation.x, translation.y)
        self._last_wait_reason = None
        self.get_logger().info(
            f"Captured fixed initial free-zone center in {self._map_frame}: "
            f"x={self._center[0]:.3f}, y={self._center[1]:.3f}, "
            f"radius={self._radius:.3f} m"
        )
        return True

    def _wait_once(self, reason):
        if reason != self._last_wait_reason:
            self.get_logger().info(reason)
            self._last_wait_reason = reason


def main(args=None):
    rclpy.init(args=args)
    node = StartZoneMap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
