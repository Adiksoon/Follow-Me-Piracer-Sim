"""Publish a derived map with one fixed, simulation-only startup assumption."""

from copy import deepcopy
from array import array
import math

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from .start_area import merge_initial_start_area


def _map_qos():
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class StartAreaNode(Node):
    def __init__(self):
        super().__init__("piracer_search_start_area")
        self.declare_parameter("raw_map_topic", "/map")
        self.declare_parameter("derived_map_topic", "/search_map")
        self.declare_parameter("base_frame", "base_link")
        self._map_frame = None
        self._initial_pose = None
        self._pending_map = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(
            OccupancyGrid, self.get_parameter("derived_map_topic").value, _map_qos()
        )
        self._map_subscription = self.create_subscription(
            OccupancyGrid,
            self.get_parameter("raw_map_topic").value,
            self._map_callback,
            _map_qos(),
        )
        self._tf_timer = self.create_timer(0.1, self._publish_pending_map)

    def _map_callback(self, message):
        frame_id = message.header.frame_id
        if self._initial_pose is not None:
            if frame_id != self._map_frame:
                self.get_logger().warning(
                    f"Search startup area waits for original map frame {self._map_frame}; got {frame_id}"
                )
                return
            self._publish_map(message)
            return
        self._map_frame = frame_id
        self._pending_map = message
        self._publish_pending_map()

    def _publish_pending_map(self):
        if self._pending_map is None or self._initial_pose is not None:
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                self.get_parameter("base_frame").value,
                Time(),
            )
        except TransformException as error:
            self.get_logger().debug(f"Search startup area waits for initial map-to-base TF: {error}")
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        self._initial_pose = (translation.x, translation.y, yaw)
        message = self._pending_map
        self._pending_map = None
        self._tf_timer.cancel()
        self.get_logger().info(
            f"Fixed Search startup area in {self._map_frame} anchored at initial "
            f"{self.get_parameter('base_frame').value} pose {self._initial_pose}"
        )
        self._publish_map(message)

    def _publish_map(self, message):
        info = message.info
        orientation = info.origin.orientation
        origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        grid = {
            "width": info.width,
            "height": info.height,
            "resolution": info.resolution,
            "origin": [info.origin.position.x, info.origin.position.y, origin_yaw],
            "data": tuple(message.data),
        }
        data, _summary = merge_initial_start_area(grid, self._initial_pose)
        derived = deepcopy(message)
        derived.data = array("b", data)
        self._publisher.publish(derived)


def main(args=None):
    rclpy.init(args=args)
    node = StartAreaNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
