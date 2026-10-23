"""Scaffold for the future Follow behavior state."""

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray


class PiracerFollowNode(Node):
    def __init__(self):
        super().__init__("piracer_follow")
        self.declare_parameter("enable_service", "/follow/enable")
        self.declare_parameter("detection_topic", "/yolo/detections")
        self.declare_parameter("target_point_topic", "/piracer/follow/target_point")
        self._enabled = False
        self._latest_detections = None
        self._latest_target_point = None
        self._enable_service = self.create_service(
            SetBool,
            self.get_parameter("enable_service").value,
            self._set_enabled,
        )
        self._detection_subscription = self.create_subscription(
            Detection2DArray,
            self.get_parameter("detection_topic").value,
            self._on_detections,
            10,
        )
        self._target_point_subscription = self.create_subscription(
            PointStamped,
            self.get_parameter("target_point_topic").value,
            self._on_target_point,
            10,
        )

    def _on_detections(self, message):
        self._latest_detections = message

    def _on_target_point(self, message):
        self._latest_target_point = message

    def _set_enabled(self, request, response):
        if request.data:
            self._enabled = False
            response.success = False
            response.message = "Follow behavior is not implemented"
            return response
        self._enabled = False
        response.success = True
        response.message = "Follow disabled"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PiracerFollowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
