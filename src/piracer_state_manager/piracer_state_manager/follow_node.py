"""Scaffold for the future Follow behavior state."""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class PiracerFollowNode(Node):
    def __init__(self):
        super().__init__("piracer_follow")
        # TODO: Add Follow behavior here; this node currently only exposes its enable service.
        self.declare_parameter("enable_service", "/follow/enable")
        self._enabled = False
        self._enable_service = self.create_service(
            SetBool,
            self.get_parameter("enable_service").value,
            self._set_enabled,
        )

    def _set_enabled(self, request, response):
        self._enabled = bool(request.data)
        response.success = True
        response.message = (
            "Follow enabled (behavior is a scaffold)"
            if self._enabled
            else "Follow disabled"
        )
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
