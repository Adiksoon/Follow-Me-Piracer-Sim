"""Scaffold for the future Acquire behavior state."""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class AcquireNode(Node):
    def __init__(self):
        super().__init__("piracer_acquire")
        # TODO: Add the Acquire behavior here; this node is only a service scaffold.
        self.declare_parameter("enable_service", "/acquire/enable")
        self._enabled = False
        self._enable_service = self.create_service(
            SetBool,
            self.get_parameter("enable_service").value,
            self._set_enabled,
        )

    def _set_enabled(self, request, response):
        if request.data:
            self._enabled = False
            response.success = False
            response.message = "Acquire behavior is not implemented"
            return response
        self._enabled = False
        response.success = True
        response.message = "Acquire disabled"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AcquireNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
