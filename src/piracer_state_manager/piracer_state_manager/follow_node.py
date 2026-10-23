"""Follow the latest RGB-D suitcase point with bounded P control."""

import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
import tf2_geometry_msgs  # Register PointStamped transforms with tf2.

from piracer_state_manager.follow_control import FollowPController


class PiracerFollowNode(Node):
    def __init__(self):
        super().__init__("piracer_follow")
        defaults = {
            "enable_service": "/follow/enable",
            "detection_topic": "/yolo/detections",
            "target_point_topic": "/piracer/follow/target_point",
            "command_topic": "/cmd_vel",
            "base_frame": "base_link",
            "target_distance_m": 2.25,
            "min_distance_m": 2.0,
            "max_distance_m": 2.5,
            "distance_kp": 0.10,
            "direction_kp": 1.0,
            # Conservative initial cap. The active Gazebo plugin allows 0.20 m/s.
            "max_speed_mps": 0.05,
            "max_steer_rad": 0.554,
            "max_target_age_s": 0.5,
            "control_rate_hz": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._enabled = False
        self._latest_detections = None
        self._latest_target_point = None
        self._latest_target_received_monotonic = None
        self._controller = self._make_controller()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._enable_service = self.create_service(
            SetBool,
            str(self.get_parameter("enable_service").value),
            self._set_enabled,
        )
        self._detection_subscription = self.create_subscription(
            Detection2DArray,
            str(self.get_parameter("detection_topic").value),
            self._on_detections,
            10,
        )
        self._target_point_subscription = self.create_subscription(
            PointStamped,
            str(self.get_parameter("target_point_topic").value),
            self._on_target_point,
            10,
        )
        self._command_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter("command_topic").value),
            10,
        )

        # A steady timer continues to send zero if simulation time pauses.
        # Message age still uses ROS time and the target's original stamp.
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not math.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and positive")
        max_target_age_s = float(self.get_parameter("max_target_age_s").value)
        if not math.isfinite(max_target_age_s) or max_target_age_s <= 0.0:
            raise ValueError("max_target_age_s must be finite and positive")
        self._control_timer = self.create_timer(
            1.0 / control_rate_hz,
            self._control_cycle,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def _make_controller(self):
        return FollowPController(
            target_distance_m=float(self.get_parameter("target_distance_m").value),
            min_distance_m=float(self.get_parameter("min_distance_m").value),
            max_distance_m=float(self.get_parameter("max_distance_m").value),
            distance_kp=float(self.get_parameter("distance_kp").value),
            direction_kp=float(self.get_parameter("direction_kp").value),
            max_speed_mps=float(self.get_parameter("max_speed_mps").value),
            max_steer_rad=float(self.get_parameter("max_steer_rad").value),
        )

    def _on_detections(self, message):
        # Kept as an input for the agreed Follow interface. Data association
        # and detection-based gating are not part of this controller step.
        self._latest_detections = message

    def _on_target_point(self, message):
        self._latest_target_point = message
        self._latest_target_received_monotonic = time.monotonic()

    def _set_enabled(self, request, response):
        if request.data:
            self._enabled = True
            response.success = True
            response.message = "Follow enabled; waiting for a fresh target point"
            self._publish_zero()
            return response

        self._enabled = False
        self._clear_target()
        self._publish_zero()
        response.success = True
        response.message = "Follow disabled"
        return response

    def _clear_target(self):
        self._latest_target_point = None
        self._latest_target_received_monotonic = None

    def _publish_zero(self):
        self._command_publisher.publish(Twist())

    def _control_cycle(self):
        if not self._enabled or self._latest_target_point is None:
            self._publish_zero()
            return

        max_age_s = float(self.get_parameter("max_target_age_s").value)
        received_at = self._latest_target_received_monotonic
        if received_at is None or time.monotonic() - received_at > max_age_s:
            self._publish_zero()
            return

        now = self.get_clock().now()
        stamp = Time.from_msg(
            self._latest_target_point.header.stamp,
            clock_type=self.get_clock().clock_type,
        )
        if now.nanoseconds == 0:
            self._publish_zero()
            return
        age_s = (now - stamp).nanoseconds / 1_000_000_000.0
        if age_s < 0.0 or age_s > max_age_s:
            self._publish_zero()
            return

        base_frame = str(self.get_parameter("base_frame").value)
        try:
            # Transform at the image/point timestamp, not at the latest TF.
            target = self._tf_buffer.transform(self._latest_target_point, base_frame)
            command = self._controller.update(
                float(target.point.x),
                float(target.point.y),
            )
        except (TransformException, ValueError, TypeError) as exc:
            self.get_logger().debug(f"Follow stopped for invalid target or TF: {exc}")
            self._publish_zero()
            return

        message = Twist()
        message.linear.x = command.linear_x
        message.angular.z = command.angular_z
        self._command_publisher.publish(message)

    def stop_and_destroy(self):
        self._enabled = False
        self._publish_zero()
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiracerFollowNode()
    try:
        rclpy.spin(node)
    finally:
        node.stop_and_destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
