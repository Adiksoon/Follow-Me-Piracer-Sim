"""Convert timestamped suitcase points into bounded Nav2 pose goals."""

from __future__ import annotations

import math
import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  # Register PointStamped transforms with tf2.

from piracer_navigation.goal_control import TargetGoalController


_GAIN_PARAMETERS = (
    "range_kp",
    "lateral_kp",
)
MIN_FOLLOW_RANGE_M = 2.0
STOP_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _yaw_from_quaternion(rotation) -> float:
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


class FollowGoalController(Node):
    """Publish map-frame Nav2 goal updates from one visible target point."""

    def __init__(self) -> None:
        super().__init__("follow_goal_controller")
        defaults = {
            "target_point_topic": "/piracer/follow/target_point",
            "goal_update_topic": "goal_update",
            "navigate_to_pose_action": "navigate_to_pose",
            "map_frame": "map",
            "base_frame": "base_link",
            "target_range_m": 2.25,
            "stop_topic": "/piracer_navigation/stop",
            "range_kp": 1.0,
            "lateral_kp": 1.0,
            "max_goal_step_m": 0.25,
            "max_target_age_s": 0.5,
            "transform_timeout_s": 0.05,
            "enable_navigation": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._controller = self._make_controller(self._parameter_values())
        self._last_input_stamp_ns = None
        self._active_goal = False
        self._goal_handle = None
        self._goal_pending = False
        self._target_too_close = False
        self._warned_no_action_server = False
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._goal_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("goal_update_topic").value),
            10,
        )
        self._stop_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("stop_topic").value),
            STOP_QOS,
        )
        self._error_publisher = self.create_publisher(
            Vector3Stamped, "/piracer/follow/error", 10
        )
        self._correction_publisher = self.create_publisher(
            Vector3Stamped, "/piracer/follow/goal_correction", 10
        )
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter("navigate_to_pose_action").value),
        )
        self._follow_tree_path = os.path.join(
            get_package_share_directory("piracer_navigation"),
            "behavior_trees",
            "follow_visible_target.xml",
        )
        self._point_subscription = self.create_subscription(
            PointStamped,
            str(self.get_parameter("target_point_topic").value),
            self._on_target_point,
            10,
        )

    def _parameter_values(self) -> dict:
        return {
            name: self.get_parameter(name).value
            for name in (
                "target_range_m",
                *_GAIN_PARAMETERS,
                "max_goal_step_m",
            )
        }

    @staticmethod
    def _make_controller(values: dict) -> TargetGoalController:
        return TargetGoalController(
            reference_range_m=float(values["target_range_m"]),
            range_kp=float(values["range_kp"]),
            lateral_kp=float(values["lateral_kp"]),
            max_goal_step_m=float(values["max_goal_step_m"]),
        )

    def _on_parameters_changed(self, parameters) -> SetParametersResult:
        values = self._parameter_values()
        for parameter in parameters:
            if parameter.name in values:
                values[parameter.name] = parameter.value
        try:
            controller = self._make_controller(values)
            for name in ("max_target_age_s", "transform_timeout_s"):
                value = float(
                    next(
                        (p.value for p in parameters if p.name == name),
                        self.get_parameter(name).value,
                    )
                )
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        except (KeyError, TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        self._controller = controller
        return SetParametersResult(successful=True)

    def _on_target_point(self, point) -> None:
        stamp = Time.from_msg(point.header.stamp, clock_type=self.get_clock().clock_type)
        stamp_ns = stamp.nanoseconds
        now = self.get_clock().now()
        if now.nanoseconds == 0 or stamp_ns <= 0:
            return
        age_s = (now - stamp).nanoseconds / 1_000_000_000.0
        max_age = float(self.get_parameter("max_target_age_s").value)
        if age_s < 0.0 or age_s > max_age:
            return
        if self._last_input_stamp_ns is not None and stamp_ns <= self._last_input_stamp_ns:
            return

        timeout = Duration(
            seconds=float(self.get_parameter("transform_timeout_s").value)
        )
        map_frame = str(self.get_parameter("map_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)
        try:
            target_base = self._tf_buffer.transform(point, base_frame, timeout=timeout)
            target_map = self._tf_buffer.transform(point, map_frame, timeout=timeout)
            base_in_map = self._tf_buffer.lookup_transform(
                map_frame, base_frame, stamp, timeout=timeout
            )
        except TransformException as exc:
            self.get_logger().debug(f"Skipping target without exact-stamp TF: {exc}")
            return

        target_range_m = math.sqrt(
            float(point.point.x) ** 2
            + float(point.point.y) ** 2
            + float(point.point.z) ** 2
        )
        if target_range_m < MIN_FOLLOW_RANGE_M:
            self._request_close_target_stop()
            self._last_input_stamp_ns = stamp_ns
            return
        if self._target_too_close:
            self._target_too_close = False
            self._stop_publisher.publish(Bool(data=False))

        try:
            correction = self._controller.update(
                target_range_m,
                float(target_base.point.y),
            )
        except ValueError as exc:
            self.get_logger().debug(f"Skipping invalid target sample: {exc}")
            return

        transform = base_in_map.transform
        base_yaw = _yaw_from_quaternion(transform.rotation)
        cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)
        goal_x = (
            transform.translation.x
            + cos_yaw * correction.forward_m
            - sin_yaw * correction.lateral_m
        )
        goal_y = (
            transform.translation.y
            + sin_yaw * correction.forward_m
            + cos_yaw * correction.lateral_m
        )
        target_dx = target_map.point.x - goal_x
        target_dy = target_map.point.y - goal_y
        goal_yaw = (
            math.atan2(target_dy, target_dx)
            if math.hypot(target_dx, target_dy) > 1e-9
            else base_yaw
        )

        goal = PoseStamped()
        goal.header.frame_id = map_frame
        goal.header.stamp = point.header.stamp
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal.pose.orientation.w = math.cos(goal_yaw / 2.0)

        self._goal_publisher.publish(goal)
        error = Vector3Stamped()
        error.header = goal.header
        error.vector.x = correction.range_error_m
        error.vector.y = correction.lateral_error_m
        error.vector.z = correction.target_range_m
        self._error_publisher.publish(error)
        output = Vector3Stamped()
        output.header = goal.header
        output.vector.x = correction.forward_m
        output.vector.y = correction.lateral_m
        self._correction_publisher.publish(output)
        self._last_input_stamp_ns = stamp_ns
        self._send_initial_goal_if_enabled(goal)

    def _request_close_target_stop(self) -> None:
        if self._target_too_close:
            return
        self._target_too_close = True
        self._stop_publisher.publish(Bool(data=True))
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def _send_initial_goal_if_enabled(self, goal: PoseStamped) -> None:
        if not bool(self.get_parameter("enable_navigation").value):
            return
        if self._active_goal or self._goal_pending:
            return
        if not self._action_client.server_is_ready():
            if not self._warned_no_action_server:
                self.get_logger().warning("NavigateToPose action server is not ready")
                self._warned_no_action_server = True
            return
        action_goal = NavigateToPose.Goal()
        action_goal.pose = goal
        action_goal.behavior_tree = self._follow_tree_path
        self._goal_pending = True
        self._action_client.send_goal_async(action_goal).add_done_callback(
            self._on_goal_response
        )

    def _on_goal_response(self, future) -> None:
        self._goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:  # Report the asynchronous action transport failure.
            self.get_logger().error(f"NavigateToPose request failed: {exc}")
            return
        if not handle.accepted:
            self.get_logger().warning("NavigateToPose goal was rejected")
            return
        self._active_goal = True
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_goal_result)
        if self._target_too_close:
            handle.cancel_goal_async()

    def _on_goal_result(self, future) -> None:
        self._active_goal = False
        self._goal_handle = None
        try:
            result = future.result()
        except Exception as exc:  # Report the asynchronous action transport failure.
            self.get_logger().error(f"NavigateToPose result failed: {exc}")
            return
        self.get_logger().info(f"NavigateToPose finished with status {result.status}")


def main(args=None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = FollowGoalController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
