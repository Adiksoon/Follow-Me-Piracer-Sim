"""Follow the latest RGB-D suitcase point with bounded P control."""

import json
import math
import os
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.clock import Clock, ClockType
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from rtabmap_msgs.msg import OdomInfo
import tf2_geometry_msgs  # Register PointStamped transforms with tf2.

from piracer_state_manager.follow_control import FollowPController
from piracer_state_manager.follow_nav2_helpers import (
    DIRECT,
    NAV2,
    NAV2_PENDING,
    SAFE_STOP,
    ClearanceHysteresis,
    bound_nav_twist,
    footprint_clear,
    goal_refresh_due,
    localization_is_fresh,
    nav2_command_policy,
    require_distinct_command_topics,
    stand_off_goal,
    stamped_input_is_fresh,
    swept_footprint_clear,
)


class PiracerFollowNode(Node):
    def __init__(self):
        super().__init__("piracer_follow")
        defaults = {
            "enable_service": "/follow/enable",
            "target_point_topic": "/piracer/follow/target_point",
            "command_topic": "/cmd_vel",
            "nav2_command_topic": "/cmd_vel",
            "follow_nav2_avoidance": False,
            "local_costmap_topic": "/local_costmap/costmap",
            "global_costmap_topic": "/global_costmap/costmap",
            "odom_info_topic": "/visual_vo/odom_info_lite",
            "max_localization_age_s": 1.0,
            "local_costmap_max_age_s": 0.6,
            "global_costmap_max_age_s": 2.0,
            "control_sample_topic": "/piracer/follow/control_sample",
            "control_diagnostics_topic": "/piracer/follow/control_diagnostics",
            "publish_cmd_vel": False,
            "base_frame": "base_link",
            "target_distance_m": 2.25,
            "min_distance_m": 2.0,
            "distance_kp": 0.10,
            "distance_ki": 0.0,
            "distance_kd": 0.0,
            "distance_filter_tau_s": 0.3,
            "distance_integral_limit_mps": 0.0,
            "direction_kp": 1.0,
            "direction_kd": 0.0,
            "direction_ki": 0.0,
            "direction_integral_limit_rad": 0.02,
            "direction_filter_tau_s": 0.3,
            # Conservative initial cap. The active Gazebo plugin allows 0.20 m/s.
            "max_speed_mps": 0.05,
            "max_steer_rad": 0.554,
            "max_target_age_s": 0.5,
            "control_rate_hz": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._publish_cmd_vel = bool(self.get_parameter("publish_cmd_vel").value)
        self._follow_nav2_avoidance = bool(
            self.get_parameter("follow_nav2_avoidance").value
        )
        if self._follow_nav2_avoidance:
            output_topic = self.resolve_topic_name(
                str(self.get_parameter("command_topic").value)
            )
            nav2_input_topic = self.resolve_topic_name(
                str(self.get_parameter("nav2_command_topic").value)
            )
            require_distinct_command_topics(output_topic, nav2_input_topic)
        self._enabled = False
        self._latest_target_point = None
        self._latest_target_received_monotonic = None
        self._mode = DIRECT if not self._follow_nav2_avoidance else SAFE_STOP
        self._mode_reason = "disabled"
        self._clearance = ClearanceHysteresis()
        self._local_grid = None
        self._local_grid_received_monotonic = None
        self._global_grid = None
        self._global_grid_received_monotonic = None
        self._odom_info = None
        self._odom_info_received_monotonic = None
        self._latest_scan = None
        self._latest_scan_received_monotonic = None
        self._last_integrated_ros_ns = None
        self._last_blocked_stamp_ns = None
        self._goal_handle = None
        self._goal_cancel_reason = None
        self._goal_state = "idle"
        self._goal_unresolved = False
        self._goal_terminal_wall = None
        self._goal_terminal_ros_ns = None
        self._goal_finish_action = None
        self._nav2_goal_started_wall = None
        self._nav2_goal_target_map = None
        self._nav2_has_moved = False
        self._nav2_latest_command = None
        self._nav2_latest_command_wall = None
        self._goal_refresh_pending = False
        self._fault_latched = False
        self._last_selected_source = "stop"
        self._last_selected_twist = Twist()
        self._cycle_counts = {
            "timer_calls": 0,
            "disabled": 0,
            "no_target": 0,
            "stale_received_target": 0,
            "ros_time_zero": 0,
            "ros_time_zero_cycles": 0,
            "stale_ros_stamp": 0,
            "tf_error": 0,
            "invalid_target": 0,
            "valid": 0,
            # This observation can overlap a terminal outcome.
            "ros_time_stalled_cycles": 0,
        }
        self._last_ros_time_ns = None
        self._last_ros_time_advance_monotonic = time.monotonic()
        self._last_diagnostics_monotonic = time.monotonic()
        self._controller = self._make_controller()
        self.get_logger().info("FOLLOW_CONFIG " + json.dumps({
            "process_id": os.getpid(),
            "node": self.get_fully_qualified_name(),
            "distance_kp": self._controller.distance_kp,
            "distance_ki": self._controller.distance_ki,
            "distance_kd": self._controller.distance_kd,
            "distance_filter_tau_s": self._controller.distance_filter_tau_s,
            "distance_integral_limit_mps": self._controller.distance_integral_limit_mps,
            "direction_kp": self._controller.direction_kp,
            "direction_kd": self._controller.direction_kd,
            "direction_ki": self._controller.direction_ki,
            "direction_integral_limit_rad": self._controller.direction_integral_limit_rad,
            "direction_filter_tau_s": self._controller.direction_filter_tau_s,
            "target_distance_m": self._controller.target_distance_m,
            "max_speed_mps": self._controller.max_speed_mps,
            "max_steering_rad": self._controller.max_steer_rad,
            "publish_cmd_vel": self._publish_cmd_vel,
        }, sort_keys=True, allow_nan=False))
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._enable_service = self.create_service(
            SetBool,
            str(self.get_parameter("enable_service").value),
            self._set_enabled,
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
        self._nav2_action = None
        if self._follow_nav2_avoidance:
            grid_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._local_grid_subscription = self.create_subscription(
                OccupancyGrid,
                str(self.get_parameter("local_costmap_topic").value),
                self._on_local_grid,
                grid_qos,
            )
            self._global_grid_subscription = self.create_subscription(
                OccupancyGrid,
                str(self.get_parameter("global_costmap_topic").value),
                self._on_global_grid,
                grid_qos,
            )
            self._odom_info_subscription = self.create_subscription(
                OdomInfo,
                str(self.get_parameter("odom_info_topic").value),
                self._on_odom_info,
                qos_profile_sensor_data,
            )
            self._scan_subscription = self.create_subscription(
                LaserScan, "/piracer/scan", self._on_scan, qos_profile_sensor_data
            )
            self._nav2_subscription = self.create_subscription(
                Twist,
                str(self.get_parameter("nav2_command_topic").value),
                self._on_nav2_command,
                10,
            )
            if self._publish_cmd_vel:
                self._nav2_action = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self._control_sample_publisher = self.create_publisher(
            String,
            str(self.get_parameter("control_sample_topic").value),
            10,
        )
        self._control_diagnostics_publisher = self.create_publisher(
            String,
            str(self.get_parameter("control_diagnostics_topic").value),
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
            distance_kp=float(self.get_parameter("distance_kp").value),
            distance_ki=float(self.get_parameter("distance_ki").value),
            distance_kd=float(self.get_parameter("distance_kd").value),
            distance_filter_tau_s=float(self.get_parameter("distance_filter_tau_s").value),
            distance_integral_limit_mps=float(
                self.get_parameter("distance_integral_limit_mps").value
            ),
            direction_kp=float(self.get_parameter("direction_kp").value),
            direction_kd=float(self.get_parameter("direction_kd").value),
            direction_ki=float(self.get_parameter("direction_ki").value),
            direction_integral_limit_rad=float(
                self.get_parameter("direction_integral_limit_rad").value
            ),
            direction_filter_tau_s=float(self.get_parameter("direction_filter_tau_s").value),
            max_speed_mps=float(self.get_parameter("max_speed_mps").value),
            max_steer_rad=float(self.get_parameter("max_steer_rad").value),
        )

    def _on_target_point(self, message):
        self._latest_target_point = message
        self._latest_target_received_monotonic = time.monotonic()

    def _on_local_grid(self, message):
        self._local_grid = message
        self._local_grid_received_monotonic = time.monotonic()

    def _on_global_grid(self, message):
        self._global_grid = message
        self._global_grid_received_monotonic = time.monotonic()

    def _on_odom_info(self, message):
        self._odom_info = message
        self._odom_info_received_monotonic = time.monotonic()

    def _on_scan(self, message):
        self._latest_scan = message
        self._latest_scan_received_monotonic = time.monotonic()

    def _on_nav2_command(self, message):
        if not self._follow_nav2_avoidance or self._mode != NAV2 or not self._goal_handle:
            return
        self._nav2_latest_command = message
        self._nav2_latest_command_wall = time.monotonic()
        bounded = bound_nav_twist(float(message.linear.x), float(message.angular.z))
        if bounded is not None and bounded[0] > 0.0:
            self._nav2_has_moved = True

    def _set_enabled(self, request, response):
        was_enabled = self._enabled
        if request.data and self._follow_nav2_avoidance and self._goal_unresolved:
            response.success = False
            response.message = "Follow enable waits for Nav2 goal ownership to resolve"
            return response
        self._enabled = bool(request.data)
        if was_enabled != self._enabled:
            self._controller.reset("enable_transition")
        if request.data:
            if self._follow_nav2_avoidance and not was_enabled:
                self._fault_latched = False
                self._mode = DIRECT
                self._mode_reason = "enabled"
                self._clearance.reset()
                self._last_blocked_stamp_ns = None
                self._nav2_latest_command = None
            response.success = True
            response.message = "Follow enabled; waiting for a fresh target point"
            if not was_enabled:
                self._publish_zero()
            return response

        self._clear_target()
        if self._follow_nav2_avoidance:
            self._mode = SAFE_STOP
            self._mode_reason = "disabled"
            if self._goal_unresolved:
                self._request_goal_cancel("disabled")
            self._publish_zero()
            response.success = not self._goal_unresolved
            response.message = (
                "Follow disabled"
                if response.success
                else "Follow disabled; waiting for Nav2 terminal result and smoother drain"
            )
            return response
        if was_enabled:
            self._publish_zero()
        response.success = True
        response.message = "Follow disabled"
        return response

    def _clear_target(self):
        self._latest_target_point = None
        self._latest_target_received_monotonic = None

    def _publish_zero(self):
        self._controller.reset("stop")
        self._publish_command(Twist())

    def _publish_command(self, message):
        if self._publish_cmd_vel:
            self._command_publisher.publish(message)

    def _publish_selected_command(self, message, source):
        self._last_selected_source = source
        self._last_selected_twist = message
        self._publish_command(message)

    def _stop_for_input_error(self, reason):
        if self._follow_nav2_avoidance:
            self._safe_stop(reason)
        else:
            self._publish_zero()

    @staticmethod
    def _stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _yaw_from_quaternion(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _tf_pose(self, frame_id, now_ns):
        transform = self._tf_buffer.lookup_transform(frame_id, "base_link", Time())
        stamp = self._stamp_ns(transform)
        max_age_s = float(self.get_parameter("max_localization_age_s").value)
        if not stamped_input_is_fresh(stamp, now_ns, 0.0, max_age_s):
            age_s = (now_ns - stamp) / 1_000_000_000.0
            raise TransformException(f"stale TF {frame_id}->base_link, age={age_s:.3f}s")
        translation = transform.transform.translation
        return (
            float(translation.x),
            float(translation.y),
            self._yaw_from_quaternion(transform.transform.rotation),
        )

    def _integrated_inputs_fresh(self, now_ns, monotonic_now):
        if (
            self._local_grid is None or self._global_grid is None
            or self._odom_info is None or self._latest_scan is None
        ):
            return False, "missing_costmap_or_localization"
        local_stamp = self._stamp_ns(self._local_grid)
        global_stamp = self._stamp_ns(self._global_grid)
        local_age = monotonic_now - self._local_grid_received_monotonic
        global_age = monotonic_now - self._global_grid_received_monotonic
        if not stamped_input_is_fresh(
            local_stamp, now_ns, local_age,
            float(self.get_parameter("local_costmap_max_age_s").value),
        ):
            return False, "stale_local_costmap"
        if not stamped_input_is_fresh(
            global_stamp, now_ns, global_age,
            float(self.get_parameter("global_costmap_max_age_s").value),
        ):
            return False, "stale_global_costmap"
        scan_stamp = self._stamp_ns(self._latest_scan)
        scan_receipt_age = monotonic_now - self._latest_scan_received_monotonic
        if not stamped_input_is_fresh(
            scan_stamp, now_ns, scan_receipt_age, 0.5
        ):
            return False, "stale_scan"
        odom_stamp = self._stamp_ns(self._odom_info)
        max_localization_age = float(
            self.get_parameter("max_localization_age_s").value
        )
        if not localization_is_fresh(
            self._odom_info.lost,
            odom_stamp,
            now_ns,
            monotonic_now - self._odom_info_received_monotonic,
            max_localization_age,
        ):
            return False, "lost_or_stale_localization"
        return True, ""

    def _safe_stop(self, reason, latch=False):
        changed = self._mode != SAFE_STOP or self._mode_reason != reason
        self._mode = SAFE_STOP
        self._mode_reason = reason
        if latch:
            self._fault_latched = True
        self._nav2_latest_command = None
        if self._goal_unresolved:
            self._request_goal_cancel(reason)
        if changed:
            self._controller.reset("safe_stop")
        self._publish_selected_command(Twist(), "stop")
        if changed:
            self.get_logger().warning(f"FOLLOW_NAV2 SAFE_STOP reason={reason}")

    def _request_goal_cancel(self, reason, finish_action=None):
        self._goal_cancel_reason = reason
        if finish_action is not None:
            self._goal_finish_action = finish_action
        if self._goal_state == "sending":
            return
        if self._goal_state in ("active", "canceling") and self._goal_handle is not None:
            if self._goal_state == "active":
                self._goal_state = "canceling"
                try:
                    self._goal_handle.cancel_goal_async()
                except Exception as error:
                    self._goal_state = "unknown"
                    self._mode = SAFE_STOP
                    self._mode_reason = "cancel_request_exception"
                    self._fault_latched = True
                    self.get_logger().error(
                        f"FOLLOW_NAV2 cancel request outcome unknown: {error}"
                    )

    def _start_nav2_goal(self):
        if self._goal_unresolved or self._goal_state != "idle":
            return
        if not self._publish_cmd_vel or self._nav2_action is None:
            self._mode = NAV2_PENDING
            self._mode_reason = "dry_or_action_unavailable"
            return
        now_ns = self.get_clock().now().nanoseconds
        now = self.get_clock().now()
        try:
            target_map = self._tf_buffer.transform(self._latest_target_point, "map")
        except TransformException as error:
            self._safe_stop("target_map_tf_invalid")
            self.get_logger().warning(f"FOLLOW_NAV2 goal target transform failed: {error}")
            return
        target_stamp_ns = self._stamp_ns(target_map)
        if target_stamp_ns <= 0 or (now_ns - target_stamp_ns) > round(
            float(self.get_parameter("max_target_age_s").value) * 1_000_000_000
        ):
            self._safe_stop("stale_map_target")
            return
        try:
            robot_x, robot_y, _robot_yaw = self._tf_pose("map", now_ns)
        except TransformException as error:
            self._safe_stop("map_localization_tf_invalid")
            self.get_logger().warning(f"FOLLOW_NAV2 map/base TF rejected: {error}")
            return
        goal_values = stand_off_goal(
            robot_x,
            robot_y,
            float(target_map.point.x),
            float(target_map.point.y),
            self._controller.target_distance_m,
        )
        if goal_values is None:
            self._safe_stop("no_standoff_goal")
            return
        goal_x, goal_y, goal_yaw = goal_values
        if (
            self._global_grid.header.frame_id != "map"
            or not footprint_clear(self._global_grid, goal_x, goal_y, goal_yaw)
        ):
            self._safe_stop("occupied_or_unknown_standoff_goal")
            return
        global_stamp = self._stamp_ns(self._global_grid)
        if self._last_blocked_stamp_ns is None or global_stamp < self._last_blocked_stamp_ns:
            return
        if not self._nav2_action.server_is_ready():
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = now.to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(goal_yaw / 2.0)
        self._goal_state = "sending"
        self._goal_unresolved = True
        self._goal_cancel_reason = None
        self._goal_finish_action = None
        self._goal_refresh_pending = False
        self._nav2_goal_started_wall = time.monotonic()
        self._nav2_goal_target_map = (
            float(target_map.point.x), float(target_map.point.y)
        )
        self._nav2_has_moved = False
        self._nav2_latest_command = None
        self._nav2_latest_command_wall = None
        self._mode = NAV2_PENDING
        self._mode_reason = "goal_request_pending"
        try:
            goal_send_future = self._nav2_action.send_goal_async(goal)
            goal_send_future.add_done_callback(self._on_nav2_goal_response)
        except Exception as error:
            self._goal_state = "unknown"
            self._mode = SAFE_STOP
            self._mode_reason = "goal_request_exception"
            self._fault_latched = True
            self.get_logger().error(f"FOLLOW_NAV2 goal request outcome unknown: {error}")

    def _on_nav2_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as error:
            self._goal_state = "unknown"
            self._mode = SAFE_STOP
            self._mode_reason = "goal_acceptance_unknown"
            self._fault_latched = True
            self.get_logger().error(f"FOLLOW_NAV2 goal acceptance unknown: {error}")
            return
        if not handle.accepted:
            self._goal_state = "idle"
            self._goal_unresolved = False
            self._goal_handle = None
            self._mode = SAFE_STOP
            self._mode_reason = "goal_rejected"
            self._fault_latched = True
            self.get_logger().error("FOLLOW_NAV2 goal rejected")
            return
        self._goal_handle = handle
        self._goal_state = "active"
        self._mode = NAV2
        self._mode_reason = "goal_accepted"
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_nav2_goal_result)
        # A disable or safety stop can arrive while goal acceptance was pending.
        if not self._enabled or self._goal_cancel_reason is not None:
            self._request_goal_cancel(self._goal_cancel_reason or "disabled")

    def _on_nav2_goal_result(self, future):
        try:
            result = future.result()
        except Exception as error:
            self._goal_state = "unknown"
            self._mode = SAFE_STOP
            self._mode_reason = "goal_result_unknown"
            self._fault_latched = True
            self.get_logger().error(f"FOLLOW_NAV2 result unknown: {error}")
            return
        try:
            status = int(result.status)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self._goal_state = "unknown"
            self._mode = SAFE_STOP
            self._mode_reason = "goal_result_status_unknown"
            self._fault_latched = True
            self.get_logger().error(
                f"FOLLOW_NAV2 result status missing or invalid; ownership retained: {error}"
            )
            return
        terminal_status_names = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        status_name = terminal_status_names.get(status)
        if status_name is None:
            self._goal_state = "unknown"
            self._mode = SAFE_STOP
            self._mode_reason = f"goal_result_nonterminal_status_{status}"
            self._fault_latched = True
            self.get_logger().error(
                f"FOLLOW_NAV2 nonterminal result status={status}; ownership retained"
            )
            return
        self.get_logger().warning(
            f"FOLLOW_NAV2 terminal result status={status_name}({status}) "
            f"cancel_reason={self._goal_cancel_reason or 'none'}"
        )
        self._goal_terminal_status = status_name
        self._goal_state = "draining"
        self._goal_terminal_wall = time.monotonic()
        self._goal_terminal_ros_ns = self.get_clock().now().nanoseconds
        if status == GoalStatus.STATUS_CANCELED and self._goal_cancel_reason == "return_direct":
            self._goal_finish_action = "direct"
        elif status == GoalStatus.STATUS_CANCELED and self._goal_cancel_reason == "target_moved":
            self._goal_finish_action = "refresh"
        elif status == GoalStatus.STATUS_CANCELED and self._mode == SAFE_STOP and not self._fault_latched:
            self._goal_finish_action = "recover"
        elif status == GoalStatus.STATUS_SUCCEEDED and self._clearance.clear >= 5:
            self._goal_finish_action = "direct"
        else:
            self._goal_finish_action = None
            self._mode = SAFE_STOP
            self._mode_reason = "goal_failed_or_ended_while_blocked"
            self._fault_latched = True
        self._mode = SAFE_STOP if self._mode == SAFE_STOP else NAV2_PENDING
        # Keep the terminal outcome visible in the transition reason through
        # the smoother drain; the explicit log above also records its status.
        self._mode_reason = f"goal_terminal_{status_name.lower()}_tail_drain"
        self._nav2_latest_command = None

    def _finish_goal_drain(self, now_ns, monotonic_now):
        if self._goal_state != "draining":
            return
        if self._goal_terminal_wall is None or self._goal_terminal_ros_ns is None:
            return
        wall_elapsed = monotonic_now - self._goal_terminal_wall
        ros_elapsed = (now_ns - self._goal_terminal_ros_ns) / 1_000_000_000.0
        # The smoother timeout is ROS-time based; a paused simulation keeps STOP latched.
        if wall_elapsed < 0.3 or ros_elapsed < 0.3:
            self._mode = SAFE_STOP
            if self._goal_terminal_status is None:
                self._mode_reason = "smoother_tail_drain"
            else:
                self._mode_reason = f"goal_terminal_{self._goal_terminal_status.lower()}_tail_drain"
            return
        finish_action = self._goal_finish_action
        self._goal_state = "idle"
        self._goal_unresolved = False
        self._goal_handle = None
        self._goal_cancel_reason = None
        self._goal_terminal_wall = None
        self._goal_terminal_ros_ns = None
        self._goal_terminal_status = None
        self._goal_finish_action = None
        if finish_action == "direct" and self._enabled and not self._fault_latched:
            self._mode = DIRECT
            self._mode_reason = "direct_path_clear"
        elif finish_action == "refresh" and self._enabled and not self._fault_latched:
            self._mode = NAV2_PENDING
            self._mode_reason = "bounded_target_refresh"
            self._goal_refresh_pending = True
        elif finish_action == "recover" and self._enabled and not self._fault_latched:
            self._mode = SAFE_STOP
            self._mode_reason = "nav2_cancel_drained_data_recovery"
        else:
            self._mode = SAFE_STOP
            self._mode_reason = "goal_finished_without_clear_path"

    def _maybe_refresh_goal(self, target_map_xy, monotonic_now):
        if (
            self._goal_state not in ("active", "canceling")
            or self._nav2_goal_target_map is None
            or self._nav2_goal_started_wall is None
        ):
            return
        elapsed = monotonic_now - self._nav2_goal_started_wall
        if goal_refresh_due(
            target_map_xy[0], target_map_xy[1],
            self._nav2_goal_target_map[0], self._nav2_goal_target_map[1], elapsed,
        ):
            self._goal_refresh_pending = True
            self._request_goal_cancel("target_moved", "refresh")
            self._mode = NAV2_PENDING
            self._mode_reason = "bounded_target_refresh_cancel"

    def _integrated_cycle(self, command, base_target, now_ns, monotonic_now):
        """Select one direct/Nav2/zero command on the existing steady timer."""
        self._finish_goal_drain(now_ns, monotonic_now)
        if now_ns <= 0:
            self._safe_stop("ros_time_invalid")
            return
        if self._last_integrated_ros_ns is not None and now_ns < self._last_integrated_ros_ns:
            self._safe_stop("ros_time_regression")
            self._clearance.reset()
            self._last_integrated_ros_ns = now_ns
            return
        self._last_integrated_ros_ns = now_ns
        if not self._enabled:
            self._publish_selected_command(Twist(), "stop")
            return
        if not self._publish_cmd_vel:
            # Dry mode never sends an action goal or a Twist on either output.
            return
        if self._fault_latched:
            self._publish_selected_command(Twist(), "stop")
            return

        fresh, reason = self._integrated_inputs_fresh(now_ns, monotonic_now)
        if not fresh:
            self._safe_stop(reason)
            return
        if self._mode == SAFE_STOP and self._goal_unresolved:
            # Keep the owned Nav2 action in a fail-closed stop while its
            # cancellation/result or smoother drain is unresolved. Preserve
            # the reason that initiated the stop for diagnostics.
            self._publish_selected_command(Twist(), "stop")
            return
        if self._mode == SAFE_STOP and not self._goal_unresolved:
            # Data waits are recoverable. Re-enter the direct-path evaluator;
            # its distinct-map hysteresis still guards any actual handoff.
            self._mode = DIRECT
            self._mode_reason = "fresh_inputs_recovered"
        target_x = float(base_target.point.x)
        target_y = float(base_target.point.y)
        minimum_distance = float(self._controller.min_distance_m)
        if target_x <= 0.0 or math.hypot(target_x, target_y) <= minimum_distance:
            self._safe_stop("target_inside_min_distance")
            return
        try:
            grid_pose = self._tf_pose(self._local_grid.header.frame_id, now_ns)
            self._tf_pose("map", now_ns)
            local_yaw = grid_pose[2]
            if not footprint_clear(
                self._local_grid, grid_pose[0], grid_pose[1], local_yaw
            ):
                self._safe_stop("current_footprint_occupied")
                return
        except (TransformException, ValueError, TypeError) as error:
            self._safe_stop("localization_tf_invalid")
            self.get_logger().warning(f"FOLLOW_NAV2 localization TF rejected: {error}")
            return

        target_map = None
        try:
            target_map = self._tf_buffer.transform(self._latest_target_point, "map")
            target_map_xy = (float(target_map.point.x), float(target_map.point.y))
            if not all(math.isfinite(value) for value in target_map_xy):
                raise ValueError("nonfinite map target")
        except (TransformException, ValueError, TypeError) as error:
            self._safe_stop("target_map_tf_invalid")
            self.get_logger().warning(f"FOLLOW_NAV2 target map transform rejected: {error}")
            return
        if self._mode == NAV2:
            self._maybe_refresh_goal(target_map_xy, monotonic_now)

        remaining = max(0.0, float(base_target.point.x) ** 2 +
                        float(base_target.point.y) ** 2) ** 0.5
        remaining = max(0.0, remaining - self._controller.target_distance_m)
        horizon = min(0.6, remaining)
        direct_clear, collision_distance = swept_footprint_clear(
            self._local_grid,
            grid_pose[0], grid_pose[1], local_yaw,
            float(command.linear_x), float(command.angular_z), horizon,
        )
        local_stamp_ns = self._stamp_ns(self._local_grid)
        clearance = self._clearance.observe(not direct_clear, local_stamp_ns)
        if clearance.reason in ("blocked_first_map", "blocked_confirmed"):
            if clearance.confirm_nav2 and self._last_blocked_stamp_ns is None:
                self._last_blocked_stamp_ns = local_stamp_ns
        if clearance.return_direct:
            self._last_blocked_stamp_ns = None
        if clearance.reason == "invalid_costmap_stamp":
            self._safe_stop("invalid_local_costmap_stamp")
            return
        if (self._mode == DIRECT and not direct_clear and
                collision_distance <= 0.10 and not clearance.confirm_nav2):
            self._safe_stop("imminent_swept_footprint_collision")
            return

        if self._mode == DIRECT:
            if clearance.confirm_nav2:
                self._mode = NAV2_PENDING
                self._mode_reason = "direct_corridor_blocked"
                self.get_logger().warning(
                    f"FOLLOW_NAV2 transition DIRECT->NAV2_PENDING stamp_ns={local_stamp_ns}"
                )
                self._publish_selected_command(Twist(), "stop")
                self._start_nav2_goal()
            elif clearance.stop_now:
                self._publish_selected_command(Twist(), "stop")
            else:
                direct = Twist()
                direct.linear.x = command.linear_x
                direct.angular.z = command.angular_z
                self._publish_selected_command(direct, "direct")
            return

        if self._mode in (NAV2_PENDING, NAV2):
            if clearance.return_direct and not self._goal_unresolved:
                self._mode = DIRECT
                self._mode_reason = "direct_path_clear_without_active_goal"
                self._publish_selected_command(Twist(), "stop")
                return
            if clearance.return_direct and self._goal_unresolved:
                self._request_goal_cancel("return_direct", "direct")
                self._mode = NAV2_PENDING
                self._mode_reason = "waiting_for_nav2_cancel_terminal"
                self._nav2_latest_command = None
                self._publish_selected_command(Twist(), "stop")
                return
            if self._goal_refresh_pending and not self._goal_unresolved:
                self._start_nav2_goal()
                self._publish_selected_command(Twist(), "stop")
                return
            if self._mode == NAV2_PENDING:
                self._publish_selected_command(Twist(), "stop")
                self._start_nav2_goal()
                return

            # Keep waiting through the planner's bounded planning time if it has
            # not emitted motion yet. Once motion started, a missing stream is a fault.
            message = self._nav2_latest_command
            command_wall = self._nav2_latest_command_wall
            if message is None or command_wall is None:
                self._publish_selected_command(Twist(), "stop")
                return
            nav_age = monotonic_now - command_wall
            linear = float(message.linear.x)
            angular = float(message.angular.z)
            policy, bounded = nav2_command_policy(
                linear, angular, nav_age, self._nav2_has_moved
            )
            if policy == "fault":
                self._safe_stop("invalid_or_stale_nav2_twist")
                return
            if policy == "wait":
                self._publish_selected_command(Twist(), "stop")
                return
            nav_clear, nav_collision_distance = swept_footprint_clear(
                self._local_grid,
                grid_pose[0], grid_pose[1], local_yaw,
                bounded[0], bounded[1], 0.10,
            )
            if not nav_clear or nav_collision_distance <= 0.10:
                self._safe_stop("nav2_imminent_footprint_collision")
                return
            selected = Twist()
            selected.linear.x, selected.angular.z = bounded
            self._publish_selected_command(selected, "nav2")
            return

        self._safe_stop("unknown_follow_mode")

    def _observe_ros_time(self, now_ns, monotonic_now, stall_threshold_s):
        """Observe a paused nonzero ROS clock without changing control behavior."""
        if now_ns <= 0:
            return False
        if self._last_ros_time_ns is None or now_ns > self._last_ros_time_ns:
            self._last_ros_time_ns = now_ns
            self._last_ros_time_advance_monotonic = monotonic_now
            return False
        if now_ns < self._last_ros_time_ns:
            # A backward jump starts a new clock epoch, not a stationary clock.
            self._last_ros_time_ns = now_ns
            self._last_ros_time_advance_monotonic = monotonic_now
            self._controller.reset("ros_clock_regression")
            return False
        return (
            monotonic_now - self._last_ros_time_advance_monotonic
            >= stall_threshold_s
        )

    def _record_cycle(self, outcome, ros_time_stalled, ros_time_ns, monotonic_now):
        self._cycle_counts[outcome] += 1
        if ros_time_stalled:
            self._cycle_counts["ros_time_stalled_cycles"] += 1

        if monotonic_now - self._last_diagnostics_monotonic < 1.0:
            return

        outcome_names = (
            "disabled",
            "no_target",
            "stale_received_target",
            "ros_time_zero",
            "stale_ros_stamp",
            "tf_error",
            "invalid_target",
            "valid",
        )
        completed = sum(self._cycle_counts[name] for name in outcome_names)
        summary = {
            "schema_version": 1,
            "process_id": os.getpid(),
            "sim_time_ns": int(ros_time_ns),
            "cycle_counts": dict(self._cycle_counts),
            "unclassified_cycles": self._cycle_counts["timer_calls"] - completed,
        }
        message = String()
        message.data = json.dumps(summary, separators=(",", ":"), allow_nan=False)
        self._control_diagnostics_publisher.publish(message)
        self._last_diagnostics_monotonic = monotonic_now

    def _control_cycle(self):
        monotonic_now = time.monotonic()
        self._cycle_counts["timer_calls"] += 1
        max_age_s = float(self.get_parameter("max_target_age_s").value)
        clock_observation = self.get_clock().now()
        now_ns = clock_observation.nanoseconds
        if now_ns == 0:
            self._cycle_counts["ros_time_zero_cycles"] += 1
        ros_time_stalled = self._observe_ros_time(
            now_ns, monotonic_now, max_age_s
        )
        if self._follow_nav2_avoidance:
            self._finish_goal_drain(now_ns, monotonic_now)

        if not self._enabled:
            self._controller.reset("disabled")
            if self._follow_nav2_avoidance:
                self._publish_selected_command(Twist(), "stop")
            self._record_cycle("disabled", ros_time_stalled, now_ns, monotonic_now)
            return

        if self._latest_target_point is None:
            self._stop_for_input_error("no_target")
            self._record_cycle("no_target", ros_time_stalled, now_ns, monotonic_now)
            return

        received_at = self._latest_target_received_monotonic
        if received_at is None or time.monotonic() - received_at > max_age_s:
            self._stop_for_input_error("stale_received_target")
            self._record_cycle(
                "stale_received_target", ros_time_stalled, now_ns, monotonic_now
            )
            return

        # Keep the control calculation on a fresh ROS-time read, as before the
        # diagnostic clock observation was added.
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        ros_time_stalled = self._observe_ros_time(
            now_ns, monotonic_now, max_age_s
        )
        stamp = Time.from_msg(
            self._latest_target_point.header.stamp,
            clock_type=self.get_clock().clock_type,
        )
        if now_ns == 0:
            self._stop_for_input_error("ros_time_zero")
            self._record_cycle("ros_time_zero", ros_time_stalled, now_ns, monotonic_now)
            return
        age_s = (now - stamp).nanoseconds / 1_000_000_000.0
        if age_s < 0.0 or age_s > max_age_s:
            self._stop_for_input_error("stale_ros_stamp")
            self._record_cycle("stale_ros_stamp", ros_time_stalled, now_ns, monotonic_now)
            return

        base_frame = str(self.get_parameter("base_frame").value)
        try:
            # Transform at the image/point timestamp, not at the latest TF.
            target = self._tf_buffer.transform(self._latest_target_point, base_frame)
        except TransformException as exc:
            self.get_logger().debug(f"Follow stopped for invalid target or TF: {exc}")
            self._stop_for_input_error("target_tf_error")
            self._record_cycle("tf_error", ros_time_stalled, now_ns, monotonic_now)
            return
        except (ValueError, TypeError) as exc:
            self.get_logger().debug(f"Follow stopped for invalid target or TF: {exc}")
            self._stop_for_input_error("invalid_target")
            self._record_cycle("invalid_target", ros_time_stalled, now_ns, monotonic_now)
            return

        try:
            x_m = float(target.point.x)
            y_m = float(target.point.y)
            target_stamp_ns = self._stamp_ns(self._latest_target_point)
            command = self._controller.update(
                x_m, y_m, measurement_stamp_ns=target_stamp_ns
            )
        except (ValueError, TypeError) as exc:
            self.get_logger().debug(f"Follow stopped for invalid target or TF: {exc}")
            self._stop_for_input_error("invalid_target")
            self._record_cycle("invalid_target", ros_time_stalled, now_ns, monotonic_now)
            return

        if self._follow_nav2_avoidance:
            if self._fault_latched:
                self._controller.reset("safe_stop")
            self._integrated_cycle(command, target, now.nanoseconds, monotonic_now)
            self._publish_control_sample(command, target, now.nanoseconds)
        else:
            self._publish_control_sample(command, target, now.nanoseconds)
            message = Twist()
            message.linear.x = command.linear_x
            message.angular.z = command.angular_z
            self._publish_command(message)
        self._record_cycle("valid", ros_time_stalled, now_ns, monotonic_now)

    def _publish_control_sample(self, command, target, sim_time_ns):
        sample = {
            "schema_version": 1,
            "process_id": os.getpid(),
            "sim_time_ns": int(sim_time_ns),
            "target_stamp_ns": (
                int(self._latest_target_point.header.stamp.sec) * 1_000_000_000
                + int(self._latest_target_point.header.stamp.nanosec)
            ),
            "target_x_base_m": float(target.point.x),
            "target_y_base_m": float(target.point.y),
            "distance_m": float(command.distance_m),
            "distance_error_m": float(
                command.distance_m - self._controller.target_distance_m
            ),
            "direction_error_rad": float(command.bearing_rad),
            "command_linear_x_mps": float(command.linear_x),
            "command_angular_z_radps": float(command.angular_z),
            "steering_angle_command_rad": float(command.steering_angle_rad),
            "direction_p_rad": float(command.direction_p_rad),
            "direction_d_rad": float(command.direction_d_rad),
            "direction_i_rad": float(command.direction_i_rad),
            "direction_filtered_derivative_radps": float(command.direction_d_filtered_radps),
            "distance_p_mps": float(command.distance_p_mps),
            "distance_i_mps": float(command.distance_i_mps),
            "distance_d_mps": float(command.distance_d_mps),
            "distance_filtered_derivative_mps": float(command.distance_d_filtered_mps),
            "distance_raw_derivative_mps": float(command.distance_d_raw_mps),
            "distance_integral_m_s": (
                float(command.integral_state_mps / self._controller.distance_ki)
                if self._controller.distance_ki > 0.0 else 0.0
            ),
            "speed_raw_mps": float(command.distance_unsaturated_mps),
            "steering_raw_rad": float(
                command.direction_p_rad + command.direction_d_rad + command.direction_i_rad
            ),
            "speed_saturated": bool(command.distance_saturated_low or command.distance_saturated_high),
            "steering_saturated": bool(command.direction_saturated),
            "measurement_dt_s": float(command.measurement_dt_s) if command.measurement_dt_s > 0.0 else None,
            "new_measurement": bool(command.measurement_new),
            "reset_reason": str(self._controller._last_reset_reason),
            "publish_cmd_vel_enabled": self._publish_cmd_vel,
            "selected_source": self._last_selected_source,
            "selected_linear_x_mps": float(self._last_selected_twist.linear.x),
            "selected_angular_z_radps": float(self._last_selected_twist.angular.z),
            "follow_nav2_mode": self._mode,
            "follow_nav2_reason": self._mode_reason,
            "nav2_goal_state": self._goal_state,
            "nav2_goal_unresolved": bool(self._goal_unresolved),
        }
        message = String()
        message.data = json.dumps(sample, separators=(",", ":"), allow_nan=False)
        self._control_sample_publisher.publish(message)

    def stop_and_destroy(self):
        self._controller.reset("shutdown")
        if self._follow_nav2_avoidance and self._goal_unresolved:
            self._request_goal_cancel("shutdown")
        if self._enabled:
            self._publish_zero()
        self._enabled = False
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
