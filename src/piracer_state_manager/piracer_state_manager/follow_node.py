"""Follow the latest RGB-D suitcase point with bounded P control."""

import json
import math
import os
import time
from threading import Thread

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory
from rclpy.clock import Clock, ClockType
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import try_shutdown
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, ExtrapolationException, TransformException, TransformListener
from rtabmap_msgs.msg import OdomInfo
from vision_msgs.msg import Detection2DArray
import tf2_geometry_msgs  # Register PointStamped transforms with tf2.

from piracer_state_manager.follow_control import FollowPController
from piracer_state_manager.follow_nav2_helpers import (
    DIRECT,
    NAV2,
    NAV2_PENDING,
    SAFE_STOP,
    bound_nav_twist,
    clear_stand_off_goal,
    footprint_clear,
    goal_refresh_due,
    localization_is_fresh,
    nav2_command_policy,
    require_distinct_command_topics,
    scan_clears_motion,
    stand_off_goal,
    stamped_input_is_fresh,
    swept_footprint_clear,
)


_DATA_WAIT_REASONS = (
    "stale_received_target", "stale_ros_stamp", "stale_scan",
    "stale_local_costmap", "stale_global_costmap",
    "ros_clock_stalled", "awaiting_target_confirmation", "target_tf_pending",
)
_LAST_SEEN_BUDGET_S = 120
_DIRECT_CLEAR_SPAN_NS = 500_000_000


class PiracerFollowNode(Node):
    def __init__(self):
        super().__init__("piracer_follow")
        self._tf_node = self._tf_executor = self._tf_thread = self._tf_listener = None
        self.declare_parameter(
            "target_distance_m", 2.25, descriptor=ParameterDescriptor(read_only=True))
        defaults = {
            "enable_service": "/follow/enable",
            "detection_topic": "/yolo/detections",
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
        self._latest_detections = None
        self._detection_receipt = None
        self._last_seen = None  # map x, y, original image stamp, receipt time
        self._memory_goal = None
        self._nav2_goal_offset = None  # fixed map-frame approach relative to the suitcase
        self._standoff_id = None
        self._active_actions = {}
        self._latest_target_point = None
        self._latest_target_received_monotonic = None
        self._pending_target_point = None  # original message and first receipt
        self._target_input_stamp_ns = None  # do not renew a replay after expiry
        self._mode = DIRECT if not self._follow_nav2_avoidance else SAFE_STOP
        self._mode_reason = "disabled"
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
        self._direct_clear_since_ns = None  # first image stamp with a safe target corridor
        self._goal_send_future = None
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
            "last_seen": 0,
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
            1,
        )
        self._command_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter("command_topic").value),
            10,
        )
        self._nav2_action = None
        self._nav2_goal_update_publisher = None
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
                LaserScan, "/piracer/obstacle_scan", self._on_scan, qos_profile_sensor_data
            )
            self._nav2_subscription = self.create_subscription(
                Twist,
                str(self.get_parameter("nav2_command_topic").value),
                self._on_nav2_command,
                10,
            )
            if self._publish_cmd_vel:
                self._nav2_goal_update_publisher = self.create_publisher(
                    PoseStamped, "/piracer/follow/nav2_goal_update", 1
                )
                status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
                for action, endpoint in (("navigate_to_pose", "/navigate_to_pose"),
                                         ("navigate_through_poses", "/navigate_through_poses"),
                                         ("follow_path", "/piracer/follow_nav/follow_path")):
                    self.create_subscription(
                        GoalStatusArray, f"{endpoint}/_action/status",
                        lambda msg, name=action: self._on_action_status(name, msg), status_qos,
                    )
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
        self._status_publisher = self.create_publisher(
            String, "/piracer/follow/status",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
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
            self._control_tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        # Only the thread-safe TF buffer is shared with this receiver.
        # Control, detections and Nav2 futures remain on the original executor.
        remaps = ["--ros-args"]
        for topic in ("/tf", "/tf_static", "/clock"):
            remaps.extend(["-r", f"{topic}:={self.resolve_topic_name(topic)}"])
        try:
            self._tf_node = Node(
                "piracer_follow_tf_listener", context=self.context,
                namespace=self.get_namespace(), use_global_arguments=False,
                cli_args=remaps, enable_rosout=False, start_parameter_services=False,
                parameter_overrides=[Parameter(
                    "use_sim_time", value=bool(self.get_parameter("use_sim_time").value))],
            )
            self._tf_executor = SingleThreadedExecutor(context=self.context)
            self._tf_executor.add_node(self._tf_node)
            self._tf_listener = TransformListener(self._tf_buffer, self._tf_node)
            self._tf_thread = Thread(target=self._receive_tf, name="piracer-follow-tf")
            self._tf_thread.start()
        except BaseException:
            self.destroy_node()
            raise

    def _receive_tf(self):
        try:
            self._tf_executor.spin()
        except ExternalShutdownException:
            pass

    def destroy_node(self):
        if self._tf_executor is not None:
            self._tf_executor.shutdown()
        if self._tf_thread is not None and self._tf_thread.ident is not None:
            self._tf_thread.join()
        if self._tf_listener is not None:
            self._tf_listener.unregister()
        if self._tf_node is not None:
            self._tf_node.executor = None
            self._tf_node.destroy_node()
            self._tf_node = None
        return super().destroy_node()

    def _make_controller(self):
        target_distance = float(self.get_parameter("target_distance_m").value)
        return FollowPController(
            target_distance_m=target_distance,
            min_distance_m=target_distance - 0.25,  # Existing 25 cm STOP margin.
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

    def _on_detections(self, message):
        ahead_ns = self._stamp_ns(message) - self.get_clock().now().nanoseconds
        if ahead_ns > 0:
            if ahead_ns > 200_000_000:
                self._latest_target_received_monotonic = None
                self._safe_stop("future_detection_stamp")
            return  # Do not cache a future image, or refresh its receipt time.
        self._latest_detections = message
        self._detection_receipt = time.monotonic()

    def _on_target_point(self, message):
        received = time.monotonic()
        stamp = self._stamp_ns(message)
        if (self._latest_target_point is not None and stamp < self._stamp_ns(self._latest_target_point)
                or self._last_seen is not None and stamp < self._last_seen[2]):
            self._latest_target_received_monotonic = None
            self._safe_stop("target_stamp_regression")
            return
        if self._memory_goal is not None and stamp == self._last_seen[2]:
            return  # A repeated point is not a new observation of the suitcase.
        ahead_ns = stamp - self.get_clock().now().nanoseconds
        if ahead_ns > 0:
            if ahead_ns > 200_000_000:
                self._latest_target_received_monotonic = None
                self._safe_stop("future_target_stamp")
            return  # The old cache keeps its original stamp and freshness limit.
        if not self._follow_nav2_avoidance:
            self._accept_target_point(message, received)
            return
        previous = self._target_input_stamp_ns
        if previous is not None and stamp <= previous:
            return
        self._target_input_stamp_ns = stamp
        try:
            live = self._tf_buffer.transform(message, "base_link").point
            if not all(math.isfinite(v) for v in (live.x, live.y, live.z)):
                raise ValueError("invalid live target")
            if live.x <= 0 or math.hypot(live.x, live.y) <= self._controller.min_distance_m:
                self._latest_target_received_monotonic = None
                self._safe_stop("target_inside_min_distance")
                return
        except ExtrapolationException:
            pass  # The same pending point will also wait for its static TF.
        except (TransformException, ValueError, TypeError):
            self._latest_target_received_monotonic = None
            self._safe_stop("target_map_tf_invalid")
            return
        if self._pending_target_point is None:
            self._pending_target_point = (message, received)
            self._retry_target_point()

    def _retry_target_point(self):
        pending = self._pending_target_point
        if pending is None:
            return False
        if self._fault_latched:
            self._pending_target_point = None
            return False
        message, received = pending
        now_ns = self.get_clock().now().nanoseconds
        max_age = float(self.get_parameter("max_target_age_s").value)
        age = time.monotonic() - received
        if not stamped_input_is_fresh(self._stamp_ns(message), now_ns, age, max_age, 0.0):
            self._pending_target_point = None
            self._stop_for_input_error("stale_received_target" if age > max_age else "stale_ros_stamp")
            return False
        try:
            self._tf_buffer.transform(message, "map")
            self._tf_buffer.transform(message, "base_link")
        except ExtrapolationException:
            cached, receipt = self._latest_target_point, self._latest_target_received_monotonic
            if (cached is not None and receipt is not None
                    and stamped_input_is_fresh(self._stamp_ns(cached), now_ns,
                                              time.monotonic() - receipt, max_age, 0.0)):
                if self._goal_unresolved and not self._can_hold_data_wait(now_ns, time.monotonic()):
                    self._safe_stop("target_tf_pending")
                    return True
                return False  # The ordinary cycle still validates the accepted measurement.
            self._safe_stop("target_tf_pending")
            return True
        except (TransformException, ValueError, TypeError):
            self._latest_target_received_monotonic = None
            self._safe_stop("target_map_tf_invalid")
            return False
        self._pending_target_point = None
        self._accept_target_point(message, received)
        return False

    def _accept_target_point(self, message, received):
        stamp = self._stamp_ns(message)
        if self._mode in (NAV2_PENDING, NAV2) or self._memory_goal is not None:
            try:
                stamp = self._stamp_ns(message)
                now_ns = self.get_clock().now().nanoseconds
                if not stamped_input_is_fresh(stamp, now_ns, time.monotonic() - received,
                                              float(self.get_parameter("max_target_age_s").value), 0.0):
                    self._safe_stop("stale_ros_stamp")
                    return
                point = self._tf_buffer.transform(message, "map").point
                if not all(math.isfinite(v) for v in (point.x, point.y, point.z)):
                    raise ValueError("invalid map point")
                live = self._tf_buffer.transform(message, "base_link").point
                if not all(math.isfinite(v) for v in (live.x, live.y, live.z)):
                    raise ValueError("invalid live point")
                if live.x <= 0 or math.hypot(live.x, live.y) <= self._controller.min_distance_m:
                    self._latest_target_received_monotonic = None
                    self._safe_stop("target_inside_min_distance")
                    return
                # Accept the measurement only after its source-time TF is available.
                self._latest_target_point = message
                self._latest_target_received_monotonic = received
                self._remember_target((point.x, point.y), stamp)
                if self._memory_goal is not None:
                    self._memory_goal = None
                    self._direct_clear_since_ns = None
                    self._controller.reset("target_reacquired")
                    if self._goal_state in ("idle", "draining"):
                        self._nav2_goal_offset = self._nav2_goal_target_map = None
                        self._goal_refresh_pending = False
                        if self._goal_state == "draining":
                            self._goal_finish_action = "recover"
            except ExtrapolationException:
                self._safe_stop("target_tf_pending")
            except (TransformException, ValueError, TypeError) as error:
                self._latest_target_received_monotonic = None
                self._safe_stop("target_map_tf_invalid")
                self.get_logger().warning(f"FOLLOW target map rejected: {type(error).__name__}: {error}")
        else:
            self._latest_target_point = message
            self._latest_target_received_monotonic = received

    def _on_local_grid(self, message):
        self._local_grid = message
        self._local_grid_received_monotonic = time.monotonic()

    def _on_global_grid(self, message):
        self._global_grid = message
        self._global_grid_received_monotonic = time.monotonic()

    def _on_odom_info(self, message):
        if message.lost and (self._last_seen is not None
                             or self._pending_target_point is not None):
            self._safe_stop("lost_or_stale_localization")
        self._odom_info = message
        self._odom_info_received_monotonic = time.monotonic()

    def _on_scan(self, message):
        self._latest_scan = message
        self._latest_scan_received_monotonic = time.monotonic()

    def _on_nav2_command(self, message):
        if (not self._follow_nav2_avoidance or self._mode != NAV2 or not self._goal_handle
                or self._goal_state != "active" or self._goal_cancel_reason is not None):
            return
        if not self._memory_command_owned():
            self._nav2_latest_command = None
            return
        self._nav2_latest_command = message
        self._nav2_latest_command_wall = time.monotonic()
        bounded = bound_nav_twist(
            float(message.linear.x), float(message.angular.z), self._controller.max_speed_mps,
        )
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
                self._nav2_goal_offset = None
                self._fault_latched = False
                self._mode = DIRECT
                self._mode_reason = "enabled"
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
            response.message += f"; fault_latched={str(self._fault_latched).lower()}"
            return response
        if was_enabled:
            self._publish_zero()
        response.success = True
        response.message = "Follow disabled"
        return response

    def _clear_target(self):
        self._direct_clear_since_ns = None
        self._pending_target_point = None
        self._nav2_goal_offset = None
        self._nav2_goal_target_map = None
        self._last_seen = self._memory_goal = None
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

    @classmethod
    def _planar_pose(cls, transform):
        p, q = transform.translation, transform.rotation
        norm2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if not (all(math.isfinite(v) for v in (p.x, p.y, p.z, norm2))
                and abs(norm2 - 1.0) < .02):
            raise TransformException("invalid translation or rotation")
        return p.x, p.y, cls._yaw_from_quaternion(q)

    def _tf_pose(self, frame_id, now_ns):
        transform = self._tf_buffer.lookup_transform(frame_id, "base_link", Time())
        stamp = self._stamp_ns(transform)
        max_age_s = float(self.get_parameter("max_localization_age_s").value)
        if not stamped_input_is_fresh(stamp, now_ns, 0.0, max_age_s):
            age_s = (now_ns - stamp) / 1_000_000_000.0
            raise TransformException(f"stale TF {frame_id}->base_link, age={age_s:.3f}s")
        return self._planar_pose(transform.transform)

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

    def _can_hold_data_wait(self, now_ns, wall):
        if (not self._enabled or not self._publish_cmd_vel or self._fault_latched
                or not self._goal_unresolved and self._goal_state != "idle"
                or self._goal_cancel_reason is not None
                or self._last_seen is None):
            return False
        _, _, stamp, receipt = self._last_seen
        if (now_ns <= 0 or not 0 <= now_ns - stamp <= _LAST_SEEN_BUDGET_S * 1_000_000_000
                or not 0 <= wall - receipt <= _LAST_SEEN_BUDGET_S
                or self._stamp_ns(self._latest_target_point) != stamp
                or (self._last_integrated_ros_ns is not None and now_ns < self._last_integrated_ros_ns)):
            return False
        if self._odom_info is None or self._odom_info.lost:
            return False
        for message, received in (
                (self._local_grid, self._local_grid_received_monotonic),
                (self._global_grid, self._global_grid_received_monotonic),
                (self._latest_scan, self._latest_scan_received_monotonic),
                (self._odom_info, self._odom_info_received_monotonic)):
            if (message is None or received is None
                    or not math.isfinite(wall - received) or wall < received
                    or not 0 < self._stamp_ns(message) <= now_ns + 200_000_000):
                return False
        detection = self._latest_detections
        if detection is not None and not 0 < self._stamp_ns(detection) <= now_ns:
            return False
        try:
            self._tf_pose("map", now_ns)
            self._tf_pose(self._local_grid.header.frame_id, now_ns)
            if self._global_grid.header.frame_id != "map":
                return False
        except (TransformException, ValueError, TypeError):
            return False
        if (self._goal_state == "idle"
                and (self._nav2_goal_offset is not None or self._memory_goal is not None)):
            if (self._active_actions.get("follow_path")
                    and not (self._memory_goal is not None and self._nav2_goal_offset is None)):
                return False
            return (not (set(self._active_actions.get("navigate_to_pose", {})) - {self._standoff_id})
                    and self._exclusive_nav_stream())
        if (self._mode == NAV2_PENDING and self._goal_state == "sending"
                and self._goal_send_future is not None and self._goal_handle is None):
            return (len(self._active_actions.get("navigate_to_pose", {})) <= 1
                    and self._exclusive_nav_stream())
        return (self._mode == NAV2 and self._goal_state == "active"
                and self._goal_handle is not None
                and not (set(self._active_actions.get("navigate_to_pose", {}))
                         - {bytes(self._goal_handle.goal_id.uuid)})
                and self._exclusive_nav_stream())

    def _safe_stop(self, reason, latch=False):
        self._direct_clear_since_ns = None
        if (not latch and reason in _DATA_WAIT_REASONS
                and self._can_hold_data_wait(self.get_clock().now().nanoseconds, time.monotonic())):
            # Keep the same owned action, but require fresh data and a new twist before motion.
            self._controller.reset("data_wait")
            self._mode_reason = reason
            self._nav2_latest_command = self._nav2_latest_command_wall = None
            self._publish_selected_command(Twist(), "stop")
            return
        if reason not in _DATA_WAIT_REASONS:
            self._pending_target_point = None
        if self._memory_goal is not None:
            latch = True
        self._last_seen = self._memory_goal = None
        self._nav2_goal_offset = None
        self._nav2_goal_target_map = None
        next_mode = (NAV2_PENDING if not latch and reason in _DATA_WAIT_REASONS
                     and self._mode == NAV2_PENDING and not self._goal_unresolved
                     else SAFE_STOP)
        changed = self._mode != next_mode or self._mode_reason != reason
        self._mode = next_mode
        self._mode_reason = reason
        if latch:
            self._pending_target_point = None
            self._fault_latched = True
        self._nav2_latest_command = None
        if self._goal_unresolved:
            self._request_goal_cancel(reason)
        if changed:
            self._controller.reset("safe_stop")
        self._publish_selected_command(Twist(), "stop")
        if changed:
            self.get_logger().warning(f"FOLLOW_NAV2 {self._mode} reason={reason}")

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
        if self._global_grid.header.frame_id != "map":
            self._safe_stop("occupied_or_unknown_standoff_goal")
            return
        global_stamp = self._stamp_ns(self._global_grid)
        if self._last_blocked_stamp_ns is None or global_stamp < self._last_blocked_stamp_ns:
            return
        if not self._nav2_action.server_is_ready():
            return
        xy = (float(target_map.point.x), float(target_map.point.y))
        self._remember_target(xy, self._stamp_ns(self._latest_target_point))
        if self._nav2_goal_target_map is not None and not self._goal_refresh_pending:
            return  # A completed blocked goal waits for accumulated target movement.
        selected = clear_stand_off_goal(
            self._global_grid, goal_values, *xy, _robot_yaw,
            approach_from=(robot_x, robot_y))
        if selected is None:
            self._safe_stop("no_free_follow_approach", latch=True)
            return
        self._nav2_goal_offset = (selected[0]-xy[0], selected[1]-xy[1])
        self._send_nav_goal(selected, xy, now)

    def _remember_target(self, xy, stamp):
        if self._last_seen is None or stamp > self._last_seen[2]:
            self._last_seen = (*xy, stamp, self._latest_target_received_monotonic)

    def _send_nav_goal(self, pose, target_xy, now):
        goal_x, goal_y, goal_yaw = pose
        goal = NavigateToPose.Goal()
        goal.behavior_tree = os.path.join(
            get_package_share_directory("piracer_navigation"),
            "behavior_trees", "follow_navigate_to_pose.xml",
        )
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = now.to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(goal_yaw / 2.0)
        self._sent_goal_stamp_ns = now.nanoseconds
        self._detour_forwarded = False
        self._goal_state = "sending"
        self._goal_unresolved = True
        self._goal_cancel_reason = None
        self._goal_finish_action = None
        self._goal_refresh_pending = False
        self._nav2_goal_started_wall = time.monotonic()
        self._nav2_goal_target_map = target_xy
        self._nav2_has_moved = False
        self._nav2_latest_command = None
        self._nav2_latest_command_wall = None
        self._mode = NAV2_PENDING
        self._mode_reason = "goal_request_pending"
        try:
            self._goal_send_future = self._nav2_action.send_goal_async(goal)
            self._goal_send_future.add_done_callback(self._on_nav2_goal_response)
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
        self._goal_send_future = None
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
        cancel_pending = not self._enabled or self._goal_cancel_reason is not None
        self._mode = SAFE_STOP if cancel_pending else NAV2
        self._mode_reason = "cancel_on_accept" if cancel_pending else "goal_accepted"
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_nav2_goal_result)
        if set(self._active_actions.get("navigate_to_pose", {})) - {bytes(handle.goal_id.uuid)}:
            self._safe_stop("foreign_nav2_goal", latch=True)
            return
        # A disable or safety stop can arrive while goal acceptance was pending.
        if cancel_pending:
            self._request_goal_cancel(self._goal_cancel_reason or "disabled")

    def _on_nav2_goal_result(self, future):
        self._publish_selected_command(Twist(), "stop")
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
        if (not self._enabled or self._fault_latched
                or status == GoalStatus.STATUS_ABORTED):
            self._memory_goal = self._last_seen = None
            self._fault_latched = self._fault_latched or status == GoalStatus.STATUS_ABORTED
            self._goal_finish_action = None
            self._mode = SAFE_STOP
        elif (self._goal_finish_action == "recover"
                and status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED)):
            self._memory_goal = None
            self._nav2_goal_offset = self._nav2_goal_target_map = None
            self._goal_finish_action = "recover"
        elif (self._memory_goal is not None and self._goal_cancel_reason is None
              and status == GoalStatus.STATUS_SUCCEEDED):
            self.get_logger().info("FOLLOW reached the observation goal after target loss")
            self._memory_goal = self._last_seen = None
            self._nav2_goal_offset = self._nav2_goal_target_map = None
            self._goal_finish_action = "recover"
        elif self._memory_goal is not None:
            self._memory_goal = self._last_seen = None
            self._fault_latched = True
            self._goal_finish_action = None
        elif status == GoalStatus.STATUS_CANCELED and self._goal_cancel_reason == "target_moved":
            self._goal_finish_action = "refresh"
        elif status == GoalStatus.STATUS_CANCELED and self._goal_cancel_reason == "target_distance_reached":
            self._goal_finish_action = "hold"
        elif status == GoalStatus.STATUS_CANCELED and self._mode == SAFE_STOP and not self._fault_latched:
            self._goal_finish_action = "recover"
        elif status == GoalStatus.STATUS_SUCCEEDED:
            self._standoff_id = bytes(self._goal_handle.goal_id.uuid)
            self._goal_finish_action = "hold"
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
        settle_s = 0.3
        if wall_elapsed < settle_s or ros_elapsed < settle_s:
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
        self._goal_send_future = None
        self._goal_cancel_reason = None
        self._goal_terminal_wall = None
        self._goal_terminal_ros_ns = None
        self._goal_terminal_status = None
        self._goal_finish_action = None
        if finish_action == "hold" and self._enabled and not self._fault_latched:
            self._mode = NAV2_PENDING
            self._mode_reason = "nav2_goal_reached_waiting_target"
        elif finish_action == "refresh" and self._enabled and not self._fault_latched:
            self._mode = NAV2_PENDING
            self._mode_reason = "bounded_target_refresh"
            self._goal_refresh_pending = True
        elif finish_action == "recover" and self._enabled and not self._fault_latched:
            self._mode = NAV2_PENDING
            self._mode_reason = "nav2_cancel_drained_data_recovery"
        else:
            self._mode = SAFE_STOP
            self._mode_reason = "goal_finished_without_clear_path"

    def _maybe_refresh_goal(self, target_map_xy, monotonic_now):
        if (
            self._goal_state not in ("active", "canceling", "idle")
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
            if self._goal_unresolved:
                self._request_goal_cancel("target_moved", "refresh")
            self._mode = NAV2_PENDING
            self._mode_reason = "bounded_target_refresh_cancel"

    def _integrated_cycle(self, command, base_target, now_ns, monotonic_now, clock_stalled=False):
        """Select one direct/Nav2/zero command on the existing steady timer."""
        self._finish_goal_drain(now_ns, monotonic_now)
        if now_ns <= 0:
            self._safe_stop("ros_time_invalid")
            return
        if self._last_integrated_ros_ns is not None and now_ns < self._last_integrated_ros_ns:
            self._safe_stop("ros_time_regression")
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
        if self._goal_unresolved and self._goal_state in ("canceling", "draining", "unknown"):
            self._publish_selected_command(Twist(), "stop")
            return

        if clock_stalled:
            self._safe_stop("ros_clock_stalled")
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
            # Ordinary data recovery still checks the full target corridor below.
            self._mode = DIRECT
            self._mode_reason = "fresh_inputs_recovered"
        if self._memory_goal is not None:
            self._try_last_seen(now_ns, monotonic_now, clock_stalled)
            return
        target_x = float(base_target.point.x)
        target_y = float(base_target.point.y)
        minimum_distance = float(self._controller.min_distance_m)
        if target_x <= 0.0 or math.hypot(target_x, target_y) <= minimum_distance:
            self._safe_stop("target_inside_min_distance")
            return
        try:
            grid_pose = self._tf_pose(self._local_grid.header.frame_id, now_ns)
            map_x, map_y, _ = self._tf_pose("map", now_ns)
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
        if self._mode in (NAV2, NAV2_PENDING):
            self._remember_target(target_map_xy, self._stamp_ns(self._latest_target_point))
        if command.linear_x <= 0:
            self._direct_clear_since_ns = None
            self._goal_refresh_pending = False
            if self._goal_unresolved:
                self._request_goal_cancel("target_distance_reached", "hold")
            self._publish_selected_command(Twist(), "stop")
            return

        remaining = max(0.0, target_x ** 2 + target_y ** 2) ** 0.5
        remaining = max(0.0, remaining - self._controller.target_distance_m)
        horizon = min(0.6, remaining)
        direct_clear, collision_distance = swept_footprint_clear(
            self._local_grid,
            grid_pose[0], grid_pose[1], local_yaw,
            float(command.linear_x), float(command.angular_z), horizon,
        )
        if self._global_grid.header.frame_id != "map":
            self._safe_stop("global_costmap_frame_invalid")
            return
        dx, dy = target_map_xy[0] - map_x, target_map_xy[1] - map_y
        # Check the corridor toward the suitcase, independently of robot heading.
        target_clear, _ = swept_footprint_clear(
            self._global_grid, map_x, map_y, math.atan2(dy, dx),
            float(command.linear_x), 0.0,
            max(0.0, math.hypot(dx, dy) - self._controller.target_distance_m),
        )
        # Obstacles beyond the intended stand-off must not trigger another goal.
        obstacle_ahead = not scan_clears_motion(
            self._latest_scan, (0., 0., 0.), 0., distance=min(2., remaining))
        danger = obstacle_ahead or not direct_clear or not target_clear
        if danger:
            self._direct_clear_since_ns = None
        local_stamp_ns = self._stamp_ns(self._local_grid)
        if local_stamp_ns <= 0:
            self._safe_stop("invalid_local_costmap_stamp")
            return
        if (not self._goal_unresolved and not direct_clear and
                collision_distance <= 0.10):
            self._safe_stop("imminent_swept_footprint_collision")
            return

        if self._mode == DIRECT:
            if danger:
                if self._last_blocked_stamp_ns is None:
                    self._last_blocked_stamp_ns = local_stamp_ns
                self._mode = NAV2_PENDING
                self._mode_reason = "obstacle_within_2m" if obstacle_ahead else "direct_corridor_blocked"
                self.get_logger().warning(
                    f"FOLLOW_NAV2 transition DIRECT->NAV2_PENDING reason={self._mode_reason} stamp_ns={local_stamp_ns}"
                )
                self._publish_selected_command(Twist(), "stop")
                self._start_nav2_goal()
            else:
                direct = Twist()
                direct.linear.x = command.linear_x
                direct.angular.z = command.angular_z
                self._publish_selected_command(direct, "direct")
            return

        if self._mode in (NAV2_PENDING, NAV2):
            if not danger:
                stamp = self._stamp_ns(self._latest_target_point)
                if self._direct_clear_since_ns is None:
                    self._direct_clear_since_ns = stamp
                if stamp - self._direct_clear_since_ns < _DIRECT_CLEAR_SPAN_NS:
                    if self._mode == NAV2 and self._goal_state == "active":
                        self._forward_nav_command(grid_pose, monotonic_now, map_xy=(map_x, map_y))
                    else:
                        self._publish_selected_command(Twist(), "stop")
                    return
                self._goal_refresh_pending = False
                if self._goal_unresolved:
                    self._mode = NAV2_PENDING
                    self._mode_reason = "direct_path_clear_cancel"
                    self._publish_selected_command(Twist(), "stop")
                    self._request_goal_cancel("direct_path_clear", "recover")
                    return
                self._last_blocked_stamp_ns = None
                self._nav2_goal_offset = None
                self._nav2_goal_target_map = None
                self._mode = DIRECT
                self._mode_reason = "direct_path_clear_without_active_goal"
                self._publish_selected_command(Twist(), "stop")
                return
            if self._last_blocked_stamp_ns is None:
                self._last_blocked_stamp_ns = local_stamp_ns
            self._maybe_refresh_goal(target_map_xy, monotonic_now)
            if self._goal_refresh_pending and not self._goal_unresolved:
                self._start_nav2_goal()
                self._publish_selected_command(Twist(), "stop")
                return
            if self._mode == NAV2_PENDING:
                self._publish_selected_command(Twist(), "stop")
                self._start_nav2_goal()
                return

            self._forward_nav_command(grid_pose, monotonic_now, map_xy=(map_x, map_y))
            return

        self._safe_stop("unknown_follow_mode")

    def _on_action_status(self, name, message):
        previous = self._active_actions.get(name)
        self._active_actions[name] = {
            bytes(item.goal_info.goal_id.uuid): (
                item.status, Time.from_msg(item.goal_info.stamp).nanoseconds,
            ) for item in message.status_list
            if item.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                               GoalStatus.STATUS_CANCELING)
        }
        if name == "follow_path" and previous != self._active_actions[name]:
            self._nav2_latest_command = self._nav2_latest_command_wall = None

    def _memory_command_owned(self):
        nav = self._active_actions.get("navigate_to_pose", {})
        paths = self._active_actions.get("follow_path", {})
        if self._goal_handle is None or len(paths) != 1:
            return False
        our_id = bytes(self._goal_handle.goal_id.uuid)
        path_status, path_stamp = next(iter(paths.values()))
        return (set(nav) == {our_id} and nav[our_id][0] == GoalStatus.STATUS_EXECUTING
                and path_status == GoalStatus.STATUS_EXECUTING
                and path_stamp >= self._sent_goal_stamp_ns)

    def _exclusive_nav_stream(self):
        for topic, publisher in (("/cmd_vel", "velocity_smoother"),
                                 ("/cmd_vel_nav", "controller_server"),
                                 ("/cmd_vel_follow", "piracer_follow")):
            sources = self.get_publishers_info_by_topic(topic)
            if [(p.node_namespace, p.node_name) for p in sources] != [("/", publisher)]:
                return False
        # ACQUIRE keeps its feedback subscription while disabled. The coordinator
        # confirms its disable before FOLLOW; active-goal checks remain separate.
        clients = self.get_subscriptions_info_by_topic("/piracer/follow_nav/follow_path/_action/feedback")
        allowed = {"bt_navigator_navigate_to_pose_rclcpp_node",
                   "bt_navigator_navigate_through_poses_rclcpp_node", "piracer_acquire"}
        return (bool(clients) and all(c.node_namespace == "/" and c.node_name in allowed
                                      for c in clients)
                and not self._active_actions.get("navigate_through_poses"))

    def _try_last_seen(self, now_ns, wall, clock_stalled):
        """Finish the committed observation goal while the target is absent."""
        self._direct_clear_since_ns = None
        if (not self._follow_nav2_avoidance or not self._publish_cmd_vel
                or not self._enabled or self._fault_latched or self._last_seen is None):
            return False
        starting = self._memory_goal is None
        if starting and (self._mode != NAV2 or self._goal_state != "active"
                         or self._goal_handle is None or self._goal_cancel_reason is not None):
            return False
        stamp, receipt = self._last_seen[2:]
        max_age = float(self.get_parameter("max_target_age_s").value)
        point_fresh = not starting and stamped_input_is_fresh(stamp, now_ns, wall-receipt, max_age, 0.0)
        detection = self._latest_detections
        detection_fresh = (detection is not None and self._detection_receipt is not None
            and stamp < self._stamp_ns(detection) <= now_ns
            and stamped_input_is_fresh(self._stamp_ns(detection), now_ns,
                                      wall-self._detection_receipt, .5))
        if (now_ns <= 0 or (self._last_integrated_ros_ns is not None and now_ns < self._last_integrated_ros_ns)
                or not 0 <= now_ns-stamp <= _LAST_SEEN_BUDGET_S*1_000_000_000
                or not 0 <= wall-receipt <= _LAST_SEEN_BUDGET_S
                or self._stamp_ns(self._latest_target_point) != stamp
                or (detection is not None and self._stamp_ns(detection) > now_ns)):
            self._safe_stop("last_seen_expired_or_unconfirmed")
            return True
        if not starting and self._goal_state in ("canceling", "draining", "unknown"):
            self._publish_selected_command(Twist(), "stop")
            return True
        if clock_stalled or not (point_fresh or detection_fresh):
            self._safe_stop("ros_clock_stalled" if clock_stalled else "awaiting_target_confirmation")
            return True
        self._last_integrated_ros_ns = now_ns
        fresh, reason = self._integrated_inputs_fresh(now_ns, wall)
        if not fresh:
            self._safe_stop(reason)
            return True
        try:
            map_x, map_y, _ = self._tf_pose("map", now_ns)
            local = self._tf_pose(self._local_grid.header.frame_id, now_ns)
            if self._global_grid.header.frame_id != "map":
                raise ValueError("last-seen map frame")
        except (TransformException, ValueError, TypeError):
            self._safe_stop("last_seen_localization_invalid")
            return True
        if not self._exclusive_nav_stream():
            self._safe_stop("last_seen_foreign_command_source")
            return True
        if not self._memory_command_owned() or starting and not self._detour_forwarded:
            self._safe_stop("awaiting_target_confirmation")
            return True
        if starting:
            goal_x, goal_y = self._nav2_goal_target_map
            offset_x, offset_y = self._nav2_goal_offset or (0.0, 0.0)
            goal_x, goal_y = goal_x + offset_x, goal_y + offset_y
            self._memory_goal = (goal_x, goal_y, math.atan2(goal_y - map_y, goal_x - map_x))
            self._mode_reason = "continuing_committed_nav2_goal"
        self._forward_nav_command(local, wall, blocked_cost=99, map_xy=(map_x, map_y))
        return True

    def _forward_nav_command(self, grid_pose, monotonic_now, blocked_cost=100, map_xy=None):
        if not self._memory_command_owned():
            self._nav2_latest_command = None
            self._publish_selected_command(Twist(), "stop")
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
            linear, angular, nav_age, self._nav2_has_moved, self._controller.max_speed_mps,
        )
        if policy == "fault":
            self._safe_stop("invalid_or_stale_nav2_twist")
            return
        if policy == "wait":
            self._publish_selected_command(Twist(), "stop")
            return
        # Nav2 checks its own arc. Recheck only when steering limiting changes it.
        if not math.isclose(bounded[1] / bounded[0], angular / linear,
                            rel_tol=1e-9, abs_tol=1e-9):
            nav_clear, nav_collision_distance = swept_footprint_clear(
                self._local_grid,
                *grid_pose,
                bounded[0], bounded[1], 0.10, blocked_cost=blocked_cost,
            )
            if not nav_clear or nav_collision_distance <= 0.10:
                self._safe_stop("nav2_imminent_footprint_collision")
                return
        if self._memory_goal is not None:
            scan = self._latest_scan
            try:
                tf = self._tf_buffer.lookup_transform(
                    "base_link", scan.header.frame_id, Time.from_msg(scan.header.stamp))
                pose = self._planar_pose(tf.transform)
                clear = scan_clears_motion(scan, pose, bounded[1] / bounded[0])
            except (TransformException, ValueError, TypeError):
                clear = False
            if not clear:
                self._safe_stop("last_seen_scan_not_clear")
                return
        else:
            self._detour_forwarded = True
        if self._mode_reason in _DATA_WAIT_REASONS:
            self._mode_reason = "fresh_inputs_recovered"
        if map_xy is not None and self._nav2_goal_target_map is not None:
            goal_x, goal_y = self._nav2_goal_target_map
            if self._nav2_goal_offset is not None:
                goal_x += self._nav2_goal_offset[0]
                goal_y += self._nav2_goal_offset[1]
            yaw = math.atan2(goal_y - map_xy[1], goal_x - map_xy[0])
            if footprint_clear(self._global_grid, goal_x, goal_y, yaw):
                # Keep the committed position; let replanning use the current approach heading.
                update = PoseStamped()
                update.header.frame_id = "map"
                update.header.stamp = self.get_clock().now().to_msg()
                update.pose.position.x, update.pose.position.y = goal_x, goal_y
                update.pose.orientation.z = math.sin(yaw / 2.0)
                update.pose.orientation.w = math.cos(yaw / 2.0)
                self._nav2_goal_update_publisher.publish(update)
        selected = Twist()
        selected.linear.x, selected.angular.z = bounded
        self._publish_selected_command(selected, "nav2")

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
            self._pending_target_point = None
            self._target_input_stamp_ns = None
            self._last_ros_time_ns = now_ns
            self._last_ros_time_advance_monotonic = monotonic_now
            self._controller.reset("ros_clock_regression")
            if self._last_seen is not None:
                self._safe_stop("ros_clock_regression")
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
            "last_seen",
        )
        completed = sum(self._cycle_counts[name] for name in outcome_names)
        target = self._latest_target_point
        target_stamp_ns = self._stamp_ns(target) if target is not None else None
        target_received = self._latest_target_received_monotonic
        summary = {
            "schema_version": 1,
            "process_id": os.getpid(),
            "sim_time_ns": int(ros_time_ns),
            "cycle_outcome": outcome,
            "target_stamp_ns": target_stamp_ns,
            "target_ros_age_s": ((ros_time_ns - target_stamp_ns) / 1e9
                                 if target_stamp_ns is not None else None),
            "target_received_age_s": (monotonic_now - target_received
                                      if target_received is not None else None),
            "cycle_counts": dict(self._cycle_counts),
            "mode_reason": self._mode_reason,
            "direct_clear_since_stamp_ns": getattr(self, "_direct_clear_since_ns", None),
            "input_received": {
                "local_costmap": getattr(self, "_local_grid", None) is not None,
                "global_costmap": getattr(self, "_global_grid", None) is not None,
                "odom_info": getattr(self, "_odom_info", None) is not None,
                "scan": getattr(self, "_latest_scan", None) is not None,
            },
            "last_seen_goal_xy": self._memory_goal[:2] if self._memory_goal else None,
            "unclassified_cycles": self._cycle_counts["timer_calls"] - completed,
        }
        message = String()
        message.data = json.dumps(summary, separators=(",", ":"), allow_nan=False)
        self._control_diagnostics_publisher.publish(message)
        self._last_diagnostics_monotonic = monotonic_now

    def _control_tick(self):
        self._control_cycle()
        self._status_publisher.publish(String(data=self._behavior_status()))

    def _behavior_status(self):
        if not self._enabled:
            return "DISABLED"
        if self._fault_latched:
            return "FAULT"
        if self._follow_nav2_avoidance and (self._goal_unresolved or self._memory_goal is not None):
            return "AVOIDING"
        return "DIRECT"

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

        if self._follow_nav2_avoidance and self._retry_target_point():
            self._record_cycle("tf_error", ros_time_stalled, now_ns, monotonic_now)
            return

        if self._latest_target_point is None:
            self._stop_for_input_error("no_target")
            self._record_cycle("no_target", ros_time_stalled, now_ns, monotonic_now)
            return

        received_at = self._latest_target_received_monotonic
        if received_at is None or time.monotonic() - received_at > max_age_s:
            if not self._try_last_seen(now_ns, monotonic_now, ros_time_stalled):
                self._stop_for_input_error("stale_received_target")
            self._record_cycle("last_seen" if self._memory_goal else "stale_received_target",
                               ros_time_stalled, now_ns, monotonic_now)
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
            if age_s < 0.0 or not self._try_last_seen(now_ns, monotonic_now, ros_time_stalled):
                self._stop_for_input_error("future_target_stamp" if age_s < 0.0 else "stale_ros_stamp")
            self._record_cycle("last_seen" if self._memory_goal else "stale_ros_stamp",
                               ros_time_stalled, now_ns, monotonic_now)
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
            if (self._controller.direction_kd > 0.0 or self._controller.direction_ki > 0.0
                    or self._controller.distance_ki > 0.0 or self._controller.distance_kd > 0.0):
                target_stamp_ns = (
                    int(self._latest_target_point.header.stamp.sec) * 1_000_000_000
                    + int(self._latest_target_point.header.stamp.nanosec)
                )
                command = self._controller.update(
                    x_m, y_m, measurement_stamp_ns=target_stamp_ns
                )
            else:
                command = self._controller.update(x_m, y_m)
        except (ValueError, TypeError) as exc:
            self.get_logger().debug(f"Follow stopped for invalid target or TF: {exc}")
            self._stop_for_input_error("invalid_target")
            self._record_cycle("invalid_target", ros_time_stalled, now_ns, monotonic_now)
            return

        if self._follow_nav2_avoidance:
            if self._fault_latched:
                self._controller.reset("safe_stop")
            self._integrated_cycle(command, target, now.nanoseconds, monotonic_now, ros_time_stalled)
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
            "direct_clear_since_stamp_ns": getattr(self, "_direct_clear_since_ns", None),
        }
        message = String()
        message.data = json.dumps(sample, separators=(",", ":"), allow_nan=False)
        self._control_sample_publisher.publish(message)

    def stop_and_destroy(self):
        try:
            self._controller.reset("shutdown")
            if rclpy.ok(context=self.context):
                if self._follow_nav2_avoidance and self._goal_unresolved:
                    self._request_goal_cancel("shutdown")
                if self._enabled:
                    self._publish_zero()
        finally:
            self._enabled = False
            self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiracerFollowNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.stop_and_destroy()
        finally:
            try_shutdown()


if __name__ == "__main__":
    main()
