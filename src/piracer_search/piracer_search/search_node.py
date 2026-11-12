"""Search lifecycle, shared frontier evaluation, and Nav2 goal handoff."""

from collections import deque
import json
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from rclpy.utilities import try_shutdown
from rtabmap_msgs.msg import OdomInfo
from std_srvs.srv import SetBool
from std_msgs.msg import String
from tf2_ros import Buffer
from visualization_msgs.msg import MarkerArray

from .search_markers import _candidate_markers, _stale_marker_deletions
from .tf_timing import SearchTfReceiver


def _map_subscription_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _odom_info_subscription_qos():
    return qos_profile_sensor_data


# Conservative default for direct and physical launches. The full simulation
# sets its own value through the existing launch chain.
DEFAULT_TF_FUTURE_LEAD_NS = 100_000_000


class SearchNode(Node):
    """Evaluate frontiers and track every Search-owned Nav2 goal to settlement."""

    def __init__(self):
        super().__init__("piracer_search")
        self.declare_parameter("map_topic", "/nav2/map")
        self.declare_parameter("start_stop_service", "/search/enable")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("behavior_tree", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("marker_topic", "/piracer/frontier_markers")
        self.declare_parameter("odom_info_topic", "/visual_vo/odom_info_lite")
        self.declare_parameter("max_localization_age_sec", 0.5)
        self.declare_parameter(
            "tf_future_lead_sec", DEFAULT_TF_FUTURE_LEAD_NS / 1_000_000_000
        )
        tf_future_lead_sec = float(self.get_parameter("tf_future_lead_sec").value)
        if not math.isfinite(tf_future_lead_sec) or tf_future_lead_sec < 0.0:
            raise ValueError("tf_future_lead_sec must be finite and non-negative")
        self._tf_future_lead_ns = round(tf_future_lead_sec * 1_000_000_000)
        self.declare_parameter("progress_timeout", 30.0)
        progress_timeout_sec = float(self.get_parameter("progress_timeout").value)
        if not math.isfinite(progress_timeout_sec) or progress_timeout_sec <= 0.0:
            raise ValueError("progress_timeout must be finite and positive")
        self._progress_timeout_ns = round(progress_timeout_sec * 1_000_000_000)

        self._enabled = False
        self._search_epoch = 0
        self._localization_generation = 0
        self._latest_frontier_result = None
        self._map = None
        self._goal_sequence = 0
        self._owned_goals = {}
        self._current_goal_token = None
        self._active_goal_token = None
        self._goal_active = False
        self._blacklisted_frontiers = []
        self._progress_goal_token = None
        self._progress_min_distance = None
        self._progress_started_ns = None
        self._published_marker_keys = set()
        self._odom_info = deque(maxlen=10)
        self._last_processed_odom_info_sequence = 0
        self._localization_available = False
        self._localization_invalidated_at_ns = None
        self._last_accepted_odom_stamp_ns = None
        self._last_accepted_full_tf_stamp_ns = None
        self._last_ros_time_ns = None
        self._clock_epoch = 0
        self._localization_reason = None

        self._tf_buffer = Buffer()
        self._search_callback_group = MutuallyExclusiveCallbackGroup()
        self._odom_info_wakeup = self.create_guard_condition(
            self._drain_odom_info_snapshot,
            callback_group=self._search_callback_group,
        )
        self._action = ActionClient(
            self, NavigateToPose, self.get_parameter("navigate_to_pose_action").value,
            callback_group=self._search_callback_group,
        )
        self._markers = self.create_publisher(
            MarkerArray, self.get_parameter("marker_topic").value, 10
        )
        self._map_subscription = self.create_subscription(
            OccupancyGrid,
            self.get_parameter("map_topic").value,
            self._map_callback,
            _map_subscription_qos(),
            callback_group=self._search_callback_group,
        )
        self._frontier_pose = self.create_publisher(
            PoseStamped, "/piracer/search/robot_pose", 10
        )
        self._frontier_results = self.create_subscription(
            String, "/piracer/search/frontiers", self._frontier_result_callback, 10,
            callback_group=self._search_callback_group,
        )
        self._start_stop = self.create_service(
            SetBool, self.get_parameter("start_stop_service").value, self._set_search,
            callback_group=self._search_callback_group,
        )
        self._timer = self.create_timer(
            1.0, self._search_tick, callback_group=self._search_callback_group
        )
        self.get_logger().info(
            "Search localization max age: "
            f"{float(self.get_parameter('max_localization_age_sec').value):.3f} s"
        )

        self._tf_receiver = SearchTfReceiver(
            self._tf_buffer, self._receiver_error,
            context=self.context, namespace=self.get_namespace(),
            use_sim_time=bool(self.get_parameter("use_sim_time").value),
            odom_info_wakeup=self._odom_info_wakeup,
        )
        self._tf_listener = self._tf_receiver.listener
        # Receive large OdomInfo messages outside both TF and Search decisions.
        self._odom_info_callback_group = MutuallyExclusiveCallbackGroup()
        try:
            self._odom_info_subscription = self.create_subscription(
                OdomInfo, self.get_parameter("odom_info_topic").value,
                self._receive_odom_info, _odom_info_subscription_qos(),
                callback_group=self._odom_info_callback_group,
            )
        except BaseException:
            self._tf_receiver.close()
            raise

    def _receive_odom_info(self, message):
        """Receive metadata only; validation stays in the Search callback group."""
        self._tf_receiver.receive_odom_info(message)

    def _map_callback(self, message):
        self._map = message

    def _frontier_result_callback(self, message):
        try:
            result = json.loads(message.data)
            if (result.get("frame") != (self._map.header.frame_id if self._map else None)
                    or not isinstance(result.get("candidates"), list)
                    or not isinstance(result.get("ranking"), list)):
                return
            result["_received_monotonic_ns"] = time.monotonic_ns()
            self._latest_frontier_result = result
        except (TypeError, ValueError):
            self.get_logger().warning("Ignoring invalid frontier-core result")

    @staticmethod
    def _stamp_ns(message):
        if hasattr(message, "stamp_ns"):
            return int(message.stamp_ns)
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _drain_odom_info_snapshot(self):
        """Apply small receiver snapshots on SearchNode's existing callback group."""
        receiver = getattr(self, "_tf_receiver", None)
        if receiver is None or not hasattr(receiver, "odom_info_snapshot"):
            return
        self._current_ros_time_ns()
        clock_epoch = self._clock_epoch
        samples, pending_loss, latest_sequence = receiver.odom_info_snapshot()
        events = [
            sample for sample in samples
            if sample.sequence > self._last_processed_odom_info_sequence
        ]
        if (pending_loss is not None
                and pending_loss.sequence > self._last_processed_odom_info_sequence):
            events = [
                sample for sample in events
                if sample.sequence > pending_loss.sequence
            ]
            events.append(pending_loss)
        events.sort(key=lambda sample: sample.sequence)
        for sample in events:
            if self._clock_epoch != clock_epoch:
                return
            self._odom_info_callback(sample)
            # The callback may observe a clock reset. Do not replay the
            # rest of a snapshot captured in the previous clock epoch.
            if self._clock_epoch != clock_epoch:
                return
            self._last_processed_odom_info_sequence = sample.sequence
            if sample.lost:
                receiver.acknowledge_odom_info_loss(sample.sequence)
        self._last_processed_odom_info_sequence = max(
            self._last_processed_odom_info_sequence, latest_sequence
        )

    def _current_ros_time_ns(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._last_ros_time_ns is not None and now_ns < self._last_ros_time_ns:
            self._handle_clock_regression(now_ns)
        self._last_ros_time_ns = now_ns
        return now_ns

    def _mark_owned_goal_localization_invalidated(self, reason):
        for goal in self._owned_goals.values():
            goal["localization_invalidated"] = True
            if goal["cancel_reason"] is None:
                goal["cancel_reason"] = reason
            self._cancel_goal(goal, reason)

    def _handle_clock_regression(self, now_ns):
        self._clock_epoch += 1
        self._localization_generation += 1
        was_available = self._localization_available
        self._odom_info.clear()
        receiver = getattr(self, "_tf_receiver", None)
        if receiver is not None and hasattr(receiver, "reset_odom_info"):
            self._last_processed_odom_info_sequence = receiver.reset_odom_info()
        self._tf_buffer.clear()
        self._localization_available = False
        self._localization_invalidated_at_ns = None
        self._last_accepted_odom_stamp_ns = None
        self._last_accepted_full_tf_stamp_ns = None
        self._mark_owned_goal_localization_invalidated("clock_regression")
        if was_available or self._localization_reason != "ROS clock moved backwards":
            self.get_logger().warning("Search localization invalidated: ROS clock moved backwards")
        self._localization_reason = "ROS clock moved backwards"

    def _invalidate_localization(
        self, reason, lost_stamp_ns=None, diagnostic_details=None,
        lost_stamp_floor_ns=None,
        after_odom_info_sequence=None,
    ):
        was_available = self._localization_available
        clock_epoch = self._clock_epoch
        now_ns = self._last_ros_time_ns
        if was_available:
            now_ns = self._current_ros_time_ns()
            if self._clock_epoch != clock_epoch:
                # Clock reset already cleared localization and handoff state.
                # Do not apply barriers computed from the previous time epoch.
                return

        boundaries = [self._localization_invalidated_at_ns, self._last_accepted_odom_stamp_ns]
        if lost_stamp_ns is not None and lost_stamp_ns > 0:
            boundaries.append(lost_stamp_ns)
        if lost_stamp_floor_ns is not None and lost_stamp_floor_ns > 0:
            boundaries.append(lost_stamp_floor_ns)
        if was_available:
            boundaries.append(now_ns)
        self._localization_generation += 1
        self._localization_invalidated_at_ns = max(
            (value for value in boundaries if value is not None), default=None
        )
        self._localization_available = False
        self._odom_info.clear()
        receiver = getattr(self, "_tf_receiver", None)
        if receiver is not None and hasattr(receiver, "invalidate_odom_info"):
            receiver.invalidate_odom_info(
                self._localization_invalidated_at_ns,
                after_sequence=(
                    self._last_processed_odom_info_sequence
                    if after_odom_info_sequence is None
                    else after_odom_info_sequence
                ),
            )
        self._mark_owned_goal_localization_invalidated("localization_invalidated")
        if was_available or self._localization_reason != reason:
            message = f"Search localization invalidated: {reason}"
            if diagnostic_details is not None:
                message += f"; {diagnostic_details}"
            self.get_logger().warning(message)
        self._localization_reason = reason

    def _odom_info_callback(self, message):
        clock_epoch = self._clock_epoch
        now_ns = self._current_ros_time_ns()
        if self._clock_epoch != clock_epoch:
            return
        stamp_ns = self._stamp_ns(message)
        if message.lost:
            lost_stamp_ns = stamp_ns
            lost_stamp_floor_ns = getattr(message, "lost_stamp_floor_ns", None)
            if self._clock_epoch > 0:
                # After a clock reset, future loss stamps may belong to an old
                # epoch. Keep the loss signal without installing its barrier.
                if lost_stamp_ns > now_ns:
                    lost_stamp_ns = None
                if lost_stamp_floor_ns is not None and lost_stamp_floor_ns > now_ns:
                    lost_stamp_floor_ns = None
            self._invalidate_localization(
                "OdomInfo reports lost", lost_stamp_ns,
                lost_stamp_floor_ns=lost_stamp_floor_ns,
                after_odom_info_sequence=getattr(message, "sequence", None),
            )
            return
        if stamp_ns == 0:
            return
        if (self._localization_invalidated_at_ns is not None
                and stamp_ns <= self._localization_invalidated_at_ns):
            return
        self._odom_info.append(message)

    def _receiver_error(self, operation, exc):
        self.get_logger().warning(
            "Search TF/OdomInfo receiver error: "
            f"operation={operation!r}, exception={type(exc).__name__}: {exc}"
        )

    def _localization_pose(self, map_frame):
        now_ns = self._current_ros_time_ns()
        max_age_ns = round(
            float(self.get_parameter("max_localization_age_sec").value)
            * 1_000_000_000
        )
        base_frame = self.get_parameter("base_frame").value
        eligible = [
            message for message in self._odom_info
            if 0 < self._stamp_ns(message) <= now_ns
            and (self._localization_invalidated_at_ns is None
                 or self._stamp_ns(message) > self._localization_invalidated_at_ns)
            and (self._last_accepted_odom_stamp_ns is None
                 or self._stamp_ns(message) >= self._last_accepted_odom_stamp_ns)
        ]
        if not eligible:
            queued = [
                message for message in self._odom_info
                if self._stamp_ns(message) > 0
            ]
            selected_message = max(
                queued, key=self._stamp_ns, default=None
            )
            selected_stamp_ns = (
                self._stamp_ns(selected_message)
                if selected_message is not None else None
            )
            info_age_ns = (
                now_ns - selected_stamp_ns
                if selected_stamp_ns is not None else None
            )
            if selected_stamp_ns is None:
                rejection_reason = "no positive OdomInfo sample"
            elif selected_stamp_ns > now_ns:
                rejection_reason = "OdomInfo stamp is in the future"
            elif (self._localization_invalidated_at_ns is not None
                  and selected_stamp_ns <= self._localization_invalidated_at_ns):
                rejection_reason = "OdomInfo does not exceed localization barrier"
            elif (self._last_accepted_odom_stamp_ns is not None
                  and selected_stamp_ns < self._last_accepted_odom_stamp_ns):
                rejection_reason = "OdomInfo timestamp regressed"
            else:
                rejection_reason = "no eligible OdomInfo sample"
            details = (
                f"selected_info_stamp_ns={selected_stamp_ns}, "
                f"info_age_ns={info_age_ns}, requested_stamp_ns=0, "
                f"requested_time='latest', target_frame={map_frame!r}, "
                f"source_frame={base_frame!r}, actual_tf_stamp_ns=None, "
                f"actual_tf_age_ns=None, validation_ros_now_ns={now_ns}, "
                f"max_age_ns={max_age_ns}, rejection_reason={rejection_reason!r}"
            )
            if self._localization_available:
                self._invalidate_localization(
                    "no current OdomInfo", diagnostic_details=details
                )
            return None

        info_message = max(eligible, key=self._stamp_ns)
        info_stamp_ns = self._stamp_ns(info_message)
        info_age_ns = now_ns - info_stamp_ns
        info_details = (
            f"selected_info_stamp_ns={info_stamp_ns}, info_age_ns={info_age_ns}, "
            f"requested_stamp_ns=0, requested_time='latest', "
            f"target_frame={map_frame!r}, source_frame={base_frame!r}, "
            f"actual_tf_stamp_ns=None, actual_tf_age_ns=None, "
            f"validation_ros_now_ns={now_ns}, max_age_ns={max_age_ns}"
        )
        if info_age_ns > max_age_ns:
            self._invalidate_localization(
                "OdomInfo is older than max_localization_age_sec",
                diagnostic_details=info_details + ", rejection_reason='stale OdomInfo'",
            )
            return None

        # TF2 chooses one common time for the complete map-to-base chain.
        query_time = Time()
        try:
            full_transform = self._tf_buffer.lookup_transform(
                map_frame, base_frame, query_time
            )
        except Exception as exc:
            self._invalidate_localization(
                "full TF unavailable",
                diagnostic_details=(
                    f"{info_details}, rejection_reason='TF lookup failed', "
                    f"exception_class={type(exc).__name__}, "
                    f"exception_message={str(exc)!r}"
                ),
            )
            return None
        full_stamp_ns = self._stamp_ns(full_transform)

        validation_now_ns = self.get_clock().now().nanoseconds
        last_ros_time_ns = self._last_ros_time_ns
        if (validation_now_ns < now_ns
                or (last_ros_time_ns is not None
                    and validation_now_ns < last_ros_time_ns)):
            self._handle_clock_regression(validation_now_ns)
            self._last_ros_time_ns = validation_now_ns
            return None

        self._last_ros_time_ns = validation_now_ns
        current_barrier_ns = self._localization_invalidated_at_ns
        accepted_info_ns = self._last_accepted_odom_stamp_ns
        current_eligible = [
            self._stamp_ns(message) for message in self._odom_info
            if 0 < self._stamp_ns(message) <= validation_now_ns
            and (current_barrier_ns is None
                 or self._stamp_ns(message) > current_barrier_ns)
            and (accepted_info_ns is None
                 or self._stamp_ns(message) >= accepted_info_ns)
        ]

        checks = (
            ("OdomInfo", info_stamp_ns, self._last_accepted_odom_stamp_ns),
            ("full TF", full_stamp_ns, self._last_accepted_full_tf_stamp_ns),
        )
        for role, stamp_ns, accepted_stamp_ns in checks:
            age_ns = validation_now_ns - stamp_ns
            reason = None
            if stamp_ns <= 0:
                reason = f"{role} stamp is zero"
            elif stamp_ns > validation_now_ns + (
                self._tf_future_lead_ns if role == "full TF" else 0
            ):
                reason = f"{role} stamp is in the future"
            elif age_ns > max_age_ns:
                reason = f"{role} is older than max_localization_age_sec"
            elif (current_barrier_ns is not None
                  and stamp_ns <= current_barrier_ns):
                reason = f"{role} does not exceed localization invalidation barrier"
            elif accepted_stamp_ns is not None and stamp_ns < accepted_stamp_ns:
                reason = f"{role} timestamp regressed"
            elif role == "OdomInfo" and info_stamp_ns != max(current_eligible, default=None):
                reason = "selected OdomInfo is no longer newest eligible sample"
            if role == "OdomInfo" and info_stamp_ns not in current_eligible:
                reason = reason or "selected OdomInfo is no longer eligible"
            if reason is not None:
                details = (
                    f"selected_info_stamp_ns={info_stamp_ns}, "
                    f"info_age_ns={validation_now_ns - info_stamp_ns}, "
                    f"requested_stamp_ns=0, requested_time='latest', "
                    f"target_frame={map_frame!r}, source_frame={base_frame!r}, "
                    f"actual_tf_stamp_ns={full_stamp_ns}, "
                    f"actual_tf_age_ns={validation_now_ns - full_stamp_ns}, "
                    f"validation_ros_now_ns={validation_now_ns}, "
                    f"max_age_ns={max_age_ns}, rejection_reason={reason!r}"
                )
                self._invalidate_localization(
                    reason, diagnostic_details=details,
                )
                return None

        translation = full_transform.transform.translation
        rotation = full_transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        was_available = self._localization_available
        self._localization_available = True
        self._localization_invalidated_at_ns = None
        self._last_accepted_odom_stamp_ns = info_stamp_ns
        self._last_accepted_full_tf_stamp_ns = self._stamp_ns(full_transform)
        self._localization_reason = None
        receiver = getattr(self, "_tf_receiver", None)
        if receiver is not None and hasattr(receiver, "set_odom_info_barrier"):
            receiver.set_odom_info_barrier(None)
        if not was_available:
            self.get_logger().info("Search localization is available")
        return translation.x, translation.y, yaw

    def _set_search(self, request, response):
        if request.data:
            if not self._enabled:
                self._search_epoch += 1
                self._blacklisted_frontiers.clear()
                self._reset_progress()
            self._enabled = True
            response.message = "Search enabled"
        else:
            if self._enabled:
                self._search_epoch += 1
            self._enabled = False
            for goal in self._owned_goals.values():
                if goal["cancel_reason"] is None:
                    goal["cancel_reason"] = "search_disabled"
                self._cancel_goal(goal, "search_disabled")
            if self._has_goal_ownership():
                selected = self._selected_for_markers({"candidates": []})
                self._publish_markers({"candidates": []}, selected)
                response.message = "Search disabled; waiting for Search-owned goals to settle"
            else:
                self._publish_markers({"candidates": []}, None)
                response.message = "Search disabled"
        response.success = bool(request.data or not self._has_goal_ownership())
        return response

    def _has_goal_ownership(self):
        return bool(self._owned_goals)

    def _search_tick(self):
        self._drain_odom_info_snapshot()
        if not self._enabled or self._map is None:
            return

        robot_pose = self._localization_pose(self._map.header.frame_id)
        if robot_pose is None:
            return
        self._check_progress_timeout()
        self._publish_frontier_pose(robot_pose)

        result = self._latest_frontier_result
        if result is None:
            return
        if result.get("map_stamp_ns") != self._stamp_ns(self._map):
            return
        if time.monotonic_ns() - result.get("_received_monotonic_ns", 0) > 2_000_000_000:
            return
        result_pose = result.get("robot_xy")
        if (not isinstance(result_pose, list) or len(result_pose) != 2
                or math.dist(result_pose, robot_pose[:2]) > 0.10):
            return
        self._latest_frontier_result = None
        self._apply_search_result(result)

    def _publish_frontier_pose(self, robot_pose):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self._map.header.frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x, pose_msg.pose.position.y = robot_pose[:2]
        pose_msg.pose.orientation.z = math.sin(robot_pose[2] / 2.0)
        pose_msg.pose.orientation.w = math.cos(robot_pose[2] / 2.0)
        self._frontier_pose.publish(pose_msg)

    def _apply_search_result(self, result):
        """Only the Search callback consumes results and changes navigation state."""
        selected = self._selected_for_markers(result)
        self._publish_markers(result, selected)
        if not self._enabled or result["status"] != "RECOMMENDATION_AVAILABLE":
            return
        if any(goal.get("localization_invalidated") for goal in self._owned_goals.values()):
            return
        if not self._action.server_is_ready():
            return

        behavior_tree = self.get_parameter("behavior_tree").value
        if not behavior_tree:
            self.get_logger().warning("Search has no NavigateToPose behavior_tree configured")
            return
        candidate = self._next_ranked_candidate(result)
        if candidate is None:
            return
        current = self._owned_goals.get(self._current_goal_token)
        if (self._goal_active and current is not None
                and self._same_point(current["candidate"], candidate)):
            self._update_progress(current, candidate)
            return
        self._send_goal(candidate, behavior_tree)

    def _selected_for_markers(self, result):
        current = self._latest_owned_goal()
        if current is not None:
            return current["candidate"]
        return self._next_ranked_candidate(result)

    def _publish_markers(self, result, selected):
        if self._map is None:
            return
        candidates = list(result.get("candidates", []))
        frame = self._map.header.frame_id or "map"
        current = self._latest_owned_goal()
        selected_frame = current.get("frame_id", frame) if current is not None else frame
        stamp = self.get_clock().now().to_msg()
        message = MarkerArray()
        message.markers = _candidate_markers(
            candidates, selected, frame, stamp, max_labels=6,
            selected_frame=selected_frame,
        )
        current_keys = {
            (marker.ns, marker.id)
            for marker in message.markers
            if marker.action == marker.ADD
        }
        message.markers.extend(
            _stale_marker_deletions(
                self._published_marker_keys, current_keys, frame, stamp
            )
        )
        self._published_marker_keys = current_keys
        self._markers.publish(message)

    def _next_ranked_candidate(self, result):
        return next(
            (candidate for candidate in self._ranked_candidates(result)
             if not self._is_blacklisted(candidate)),
            None,
        )

    @staticmethod
    def _ranked_candidates(result):
        candidates = list(result.get("candidates", []))
        by_id = {candidate["id"]: candidate for candidate in candidates}
        return [by_id[candidate_id] for candidate_id in result.get("ranking", [])
                if candidate_id in by_id]

    def _is_blacklisted(self, candidate):
        x, y = (float(value) for value in candidate["pose_world"][:2])
        tolerance = 5.0 * float(self._map.info.resolution)
        return any(
            abs(x - blacklisted_x) < tolerance
            and abs(y - blacklisted_y) < tolerance
            for blacklisted_x, blacklisted_y in self._blacklisted_frontiers
        )

    @staticmethod
    def _same_point(first, second):
        return math.hypot(
            float(first["pose_world"][0]) - float(second["pose_world"][0]),
            float(first["pose_world"][1]) - float(second["pose_world"][1]),
        ) < 0.01

    def _latest_owned_goal(self):
        return max(self._owned_goals.values(), key=lambda goal: goal["token"], default=None)

    def _reset_progress(self, token=None, candidate=None):
        self._progress_goal_token = token
        self._progress_min_distance = None
        self._progress_started_ns = None
        if token is not None and candidate is not None:
            self._progress_min_distance = candidate.get("min_distance")
            self._progress_started_ns = self._current_ros_time_ns()

    def _update_progress(self, goal_record, candidate):
        if goal_record["token"] != self._progress_goal_token:
            return
        distance = candidate.get("min_distance")
        if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
                or not math.isfinite(distance)):
            return
        if self._progress_min_distance is None or distance < self._progress_min_distance:
            self._progress_min_distance = float(distance)
            self._progress_started_ns = self._current_ros_time_ns()

    def _check_progress_timeout(self):
        token = self._progress_goal_token
        goal_record = self._owned_goals.get(token)
        if (goal_record is None
                or not self._goal_active
                or goal_record.get("localization_invalidated")
                or self._progress_started_ns is None):
            return
        if self._current_ros_time_ns() - self._progress_started_ns <= self._progress_timeout_ns:
            return
        self._blacklisted_frontiers.append(tuple(
            float(value) for value in goal_record["candidate"]["pose_world"][:2]
        ))
        self._reset_progress()

    def _publish_disabled_markers_if_settled(self):
        if not self._has_goal_ownership():
            self._publish_markers({"candidates": []}, None)

    def _send_goal(self, candidate, behavior_tree):
        robot_pose = self._localization_pose(self._map.header.frame_id)
        if robot_pose is None:
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._map.header.frame_id
        now = self.get_clock().now()
        goal.pose.header.stamp = now.to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = candidate["pose_world"][:2]
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z = 0.0
        goal.pose.pose.orientation.w = 1.0
        yaw = 0.0
        goal.behavior_tree = behavior_tree

        self._goal_sequence += 1
        token = self._goal_sequence
        goal_record = {
            "token": token,
            "epoch": self._search_epoch,
            "candidate": candidate,
            "frame_id": self._map.header.frame_id,
            "handle": None,
            "cancel_requested": False,
            "cancel_reason": None,
            "localization_invalidated": False,
            "state": "pending",
        }
        self._owned_goals[token] = goal_record
        self._current_goal_token = token
        self._goal_active = True
        self._reset_progress(token, candidate)
        future = self._action.send_goal_async(goal)
        stamp_sec, stamp_nanosec = divmod(int(now.nanoseconds), 1_000_000_000)
        self.get_logger().info(
            "Search goal submitted: "
            f"token={token} candidate={candidate['id']} "
            f"frame={goal.pose.header.frame_id} "
            f"stamp={stamp_sec}.{stamp_nanosec:09d} "
            f"x={goal.pose.pose.position.x:.6f} "
            f"y={goal.pose.pose.position.y:.6f} yaw={yaw:.6f}"
        )
        future.add_done_callback(lambda completed: self._goal_response(completed, token))
        return True

    def _goal_response(self, future, token):
        goal_record = self._owned_goals.get(token)
        if goal_record is None:
            return
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().warning(f"Nav2 goal request failed: {error}")
            goal_record["outcome_unknown"] = True
            return

        if not handle.accepted:
            candidate_id = goal_record["candidate"]["id"]
            self.get_logger().info(f"Nav2 rejected Search candidate {candidate_id}")
            self._owned_goals.pop(token, None)
            if self._current_goal_token == token:
                self._current_goal_token = None
                self._goal_active = False
                self._reset_progress()
            if not self._enabled:
                self._publish_disabled_markers_if_settled()
            return

        goal_record["handle"] = handle
        goal_record["state"] = "active"
        self._active_goal_token = token
        if self._current_goal_token == token and self._progress_goal_token != token:
            self._reset_progress(token, goal_record["candidate"])
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self._goal_result(completed, token)
        )
        if (
            goal_record["cancel_reason"] is not None
            or not self._enabled
            or goal_record.get("localization_invalidated")
        ):
            self._cancel_goal(
                goal_record, goal_record["cancel_reason"] or "search_state_changed"
            )

    def _cancel_goal(self, goal_record, reason):
        if goal_record["handle"] is None or goal_record["cancel_requested"]:
            return
        goal_record["cancel_requested"] = True
        if goal_record["cancel_reason"] is None:
            goal_record["cancel_reason"] = reason
        goal_record["handle"].cancel_goal_async()

    def _goal_result(self, future, token):
        goal_record = self._owned_goals.get(token)
        if goal_record is None:
            return
        candidate_id = goal_record["candidate"]["id"]
        try:
            wrapped_result = future.result()
        except Exception as error:
            self.get_logger().warning(f"Nav2 result failed for {candidate_id}: {error}")
            goal_record["outcome_unknown"] = True
            self._cancel_goal(goal_record, "result_unknown")
            return

        status = getattr(wrapped_result, "status", None)
        terminal_statuses = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        }
        if status not in terminal_statuses:
            goal_record["outcome_unknown"] = True
            self._cancel_goal(goal_record, "result_unknown")
            return

        is_current = self._active_goal_token == token
        current_and_valid = (
            is_current and self._enabled
            and goal_record["epoch"] == self._search_epoch
            and not goal_record.get("localization_invalidated", False)
        )
        self._owned_goals.pop(token, None)
        if current_and_valid and status == GoalStatus.STATUS_ABORTED:
            self._blacklisted_frontiers.append(tuple(
                float(value) for value in goal_record["candidate"]["pose_world"][:2]
            ))
        if is_current:
            self._active_goal_token = None
            self._goal_active = False
        if is_current:
            if self._current_goal_token == token:
                self._current_goal_token = None
                self._reset_progress()
        if (current_and_valid and status == GoalStatus.STATUS_SUCCEEDED
                and self._map is not None):
            robot_pose = self._localization_pose(self._map.header.frame_id)
            if robot_pose is not None:
                self._publish_frontier_pose(robot_pose)
        if not self._enabled:
            self._publish_disabled_markers_if_settled()


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    executor = None
    try:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        node._tf_receiver.start()
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        node._tf_receiver.close()
        node.destroy_node()
        try_shutdown()


if __name__ == "__main__":
    main()
