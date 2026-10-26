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
from rtabmap_msgs.msg import OdomInfo
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TimeoutException,
)
from visualization_msgs.msg import MarkerArray

from .search_markers import _candidate_markers, _stale_marker_deletions
from .search_algorithm import evaluate_search
from .tf_timing import InternalTfTrace, SearchTfBuffer, SearchTransformListener


def _map_subscription_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _odom_info_subscription_qos():
    return qos_profile_sensor_data


class SearchNode(Node):
    """Evaluate one map snapshot, visualize it, and own at most one Nav2 goal."""

    def __init__(self):
        super().__init__("piracer_search")
        self.declare_parameter("map_topic", "/nav2/map")
        self.declare_parameter("start_stop_service", "/search/enable")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("behavior_tree", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("marker_topic", "/piracer/frontier_markers")
        self.declare_parameter("odom_info_topic", "/visual_vo/odom_info")
        self.declare_parameter("max_localization_age_sec", 0.5)
        self.declare_parameter("goal_switch_margin", 0.10)
        # Simulation settings only; physical camera calibration is TBD.
        self.declare_parameter("camera.horizontal_fov_deg", 65.36016047954865)
        self.declare_parameter("camera.max_range_m", 5.0)

        self._enabled = False
        self._search_epoch = 0
        self._map = None
        self._frontier_key = None
        self._failed_attempts = {}
        self._completed_candidates = set()
        self._retry_target = None
        self._replacement_target = None
        self._goal_sequence = 0
        self._pending_goal = None
        self._active_goal = None
        self._challenger_identity = None
        self._challenger_target = None
        self._challenger_epoch = None
        self._challenger_count = 0
        self._published_marker_keys = set()
        self._odom_info = deque(maxlen=10)
        self._localization_available = False
        self._localization_invalidated_at_ns = None
        self._last_accepted_odom_stamp_ns = None
        self._last_ros_time_ns = None
        self._localization_reason = None
        self._tf_snapshot_attempted = False

        self._internal_tf_trace = InternalTfTrace()
        self._tf_buffer = SearchTfBuffer(self._internal_tf_trace)
        self._tf_listener = SearchTransformListener(
            self._tf_buffer, self, self._internal_tf_trace
        )
        self._action = ActionClient(
            self, NavigateToPose, self.get_parameter("navigate_to_pose_action").value
        )
        self._markers = self.create_publisher(
            MarkerArray, self.get_parameter("marker_topic").value, 10
        )
        self._map_subscription = self.create_subscription(
            OccupancyGrid,
            self.get_parameter("map_topic").value,
            self._map_callback,
            _map_subscription_qos(),
        )
        self._odom_info_subscription = self.create_subscription(
            OdomInfo,
            self.get_parameter("odom_info_topic").value,
            self._odom_info_callback,
            _odom_info_subscription_qos(),
        )
        self._start_stop = self.create_service(
            SetBool, self.get_parameter("start_stop_service").value, self._set_search
        )
        self._timer = self.create_timer(1.0, self._search_tick)
        self.get_logger().info(
            "Search localization max age: "
            f"{float(self.get_parameter('max_localization_age_sec').value):.3f} s"
        )

    def _map_callback(self, message):
        self._map = message

    @staticmethod
    def _stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _current_ros_time_ns(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._last_ros_time_ns is not None and now_ns < self._last_ros_time_ns:
            self._handle_clock_regression(now_ns)
        self._last_ros_time_ns = now_ns
        return now_ns

    def _mark_owned_goal_localization_invalidated(self, reason):
        if self._pending_goal is not None:
            self._pending_goal["localization_invalidated"] = True
            if self._pending_goal["cancel_reason"] is None:
                self._pending_goal["cancel_reason"] = reason
        if self._active_goal is not None:
            self._active_goal["localization_invalidated"] = True
            if not self._active_goal["cancel_requested"]:
                self._cancel_active_goal(reason)

    def _handle_clock_regression(self, now_ns):
        was_available = self._localization_available
        self._odom_info.clear()
        self._tf_buffer.clear()
        self._localization_available = False
        self._localization_invalidated_at_ns = None
        self._last_accepted_odom_stamp_ns = None
        self._mark_owned_goal_localization_invalidated("clock_regression")
        self._reset_challenger()
        if was_available or self._localization_reason != "ROS clock moved backwards":
            self.get_logger().warning("Search localization invalidated: ROS clock moved backwards")
        self._localization_reason = "ROS clock moved backwards"

    def _invalidate_localization(
        self, reason, lost_stamp_ns=None, diagnostic_details=None,
        tf_failure_monotonic_ns=None,
    ):
        was_available = self._localization_available
        boundaries = [self._localization_invalidated_at_ns, self._last_accepted_odom_stamp_ns]
        if lost_stamp_ns is not None and lost_stamp_ns > 0:
            boundaries.append(lost_stamp_ns)
        if was_available:
            boundaries.append(self._current_ros_time_ns())
        self._localization_invalidated_at_ns = max(
            (value for value in boundaries if value is not None), default=None
        )
        self._localization_available = False
        self._odom_info.clear()
        self._reset_challenger()
        self._mark_owned_goal_localization_invalidated("localization_invalidated")
        if (was_available and tf_failure_monotonic_ns is not None
                and not self._tf_snapshot_attempted):
            self._tf_snapshot_attempted = True
            snapshot_started_ns = time.monotonic_ns()
            snapshot_yaml = None
            capture_error = None
            try:
                snapshot_yaml = self._tf_buffer.all_frames_as_yaml()
            except Exception as exc:
                capture_error = f"{type(exc).__name__}: {str(exc)!r}"
            snapshot_ended_ns = time.monotonic_ns()
            snapshot_details = (
                f"tf_failure_monotonic_ns={tf_failure_monotonic_ns}, "
                f"buffer_snapshot_start_monotonic_ns={snapshot_started_ns}, "
                f"buffer_snapshot_end_monotonic_ns={snapshot_ended_ns}, "
                f"buffer_snapshot_yaml={snapshot_yaml!r}, "
                f"buffer_snapshot_capture_error={capture_error!r}"
            )
            try:
                trace_capture = self._internal_tf_trace.capture()
                trace_details = (
                    f"internal_tf_trace_capture_start_monotonic_ns="
                    f"{trace_capture['capture_start_monotonic_ns']}, "
                    f"internal_tf_trace_capture_end_monotonic_ns="
                    f"{trace_capture['capture_end_monotonic_ns']}, "
                    f"internal_tf_trace_evicted_count={trace_capture['evicted_count']}, "
                    f"internal_tf_trace_oldest_retained_monotonic_ns="
                    f"{trace_capture['oldest_retained_monotonic_ns']}, "
                    f"internal_tf_trace_json={json.dumps(trace_capture, separators=(',', ':'))!r}"
                )
            except Exception as exc:
                trace_details = f"internal_tf_trace_capture_error={type(exc).__name__}: {str(exc)!r}"
            snapshot_details += "; " + trace_details
            if diagnostic_details is None:
                diagnostic_details = snapshot_details
            else:
                diagnostic_details += "; " + snapshot_details
        if was_available or self._localization_reason != reason:
            message = f"Search localization invalidated: {reason}"
            if diagnostic_details is not None:
                message += f"; {diagnostic_details}"
            self.get_logger().warning(message)
        self._localization_reason = reason

    def _odom_info_callback(self, message):
        self._current_ros_time_ns()
        stamp_ns = self._stamp_ns(message)
        if message.lost:
            self._invalidate_localization("OdomInfo reports lost", stamp_ns)
            return
        if stamp_ns == 0:
            return
        if (self._localization_invalidated_at_ns is not None
                and stamp_ns <= self._localization_invalidated_at_ns):
            return
        self._odom_info.append(message)

    def _localization_pose(self, map_frame):
        now_ns = self._current_ros_time_ns()
        eligible = [
            message for message in self._odom_info
            if 0 < self._stamp_ns(message) <= now_ns
            and (self._localization_invalidated_at_ns is None
                 or self._stamp_ns(message) > self._localization_invalidated_at_ns)
            and (self._last_accepted_odom_stamp_ns is None
                 or self._stamp_ns(message) >= self._last_accepted_odom_stamp_ns)
        ]
        if not eligible:
            if self._localization_available:
                self._invalidate_localization("no current OdomInfo")
            return None

        max_age_ns = round(float(self.get_parameter("max_localization_age_sec").value) * 1_000_000_000)
        candidates = []
        for message in sorted(eligible, key=self._stamp_ns, reverse=True):
            stamp_ns = self._stamp_ns(message)
            age_ns = now_ns - stamp_ns
            if age_ns <= max_age_ns:
                candidates.append((stamp_ns, age_ns))

        if not candidates:
            self._invalidate_localization("OdomInfo is older than max_localization_age_sec")
            return None

        base_frame = self.get_parameter("base_frame").value
        first_failure = None
        for stamp_ns, age_ns in candidates:
            measurement_time = Time(
                nanoseconds=stamp_ns,
                clock_type=self.get_clock().clock_type,
            )
            candidate_transform = None
            candidate_failed = False
            for target_frame, reason in (
                ("vo_odom", "vo_odom to base frame TF unavailable"),
                (map_frame, "map to base frame TF unavailable"),
            ):
                try:
                    candidate_transform = self._tf_buffer.lookup_transform(
                        target_frame, base_frame, measurement_time
                    )
                except Exception as exc:
                    availability_exception = isinstance(exc, (
                        LookupException,
                        ConnectivityException,
                        ExtrapolationException,
                        TimeoutException,
                    ))
                    failure_monotonic_ns = (
                        time.monotonic_ns() if availability_exception else None
                    )
                    diagnostic_details = (
                        f"selected_measurement_stamp_ns={stamp_ns}, "
                        f"requested_stamp_ns={measurement_time.nanoseconds}, "
                        f"ros_now_ns={now_ns}, age_ns={age_ns}, max_age_ns={max_age_ns}, "
                        f"target_frame={target_frame!r}, source_frame={base_frame!r}, "
                        f"exception_class={type(exc).__name__}, "
                        f"exception_message={str(exc)!r}"
                    )
                    if availability_exception:
                        if first_failure is None:
                            first_failure = (
                                reason, diagnostic_details, failure_monotonic_ns
                            )
                        candidate_failed = True
                        break
                    self._invalidate_localization(
                        reason, diagnostic_details=diagnostic_details
                    )
                    return None
                if target_frame == map_frame:
                    transform = candidate_transform
            if not candidate_failed:
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                yaw = math.atan2(
                    2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                    1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
                )
                was_available = self._localization_available
                self._localization_available = True
                self._localization_invalidated_at_ns = None
                self._last_accepted_odom_stamp_ns = stamp_ns
                self._localization_reason = None
                if not was_available:
                    self.get_logger().info("Search localization is available")
                return translation.x, translation.y, yaw

        if first_failure is not None:
            reason, diagnostic_details, failure_monotonic_ns = first_failure
            self._invalidate_localization(
                reason, diagnostic_details=diagnostic_details,
                tf_failure_monotonic_ns=failure_monotonic_ns,
            )
            return None

        return None

    def _set_search(self, request, response):
        if request.data:
            if not self._enabled:
                self._search_epoch += 1
                self._frontier_key = None
                self._failed_attempts.clear()
                self._completed_candidates.clear()
                self._retry_target = None
                self._replacement_target = None
                self._reset_challenger()
            self._enabled = True
            response.message = "Search enabled"
        else:
            if self._enabled:
                self._search_epoch += 1
            self._enabled = False
            if self._pending_goal is not None:
                self._pending_goal["cancel_reason"] = "search_disabled"
            self._cancel_active_goal("search_disabled")
            self._reset_challenger()
            if self._has_goal_ownership():
                selected = self._selected_for_markers({"candidates": []}, {})
                self._publish_markers({"candidates": []}, selected)
                response.message = "Search disabled; waiting for Nav2 terminal result"
            else:
                self._publish_markers({"candidates": []}, None)
                response.message = "Search disabled"
        response.success = bool(request.data or not self._has_goal_ownership())
        return response

    def _has_goal_ownership(self):
        return self._pending_goal is not None or self._active_goal is not None

    def _grid(self):
        info = self._map.info
        origin = info.origin
        orientation = origin.orientation
        origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        return {
            "width": info.width,
            "height": info.height,
            "resolution": info.resolution,
            "origin": [origin.position.x, origin.position.y, origin_yaw],
            "data": tuple(self._map.data),
        }

    def _search_tick(self):
        trace = getattr(self, "_internal_tf_trace", None)
        if trace is None:
            return self._search_tick_body(None)
        span_id, tick_sequence = trace.begin("search_tick")
        error = None
        try:
            return self._search_tick_body(tick_sequence)
        except BaseException as exc:
            error = exc
            raise
        finally:
            trace.finish(span_id, error)

    def _search_tick_body(self, tick_sequence):
        if not self._enabled or self._map is None:
            return

        camera = {
            "horizontal_fov_deg": self.get_parameter(
                "camera.horizontal_fov_deg"
            ).value,
            "max_range_m": self.get_parameter("camera.max_range_m").value,
        }
        robot_pose = self._localization_pose(self._map.header.frame_id)
        if robot_pose is None:
            return
        trace = getattr(self, "_internal_tf_trace", None)
        grid_span = None
        grid_error = None
        try:
            if trace is not None:
                grid_span, _ = trace.begin(
                    "search_computation", tick_sequence=tick_sequence, operation="grid"
                )
            grid = self._grid()
        except Exception as error:
            grid_error = error
            self.get_logger().debug(f"Search waits for current map/TF: {error}")
            return
        except BaseException as error:
            grid_error = error
            raise
        finally:
            if trace is not None and grid_span is not None:
                trace.finish(grid_span, grid_error)

        evaluation_span = None
        evaluation_error = None
        try:
            if trace is not None:
                evaluation_span, _ = trace.begin(
                    "search_computation",
                    tick_sequence=tick_sequence,
                    operation="evaluate_search",
                )
            result = evaluate_search(grid, robot_pose, camera=camera)
        except Exception as error:
            evaluation_error = error
            self.get_logger().debug(f"Search waits for current map/TF: {error}")
            return
        except BaseException as error:
            evaluation_error = error
            raise
        finally:
            if trace is not None and evaluation_span is not None:
                trace.finish(evaluation_span, evaluation_error)

        if self._localization_pose(self._map.header.frame_id) is None:
            return

        candidates = {candidate["id"]: candidate for candidate in result["candidates"]}
        eligible_identities = {
            self._goal_identity(candidate, self._map.header.frame_id)
            for candidate in candidates.values()
        }
        if "frontier_cells" in result:
            frontier_key = tuple(tuple(cell) for cell in result["frontier_cells"])
            if frontier_key != self._frontier_key:
                self._frontier_key = frontier_key
                self._failed_attempts = {
                    identity: count
                    for identity, count in self._failed_attempts.items()
                    if identity in eligible_identities
                }
                self._completed_candidates.clear()
                if self._replacement_target not in eligible_identities:
                    self._replacement_target = None
                self._reset_challenger()

        self._invalidate_missing_owned_retry(candidates)
        selected = self._selected_for_markers(result, candidates)
        self._publish_markers(result, selected)
        self._consider_goal_switch(result, candidates)

        if self._pending_goal is not None or self._active_goal is not None:
            return
        if result["status"] != "RECOMMENDATION_AVAILABLE":
            return
        if not self._action.server_is_ready():
            return

        behavior_tree = self.get_parameter("behavior_tree").value
        if not behavior_tree:
            self.get_logger().warning("Search has no NavigateToPose behavior_tree configured")
            return
        candidate = self._next_ranked_candidate(result)
        if candidate is not None:
            self._send_goal(candidate, behavior_tree)

    def _selected_for_markers(self, result, candidates):
        current = self._pending_goal or self._active_goal
        if current is not None:
            return current["candidate"]
        return self._next_ranked_candidate(result)

    def _invalidate_missing_owned_retry(self, candidates):
        eligible_identities = {
            self._goal_identity(candidate, self._map.header.frame_id)
            for candidate in candidates.values()
        }
        if (self._retry_target is not None
                and self._retry_target not in eligible_identities):
            self._retry_target = None
        owner = self._pending_goal or self._active_goal
        if owner is None or not owner.get("is_retry"):
            return
        identity = owner["identity"]
        if identity in eligible_identities:
            return
        if self._retry_target == identity:
            self._retry_target = None
        if owner.get("cancel_reason") is not None:
            return
        if owner is self._pending_goal:
            owner["cancel_reason"] = "retry_invalidated"
        else:
            self._cancel_active_goal("retry_invalidated")

    def _publish_markers(self, result, selected):
        if self._map is None:
            return
        candidates = list(result.get("candidates", []))
        frame = self._map.header.frame_id or "map"
        current = self._pending_goal or self._active_goal
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
        by_id = {candidate["id"]: candidate for candidate in result["candidates"]}
        ranked = [by_id[candidate_id] for candidate_id in result["ranking"]
                  if candidate_id in by_id]

        if self._replacement_target is not None:
            replacement = next(
                (candidate for candidate in ranked
                 if self._goal_identity(candidate, self._map.header.frame_id)
                 == self._replacement_target
                 and self._goal_identity(candidate, self._map.header.frame_id)
                 not in self._completed_candidates
                 and self._failed_attempts.get(self._replacement_target, 0) < 2),
                None,
            )
            if replacement is not None:
                return replacement
            self._replacement_target = None

        if self._retry_target is not None:
            retry = next(
                (candidate for candidate in ranked
                 if self._goal_identity(candidate, self._map.header.frame_id)
                 == self._retry_target
                 and self._retry_target not in self._completed_candidates
                 and self._failed_attempts.get(self._retry_target, 0) < 2),
                None,
            )
            if retry is not None:
                return retry
            self._retry_target = None

        return next(
            (
                candidate for candidate in ranked
                if self._goal_identity(candidate, self._map.header.frame_id)
                not in self._completed_candidates
                and self._failed_attempts.get(
                    self._goal_identity(candidate, self._map.header.frame_id), 0
                ) < 2
            ),
            None,
        )

    @staticmethod
    def _goal_identity(candidate, frame_id):
        return (candidate["id"], frame_id, tuple(candidate["pose_world"]))

    def _send_goal(self, candidate, behavior_tree):
        if self._localization_pose(self._map.header.frame_id) is None:
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._map.header.frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = candidate["pose_world"][:2]
        goal.pose.pose.position.z = 0.0
        yaw = candidate["pose_world"][2]
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.behavior_tree = behavior_tree

        self._goal_sequence += 1
        token = self._goal_sequence
        pending = {
            "token": token,
            "epoch": self._search_epoch,
            "candidate": candidate,
            "frame_id": self._map.header.frame_id,
            "identity": self._goal_identity(candidate, self._map.header.frame_id),
            "is_retry": self._retry_target == self._goal_identity(
                candidate, self._map.header.frame_id
            ),
            "cancel_reason": None,
        }
        self._pending_goal = pending
        if self._replacement_target == pending["identity"]:
            self._replacement_target = None
        future = self._action.send_goal_async(goal)
        future.add_done_callback(lambda completed: self._goal_response(completed, token))
        return True

    def _goal_response(self, future, token):
        pending = self._pending_goal
        if pending is None or pending["token"] != token:
            return
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().warning(f"Nav2 goal request failed: {error}")
            pending["outcome_unknown"] = True
            return

        self._pending_goal = None
        if not handle.accepted:
            candidate_id = pending["candidate"]["id"]
            self.get_logger().info(f"Nav2 rejected Search candidate {candidate_id}")
            if (pending["epoch"] == self._search_epoch
                    and not pending.get("localization_invalidated", False)):
                if pending["cancel_reason"] not in (
                    "higher_scoring_candidate", "retry_invalidated"
                ):
                    self._record_failure(pending)
            if not self._enabled:
                self._publish_markers({"candidates": []}, None)
            return

        self._active_goal = {
            **pending,
            "handle": handle,
            "cancel_requested": False,
        }
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self._goal_result(completed, token)
        )
        if (
            pending["cancel_reason"] is not None
            or not self._enabled
            or pending["epoch"] != self._search_epoch
        ):
            self._cancel_active_goal(
                pending["cancel_reason"] or "search_state_changed"
            )

    def _consider_goal_switch(self, result, candidates):
        if self._pending_goal is not None and self._pending_goal["cancel_reason"] is not None:
            return
        current = self._pending_goal or self._active_goal
        if current is None and self._retry_target is not None:
            current = {"identity": self._retry_target, "epoch": self._search_epoch}
        if current is not None and current.get("epoch", self._search_epoch) != self._search_epoch:
            return
        if current is None or (
            self._active_goal is not None and self._active_goal["cancel_requested"]
        ):
            return

        target_identity = current.get(
            "identity",
            self._goal_identity(current["candidate"], self._map.header.frame_id)
            if "candidate" in current else None,
        )
        target_epoch = current.get("epoch", self._search_epoch)
        current_candidate = next(
            (candidate for candidate in candidates.values()
             if self._goal_identity(candidate, self._map.header.frame_id)
             == target_identity),
            None,
        )
        if current_candidate is None:
            self._reset_challenger()
            if self._retry_target == target_identity:
                self._retry_target = None
                if self._pending_goal is not None:
                    self._pending_goal["cancel_reason"] = "retry_invalidated"
                elif self._active_goal is not None and self._active_goal["is_retry"]:
                    self._cancel_active_goal("retry_invalidated")
            return

        challenger = next(
            (
                candidates[candidate_id]
                for candidate_id in result["ranking"]
                if candidate_id in candidates
                and self._goal_identity(candidates[candidate_id], self._map.header.frame_id)
                != target_identity
                and self._goal_identity(candidates[candidate_id], self._map.header.frame_id)
                not in self._completed_candidates
                and self._failed_attempts.get(
                    self._goal_identity(candidates[candidate_id], self._map.header.frame_id), 0
                ) < 2
            ),
            None,
        )
        margin = float(self.get_parameter("goal_switch_margin").value)
        if (challenger is None
                or challenger["score"] - current_candidate["score"] + 1e-12 < margin):
            self._reset_challenger()
            return

        challenger_identity = self._goal_identity(challenger, self._map.header.frame_id)
        if (challenger_identity == self._challenger_identity
                and target_identity == self._challenger_target
                and target_epoch == self._challenger_epoch):
            self._challenger_count += 1
        else:
            self._challenger_identity = challenger_identity
            self._challenger_target = target_identity
            self._challenger_epoch = target_epoch
            self._challenger_count = 1
        if self._challenger_count < 2:
            return

        self._replacement_target = challenger_identity
        if self._retry_target == target_identity:
            self._retry_target = None
        if self._pending_goal is not None:
            self._pending_goal["cancel_reason"] = "higher_scoring_candidate"
        else:
            if self._active_goal is not None:
                self._cancel_active_goal("higher_scoring_candidate")
        self._reset_challenger()

    def _cancel_active_goal(self, reason):
        if self._active_goal is None or self._active_goal["cancel_requested"]:
            return
        self._active_goal["cancel_requested"] = True
        self._active_goal["cancel_reason"] = reason
        self._active_goal["handle"].cancel_goal_async()

    def _goal_result(self, future, token):
        active = self._active_goal
        if active is None or active["token"] != token:
            return
        candidate_id = active["candidate"]["id"]
        active.setdefault("identity", self._goal_identity(
            active["candidate"], self._map.header.frame_id
        ))
        active.setdefault("epoch", self._search_epoch)
        active.setdefault("is_retry", False)
        try:
            wrapped_result = future.result()
        except Exception as error:
            self.get_logger().warning(f"Nav2 result failed for {candidate_id}: {error}")
            active["outcome_unknown"] = True
            self._cancel_active_goal("result_unknown")
            return

        status = getattr(wrapped_result, "status", None)
        terminal_statuses = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        }
        if status not in terminal_statuses:
            active["outcome_unknown"] = True
            self._cancel_active_goal("result_unknown")
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            if active["epoch"] == self._search_epoch:
                self._completed_candidates.add(active["identity"])
        elif not active.get("localization_invalidated", False) and not (
            status == GoalStatus.STATUS_CANCELED
            and active["cancel_reason"] is not None
        ) and active["cancel_reason"] not in (
            "higher_scoring_candidate", "retry_invalidated"
        ):
            # Only an intentional canceled result avoids a failed attempt.
            if active["epoch"] == self._search_epoch:
                self._record_failure(active)

        self._active_goal = None
        if active["cancel_reason"] in ("retry_invalidated", "higher_scoring_candidate"):
            self._retry_target = None
        if not self._enabled:
            self._publish_markers({"candidates": []}, None)

    def _record_failure(self, goal):
        identity = goal["identity"]
        count = self._failed_attempts.get(identity, 0) + 1
        self._failed_attempts[identity] = count
        if count == 1 and goal["cancel_reason"] != "higher_scoring_candidate":
            self._retry_target = identity
        elif count >= 2 and self._retry_target == identity:
            self._retry_target = None

    def _reset_challenger(self):
        self._challenger_identity = None
        self._challenger_target = None
        self._challenger_epoch = None
        self._challenger_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
