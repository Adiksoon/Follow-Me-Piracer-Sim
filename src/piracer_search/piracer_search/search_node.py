"""Search lifecycle, shared frontier evaluation, and Nav2 goal handoff."""

import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from .search_markers import _candidate_markers, _stale_marker_deletions
from .search_algorithm import evaluate_search


def _map_subscription_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


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

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
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
        self._start_stop = self.create_service(
            SetBool, self.get_parameter("start_stop_service").value, self._set_search
        )
        self._timer = self.create_timer(1.0, self._search_tick)

    def _map_callback(self, message):
        self._map = message

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

    def _robot_pose(self, map_frame):
        transform = self._tf_buffer.lookup_transform(
            map_frame, self.get_parameter("base_frame").value, Time()
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        return translation.x, translation.y, yaw

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
        if not self._enabled or self._map is None:
            return

        camera = {
            "horizontal_fov_deg": self.get_parameter(
                "camera.horizontal_fov_deg"
            ).value,
            "max_range_m": self.get_parameter("camera.max_range_m").value,
        }
        try:
            result = evaluate_search(
                self._grid(),
                self._robot_pose(self._map.header.frame_id),
                camera=camera,
            )
        except Exception as error:
            self.get_logger().debug(f"Search waits for current map/TF: {error}")
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
            if pending["epoch"] == self._search_epoch:
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
        elif not (
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
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
