from types import SimpleNamespace
from unittest.mock import Mock

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

from piracer_search.search_node import SearchNode


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callback = callback


class Handle:
    accepted = True

    def __init__(self):
        self.cancel_count = 0

    def get_result_async(self):
        return Future(None)

    def cancel_goal_async(self):
        self.cancel_count += 1


class Action:
    def __init__(self):
        self.sent = []

    def send_goal_async(self, goal):
        self.sent.append(goal)
        return Future(None)

    def server_is_ready(self):
        return True


def make_node():
    node = object.__new__(SearchNode)
    node._enabled = True
    node._search_epoch = 0
    node._localization_generation = 0
    node._clock_epoch = 0
    node._map = SimpleNamespace(
        header=SimpleNamespace(frame_id="map"), info=SimpleNamespace(resolution=0.1)
    )
    node._goal_sequence = 0
    node._now_ns = 1_000_000_000
    node._current_ros_time_ns = lambda: node._now_ns
    node._owned_goals = {}
    node._current_goal_token = None
    node._active_goal_token = None
    node._goal_active = False
    node._blacklisted_frontiers = []
    node._progress_timeout_ns = 30_000_000_000
    node._progress_goal_token = None
    node._progress_min_distance = None
    node._progress_started_ns = None
    node._published_marker_keys = set()
    node._action = Action()
    node._logger = SimpleNamespace(info=Mock(), warning=Mock(), debug=Mock())
    node.get_logger = lambda: node._logger
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(
            nanoseconds=node._now_ns, to_msg=lambda: Time(sec=1)
        )
    )
    node.get_parameter = lambda name: SimpleNamespace(value={
        "behavior_tree": "/search.xml", "progress_timeout": 30.0,
    }[name])
    node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
    node._publish_markers = Mock()
    node._drain_odom_info_snapshot = Mock()
    return node


def frontier(name, x, score, min_distance=5.0):
    return {
        "id": name,
        "cluster_id": name,
        "cluster_size": 2,
        "score": score,
        "cost": -score,
        "min_distance": min_distance,
        "pose_world": [x, 2.0, 0.0],
        "frontiers_world": [[x, 1.0], [x, 2.0]],
    }


def evaluation(candidates):
    ranked = sorted(candidates, key=lambda item: -item["score"])
    return {
        "status": "RECOMMENDATION_AVAILABLE" if ranked else "NO_FRONTIERS",
        "clustered": True,
        "candidates": ranked,
        "ranking": [item["id"] for item in ranked],
    }


def test_goal_uses_upstream_centroid_in_map_frame_with_zero_yaw():
    node = make_node()
    candidate = frontier("frontier-0", 4.25, -2.0)

    assert node._send_goal(candidate, "/search.xml")
    goal = node._action.sent[0].pose
    assert goal.header.frame_id == "map"
    assert (goal.pose.position.x, goal.pose.position.y) == (4.25, 2.0)
    assert goal.pose.orientation.z == 0.0
    assert goal.pose.orientation.w == 1.0


def test_changed_top_centroid_replaces_active_goal_without_cancel_or_score_margin():
    node = make_node()
    first = frontier("first", 4.0, 1.0)
    second = frontier("second", 8.0, 1.05)
    node._apply_search_result(evaluation([first]))
    first_token = node._current_goal_token
    first_handle = Handle()
    node._goal_response(Future(first_handle), first_token)

    node._apply_search_result(evaluation([first, second]))

    assert len(node._action.sent) == 2
    assert first_handle.cancel_count == 0
    assert node._current_goal_token != first_token
    assert node._owned_goals[node._current_goal_token]["candidate"] is second
    assert second["score"] - first["score"] < 0.10

    same_point = frontier("same", 8.005, 1.06)
    node._apply_search_result(evaluation([same_point]))
    assert len(node._action.sent) == 2


def test_disable_cancels_only_owned_goal_and_waits_for_terminal_result():
    node = make_node()
    candidate = frontier("owned", 4.0, 1.0)
    node._send_goal(candidate, "/search.xml")
    token = node._current_goal_token

    response = SimpleNamespace()
    node._set_search(SimpleNamespace(data=False), response)
    assert response.success is False
    assert node._owned_goals[token]["cancel_reason"] == "search_disabled"

    handle = Handle()
    node._goal_response(Future(handle), token)
    assert handle.cancel_count == 1
    assert node._has_goal_ownership()
    node._goal_result(Future(SimpleNamespace(status=GoalStatus.STATUS_CANCELED)), token)
    assert not node._has_goal_ownership()

    response = SimpleNamespace()
    node._set_search(SimpleNamespace(data=False), response)
    assert response.success is True


def test_stale_localization_prevents_frontier_pose_request():
    node = make_node()
    node._frontier_pose = SimpleNamespace(publish=Mock())
    node._localization_pose = lambda frame: None

    node._search_tick()

    node._frontier_pose.publish.assert_not_called()
