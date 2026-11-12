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
    def __init__(self, handle=None):
        self.handle = handle or Handle()
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
    node._map = SimpleNamespace(header=SimpleNamespace(frame_id="map"))
    node._goal_sequence = 0
    node._now_ns = 1_000_000_000
    node._current_ros_time_ns = lambda: node._now_ns
    node._pending_goal = None
    node._active_goal = None
    node._retry_target = None
    node._replacement_target = None
    node._replacement_proposal = None
    node._next_goal_dynamic = False
    node._next_goal_requested = False
    node._last_cluster_ranking = None
    node._cluster_ranking_context = None
    node._failed_attempts = {}
    node._exhausted_candidates = set()
    node._completed_candidates = set()
    node._published_marker_keys = set()
    node._action = Action()
    node._logger = SimpleNamespace(info=Mock(), warning=Mock(), debug=Mock())
    node.get_logger = lambda: node._logger
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(
            nanoseconds=1_000_000_000, to_msg=lambda: Time(sec=1)
        )
    )
    node.get_parameter = lambda name: SimpleNamespace(value={
        "behavior_tree": "/search.xml", "goal_switch_margin": 0.10,
    }[name])
    node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
    node._grid = lambda: {
        "width": 20, "height": 20, "resolution": 1.0,
        "origin": [0.0, 0.0, 0.0], "data": (0,) * 400,
    }
    node._cancel_active_goal = lambda reason: SearchNode._cancel_active_goal(node, reason)
    node._reset_challenger = lambda: None
    node._publish_markers = Mock()
    node._drain_odom_info_snapshot = Mock()
    return node


def cluster(name, x, score, points):
    return {
        "id": name, "cluster_id": name, "cluster_size": len(points),
        "score": score, "cost": -score, "pose_world": [x, 2.0, 0.0],
        "frontiers_world": points,
    }


def test_upstream_centroid_goal_uses_map_frame_and_zero_yaw():
    node = make_node()
    node._localization_pose = lambda frame: (0.0, 0.0, 1.2)
    candidate = cluster("frontier-0", 4.25, -2.0, [[4.0, 2.0]])

    assert node._send_goal(candidate, "/search.xml")
    goal = node._action.sent[0].pose
    assert goal.header.frame_id == "map"
    assert (goal.pose.position.x, goal.pose.position.y) == (4.25, 2.0)
    assert goal.pose.orientation.z == 0.0
    assert goal.pose.orientation.w == 1.0


def test_switch_margin_compares_clusters_from_one_upstream_result():
    node = make_node()
    active = cluster("active", 5.0, -5.0, [[5.0, 2.0]])
    below = cluster("below", 8.0, -4.95, [[8.0, 2.0]])
    handle = Handle()
    node._active_goal = {
        "candidate": active, "epoch": 0, "cancel_requested": False,
        "localization_invalidated": False, "outcome_unknown": False,
        "identity": SearchNode._goal_identity(active, "map"),
        "handle": handle,
    }
    node._goal_switch_deadline_ns = 0
    result = {"candidates": [active, below], "cluster_candidates": [active, below]}
    node._consider_goal_switch(result, {x["id"]: x for x in result["candidates"]})
    assert handle.cancel_count == 0

    node._goal_switch_deadline_ns = 0
    above = cluster("above", 9.0, -4.89, [[9.0, 2.0]])
    result = {"candidates": [active, above], "cluster_candidates": [active, above]}
    node._consider_goal_switch(result, {x["id"]: x for x in result["candidates"]})
    assert handle.cancel_count == 1
    assert node._active_goal["cancel_reason"] == "higher_scoring_candidate"


def test_disable_during_goal_acceptance_waits_for_its_terminal_result():
    node = make_node()
    candidate = cluster("frontier-0", 4.0, -1.0, [[4.0, 2.0]])
    node._send_goal(candidate, "/search.xml")
    token = node._pending_goal["token"]

    response = SimpleNamespace()
    node._set_search(SimpleNamespace(data=False), response)
    assert response.success is False
    assert node._pending_goal["cancel_reason"] == "search_disabled"
    handle = Handle()
    node._goal_response(Future(handle), token)
    assert handle.cancel_count == 1
    assert node._active_goal is not None

    node._goal_result(Future(SimpleNamespace(status=GoalStatus.STATUS_CANCELED)), token)
    assert node._active_goal is None
    response = SimpleNamespace()
    node._set_search(SimpleNamespace(data=False), response)
    assert response.success is True


def test_stale_localization_prevents_frontier_pose_request():
    node = make_node()
    node._map = SimpleNamespace(header=SimpleNamespace(frame_id="map"))
    node._frontier_pose = SimpleNamespace(publish=Mock())
    node._localization_pose = lambda frame: None

    node._search_tick()

    node._frontier_pose.publish.assert_not_called()
    assert node._last_cluster_ranking is None
