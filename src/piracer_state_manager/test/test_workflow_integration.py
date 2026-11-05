"""Detector, 3D confirmation and behavior ownership integration, fully offline."""
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from piracer_state_manager.follow_node import PiracerFollowNode
from piracer_state_manager.state_machine import ACQUIRE, FOLLOW, SEARCH, StateMachine
from piracer_state_manager.state_node import StateManagerNode
from test_state_node_clock import node, detection


@pytest.fixture
def workflow(monkeypatch):
    n, clock = node()
    wall = [100.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: wall[0])
    for name in ("_on_target_point", "_on_follow_status", "_on_behavior_enable_result"):
        setattr(n, name, MethodType(getattr(StateManagerNode, name), n))
    n.behavior_coordinator.confirmed = {"search": False, "acquire": False, "follow": False}
    n.behavior_coordinator.pending = None
    n._handoff_failure_logged = False

    def at(t, positive=True, point=True, point_first=False):
        clock.nanoseconds = round(t * 1e9)
        wall[0] = 100.0 + t
        msg = detection(clock.nanoseconds)
        if positive:
            msg.detections = [object()]
        target = PointStamped()
        target.header.stamp.sec, target.header.stamp.nanosec = divmod(clock.nanoseconds, 10**9)
        target.header.frame_id = "camera_color_optical_frame"
        target.point.z = 3.0
        if point and point_first:
            n._on_target_point(target)
        n._on_detections(msg)
        if point and not point_first:
            n._on_target_point(target)
        return target

    return n, clock, wall, at


@pytest.mark.parametrize("point_first", [False, True])
def test_two_yolo_frames_then_five_new_3d_points_in_either_callback_order(workflow, point_first):
    n, _, _, at = workflow
    at(10.0, point_first=point_first)
    at(10.1, point_first=point_first)
    assert n.machine.state == ACQUIRE and n.machine.hit_streak == 0
    for t in (10.2, 10.3, 10.4, 10.5):
        at(t, point_first=point_first)
    assert n.machine.state == ACQUIRE and n.machine.hit_streak == 4
    at(10.6, point_first=point_first)
    assert n.machine.state == FOLLOW


def test_yolo_alone_does_not_admit_acquire_or_follow(workflow):
    n, _, _, at = workflow
    for i in range(12):
        at(10 + i / 10, point=False)
    assert n.machine.active and n.machine.state == SEARCH


def test_duplicate_and_older_points_do_not_renew_cache_or_confirmation(workflow):
    n, _, wall, at = workflow
    at(10)
    p = at(10.1)
    receipt = n._target_received
    wall[0] += 0.1
    n._on_target_point(p)
    p.header.stamp.nanosec -= 1
    n._on_target_point(p)
    assert n.machine.hit_streak == 0 and n._target_received == receipt


def test_negative_frame_barrier_rejects_late_positive_point(workflow):
    n, _, _, at = workflow
    at(10); at(10.1); old = at(10.2, point=False)
    at(10.3, positive=False, point=False)
    n._on_target_point(old)
    assert n.machine.hit_streak == 0
    at(10.4)
    assert n.machine.hit_streak == 1


def test_stale_point_breaks_confirmation_streak_before_reacquisition(workflow):
    n, _, _, at = workflow
    at(10); at(10.1); at(10.2)
    assert n.machine.hit_streak == 1
    at(10.8, point=False)
    n._on_timer()
    assert n.machine.hit_streak == 0
    at(10.9)
    assert n.machine.hit_streak == 1


@pytest.mark.parametrize("bad", ["nan", "frame", "future", "old"])
def test_invalid_point_cannot_admit_acquire(workflow, bad):
    n, _, _, at = workflow
    at(10, point=False); p = at(10.1, point=False)
    if bad == "nan":
        p.point.z = float("nan")
    elif bad == "frame":
        p.header.frame_id = ""
    elif bad == "future":
        p.header.stamp.sec += 1
    else:
        p.header.stamp.sec -= 1
    n._on_target_point(p)
    assert n._target_stamp is None and n.machine.state == SEARCH


def reach_follow(at):
    for i in range(7):
        at(10 + i / 10)


def test_fresh_avoidance_status_suspends_misses_but_detector_timeout_still_stops(workflow):
    n, clock, wall, at = workflow
    reach_follow(at)
    n.behavior_coordinator.confirmed["follow"] = True
    n._follow_enabled_wall = wall[0]
    for t in (10.7, 10.8, 10.9, 11.0):
        n._on_follow_status(String(data="AVOIDING"))
        at(t, positive=False, point=False)
    assert n.machine.state == FOLLOW and n.machine.miss_streak == 0
    # FOLLOW status cannot replace the detector heartbeat.
    clock.nanoseconds = 13_100_000_000; wall[0] = 113.1
    n._on_follow_status(String(data="AVOIDING"))
    n._on_timer()
    assert not n.machine.active and n.machine.state == SEARCH


def test_direct_loss_requires_new_misses_after_avoidance_ends(workflow):
    n, _, wall, at = workflow
    reach_follow(at)
    n.behavior_coordinator.confirmed["follow"] = True
    n._follow_enabled_wall = wall[0]
    n._on_follow_status(String(data="AVOIDING"))
    at(10.7, positive=False, point=False)
    n._on_follow_status(String(data="DIRECT"))
    for t in (10.8, 10.9):
        at(t, positive=False, point=False)
        assert n.machine.state == FOLLOW
    at(11.0, positive=False, point=False)
    assert n.machine.state == ACQUIRE


def test_old_history_after_bypass_returns_to_search(workflow):
    n, _, wall, at = workflow
    reach_follow(at)
    n.behavior_coordinator.confirmed["follow"] = True
    n._follow_enabled_wall = wall[0]
    for i in range(7, 42):
        n._on_follow_status(String(data="AVOIDING"))
        at(10 + i / 10, positive=False, point=False)
    n._on_follow_status(String(data="DIRECT"))
    for t in (14.2, 14.3, 14.4):
        at(t, positive=False, point=False)
    assert n.machine.state == SEARCH


def test_missing_follow_status_latches_stop_instead_of_inferring_finished_bypass(workflow):
    n, clock, wall, at = workflow
    reach_follow(at)
    n.behavior_coordinator.confirmed["follow"] = True
    n._follow_enabled_wall = wall[0]
    n._on_follow_status(String(data="AVOIDING"))
    clock.nanoseconds += 600_000_000; wall[0] += 0.6
    n._on_timer()
    assert not n.machine.enabled and not n.machine.active


def test_manual_stop_wins_over_avoidance(workflow):
    n, _, wall, at = workflow
    reach_follow(at)
    n.behavior_coordinator.confirmed["follow"] = True
    n._follow_enabled_wall = wall[0]
    n._on_follow_status(String(data="AVOIDING"))
    n._on_enable(SimpleNamespace(data=False), SimpleNamespace())
    at(10.7)
    assert not n.machine.enabled and not n.machine.active


def test_fault_while_disabling_follow_blocks_successor_even_after_logical_transition(workflow):
    n, _, _, at = workflow
    reach_follow(at)
    n.machine._reset_state(ACQUIRE, 11.0)
    from piracer_state_manager.state_machine import BehaviorEnableCoordinator
    n.behavior_coordinator = BehaviorEnableCoordinator(("search", "acquire", "follow"))
    n.behavior_coordinator.confirmed.update(search=False, acquire=False, follow=True)
    n.behavior_coordinator.begin("follow", False)
    f = Future(); n._handoff_future = f
    f.set_result(SimpleNamespace(success=True, message="Follow disabled; fault_latched=true"))
    n._on_behavior_enable_result("follow", False, f)
    assert not n.machine.enabled and not n.machine.active
    assert n.behavior_coordinator.confirmed["follow"] is False


def test_enable_epoch_drops_old_fault_status_and_requires_new_heartbeat(workflow):
    n, _, _, _ = workflow
    from piracer_state_manager.state_machine import BehaviorEnableCoordinator
    n.behavior_coordinator = BehaviorEnableCoordinator(("search", "acquire", "follow"))
    n.behavior_coordinator.confirmed.update(search=False, acquire=False, follow=False)
    n.behavior_coordinator.begin("follow", True)
    n._on_follow_status(String(data="FAULT"))
    f = Future(); n._handoff_future = f
    f.set_result(SimpleNamespace(success=True, message="enabled"))
    n._on_behavior_enable_result("follow", True, f)
    assert n._follow_status is None and n.machine.enabled


def test_point_callbacks_cannot_replace_detector_heartbeat(workflow):
    n, _, _, at = workflow
    at(10); at(10.1)
    p = at(12.2, point=False)
    # Simulate a new 3D point without another detector frame after its deadline.
    n.machine.last_receipt_time = 10.1
    n._on_target_point(p)
    assert not n.machine.active and n.machine.state == SEARCH


@pytest.mark.parametrize("outcome,expected", [("exhausted", SEARCH), ("blocked", SEARCH),
                                             ("canceled", SEARCH), ("already_close", ACQUIRE)])
def test_normal_acquire_results_and_30s_budget(outcome, expected):
    m = StateMachine(detector_timeout_s=100)
    m.observe(1, True, 0, can_acquire=True)
    m.observe(2, True, .1, can_acquire=True)
    m.finish_acquire(outcome)
    assert m.state == expected and m.active
    if outcome == "already_close":
        assert m.acquire_finished
        assert m.tick(30.2) == ["ACQUIRE_TIMEOUT"]


def test_acquire_fault_latches_stop():
    m = StateMachine()
    m.observe(1, True, 0, can_acquire=True)
    m.observe(2, True, .1, can_acquire=True)
    m.finish_acquire("failed")
    assert not m.active and not m.enabled


@pytest.mark.parametrize("enabled,fault,owned,memory,expected", [
    (False, True, True, None, "DISABLED"),
    (True, True, True, None, "FAULT"),
    (True, False, True, None, "AVOIDING"),
    (True, False, False, (1, 2), "AVOIDING"),
    (True, False, False, None, "DIRECT"),
])
def test_follow_status_is_derived_from_owned_motion_not_unowned_intent(enabled, fault, owned, memory, expected):
    n = SimpleNamespace(_enabled=enabled, _fault_latched=fault, _follow_nav2_avoidance=True,
                        _goal_unresolved=owned, _memory_goal=memory, _mode="NAV2_PENDING")
    assert PiracerFollowNode._behavior_status(n) == expected


from action_msgs.msg import GoalStatus
from test_acquire_path import acquire
from test_acquire_sequence import dispatch_leg, finish_leg, advance
from geometry_msgs.msg import Twist
import json


def test_acquire_localization_cancel_is_fault_after_real_terminal_stop(acquire):
    follower, goal = dispatch_leg(acquire)
    acquire._invalidate_localization("visual localization lost", acquire.clock.nanoseconds)
    assert acquire._operation["fault"] and follower.cancel_count == 1
    finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    assert acquire._operation is None
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "failed"


def test_acquire_disable_cancellation_is_not_fault(acquire):
    follower, goal = dispatch_leg(acquire)
    response = acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success
    assert not acquire._operation.get("fault", False)
    finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "canceled" and acquire._operation is None


def test_acquire_fault_during_terminal_stop_is_not_reported_as_success(acquire):
    follower, goal = dispatch_leg(acquire)
    follower.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED))
    acquire._invalidate_localization("visual localization lost", acquire.clock.nanoseconds)
    assert acquire._operation["fault"]
    acquire._on_command(Twist())
    for _ in range(22):
        advance(acquire, .1, goal)
    assert acquire._operation is None
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "failed"


def test_acquire_fault_during_handoff_cannot_enable_selected_follow(acquire):
    from test_acquire_state_integration import manager
    from test_state_machine import frame
    n = manager(acquire)
    n._sync_behavior_services()
    follower, goal = dispatch_leg(acquire)
    for i in range(3, 8):
        frame(n.machine, i, True, 100 + i / 10)
    assert n.machine.state == FOLLOW
    acquire._invalidate_localization("visual localization lost", acquire.clock.nanoseconds)
    n._sync_behavior_services()
    assert not n.enabled["follow"]
    finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    n._sync_behavior_services()
    assert not n.machine.enabled and not n.enabled["follow"]


@pytest.mark.parametrize("terminal_first", [False, True])
def test_acquire_disable_fault_ack_blocks_successor_before_delayed_result(acquire, terminal_first):
    from test_acquire_state_integration import manager
    from test_state_machine import frame
    n = manager(acquire)
    n._sync_behavior_services()
    # Unlike the synchronous fixture, hold the topic message until after the ACK.
    acquire._result_pub.publish.side_effect = None
    follower, goal = dispatch_leg(acquire)
    for i in range(3, 8):
        frame(n.machine, i, True, 100 + i / 10)
    acquire._invalidate_localization("visual localization lost", acquire.clock.nanoseconds)
    if terminal_first:
        finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    n._sync_behavior_services()
    assert not n.machine.enabled and not n.enabled["follow"]
    if not terminal_first:
        finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    n._sync_behavior_services()
    assert n.behavior_coordinator.confirmed["acquire"] is False
    assert n.owners[-1] == "stop" and ("follow", True) not in n.requests
    late = acquire._result_pub.publish.call_args.args[0]
    assert json.loads(late.data)["outcome"] == "failed"
    n._on_acquire_result(late)
    assert not n.machine.enabled and not n.enabled["follow"]


def test_clock_regression_cannot_lose_acquire_fault_when_attempt_id_is_cleared(acquire):
    from test_acquire_state_integration import manager
    n = manager(acquire)
    n._sync_behavior_services()
    follower, goal = dispatch_leg(acquire)
    acquire._result_pub.publish.side_effect = None
    n._last_sim_time_ns = acquire.clock.nanoseconds
    n._sim_epoch_start_ns = 0
    n._target_stamp = n._target_received = None
    n._follow_status = n._follow_status_received = None
    n.get_parameter = lambda _: SimpleNamespace(value=True)
    n.get_clock = acquire.get_clock
    acquire.clock.nanoseconds = 2_000_000_000
    assert StateManagerNode._freshness_now(n) is None
    assert n._acquire_attempt_id is None and not n.machine.active
    # A new detector epoch must not enable SEARCH before the old fault releases.
    n.machine.observe((2, 100), False, 2.1)
    n._sync_behavior_services()
    assert not n.machine.enabled and not n.enabled["search"]
    assert acquire._fault_latched and acquire._operation is not None
    finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    n._sync_behavior_services()
    assert ("search", True) not in n.requests and n.owners[-1] == "stop"


def test_validator_odom_tf_regression_is_fault_not_normal_path_rejection(acquire):
    from test_acquire_path import start_plan, finish_plan
    from nav_msgs.msg import Path
    from builtin_interfaces.msg import Time as Stamp
    import copy
    planner = start_plan(acquire)
    request = acquire._planner.goals[-1]
    path = Path(); path.header = copy.deepcopy(request.goal.header)
    path.poses = [copy.deepcopy(request.start), copy.deepcopy(request.goal)]
    finish_plan(acquire, planner, path)
    assert acquire._validation is not None
    acquire._tf.odom_tf.header.stamp = Stamp(sec=9, nanosec=900_000_000)
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert acquire._controller.goals == []
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "failed" and acquire._fault_latched


def test_shared_startup_setpoint_reaches_acquire_after_yaml_override():
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue
    from test_acquire_launch_parameters import launch_description
    description = launch_description("state_manager.launch.py")
    context = LaunchContext()
    context.launch_configurations.update(follow_target_distance_m="1.0", use_sim_time="true")
    nodes = [e for e in description.entities if isinstance(e, Node)]
    acquire_node = next(n for n in nodes if n._Node__node_name == "piracer_acquire")
    values = acquire_node._Node__parameters[-1]
    key = next(k for k in values if perform_substitutions(context, k) == "standoff_m")
    assert isinstance(values[key], ParameterValue)
    assert ParameterValue(values[key].value, value_type=float).evaluate(context) == 1.0
