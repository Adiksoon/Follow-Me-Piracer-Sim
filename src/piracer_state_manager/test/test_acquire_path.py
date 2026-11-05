from concurrent.futures import Future
from collections import deque
from types import MethodType, SimpleNamespace
from unittest.mock import Mock
import copy
import json
import math
from pathlib import Path as FilePath

import pytest
import tf2_geometry_msgs
import yaml
from action_msgs.msg import GoalInfo, GoalStatus
from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Time as Stamp
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import ComputePathToPose
from tf2_ros import TransformException
from unique_identifier_msgs.msg import UUID

from piracer_state_manager.acquire_geometry import observation_goal, path_length
from piracer_state_manager.acquire_node import AcquireNode


class FakeTF:
    def __init__(self):
        self.fail = False
        self.map_tf = TransformStamped()
        self.odom_tf = TransformStamped()
        for tf in (self.map_tf, self.odom_tf):
            tf.header.stamp = Stamp(sec=10)
            tf.transform.rotation.w = 1.0

    def lookup_transform(self, target, source, stamp):
        if self.fail:
            raise TransformException("missing TF")
        return self.odom_tf if target == "vo_odom" else self.map_tf

    def transform(self, point, frame):
        if self.fail:
            raise TransformException("missing TF")
        result = copy.deepcopy(point)
        result.header.frame_id = frame
        return result

    def clear(self):
        pass


class FakeAction:
    def __init__(self):
        self.ready = True
        self.goals = []
        self.response = Future()

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        if self.response.done():
            self.response = Future()
        self.goals.append(goal)
        return self.response


class FakeValidator:
    def __init__(self):
        self.ready = True
        self.requests = []
        self.response = Future()

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        if self.response.done():
            self.response = Future()
        self.requests.append(request)
        return self.response


class FakeHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancel_result = Future()
        self.cancel_count = 0
        self.goal_id = UUID(uuid=list(range(16)))

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_count += 1
        return self.cancel_result


def make_path(xs=(0.0, 0.5)):
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp = Stamp(sec=10)
    for x in xs:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(x)
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


@pytest.fixture
def acquire(monkeypatch):
    wall = [100.0]
    monkeypatch.setattr("piracer_state_manager.acquire_node.time.monotonic", lambda: wall[0])
    params = {
        "map_frame": "map", "odom_frame": "vo_odom", "base_frame": "base_link",
        "standoff_m": 2.2, "max_goal_distance_m": 0.5, "max_path_length_m": 1.0,
        "max_travel_distance_m": 1.2, "stop_margin_m": 0.15,
        "max_maneuver_time_s": 30.0, "observation_max_age_s": 3.0, "max_target_shift_m": 0.25,
        "localization_max_age_s": 1.0, "scan_max_age_s": 1.0, "plan_max_age_s": 2.0,
        "start_tolerance_m": 0.1, "start_yaw_tolerance_rad": 0.3, "end_tolerance_m": 0.15,
        "stop_linear_mps": 0.02, "stop_angular_radps": 0.05,
        "stop_confirmation_s": 2.0, "stop_position_tolerance_m": 0.01, "tf_future_lead_s": 0.2,
        "max_maneuvers": 1, "turn_radius_m": 0.75, "turn_angle_rad": 0.3,
        "goal_checker_id": "acquire_goal_checker",
    }
    clock = SimpleNamespace(nanoseconds=10_000_000_000)
    clock.to_msg = lambda: Stamp(sec=clock.nanoseconds // 10**9, nanosec=clock.nanoseconds % 10**9)
    point = PointStamped()
    point.header.frame_id = "vo_odom"
    point.header.stamp = Stamp(sec=9, nanosec=900_000_000)
    point.point.x = 4.0
    n = SimpleNamespace(
        _enabled=True, _epoch=1, _attempted=False, _operation=None, _validation=None,
        _sequence=None, _finished=False, _fault_latched=False,
        _attempt_id="fixture-attempt", _result_pub=Mock(), params=params,
        _target=point, _target_received=100.0, _target_turn_direction=1,
        _odom_info_history=deque([10_000_000_000], maxlen=10),
        _localization_barrier=None, _last_accepted_info_stamp=None,
        _last_accepted_map_tf_stamp=None, _localization_available=False,
        _localization_reason=None,
        _odom_stamp=10_000_000_000, _odom_received=100.0, _odom_velocity=(0.0, 0.0, 0.0),
        _last_command=(100.0, True),
        _scan_stamp=10_000_000_000, _scan_received=100.0, _scan_frame="camera",
        _last_now=None, _wait_reason=None, _tf=FakeTF(), _planner=FakeAction(),
        _controller=FakeAction(), _validator=FakeValidator(), _param=lambda name: params[name],
        get_clock=lambda: SimpleNamespace(now=lambda: clock), get_logger=Mock(return_value=Mock()),
        _stamp=AcquireNode._stamp, _fresh=AcquireNode._fresh, clock=clock, wall=wall,
    )
    for name in ("_now", "_wait", "_on_info", "_invalidate_localization", "_on_odom", "_on_command", "_on_scan", "_on_point",
                 "_localization_fresh", "_stationary", "_observation_usable", "_robot_tf", "_ready", "_set_enabled",
                 "_sequence_failure", "_track_travel", "_finish", "_after_stop",
                 "_reset_terminal_stop_window", "_terminal_stop_sample",
                 "_tick", "_send_action", "_goal_response", "_cancel", "_cancel_response",
                 "_goal_result", "_check_path", "_validated"):
        setattr(n, name, MethodType(getattr(AcquireNode, name), n))
    return n


def start_plan(n):
    n._tick()
    assert len(n._planner.goals) == 1
    handle = FakeHandle()
    n._planner.response.set_result(handle)
    return handle


def finish_plan(n, handle, path):
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED,
                                             result=SimpleNamespace(path=path)))


def start_follow(n, path=None):
    handle = start_plan(n)
    path = path or make_path()
    finish_plan(n, handle, path)
    assert len(n._validator.requests) == 1
    n._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert len(n._controller.goals) == 1
    return path


def test_approach_standoff_and_direction():
    def approach(robot, target):
        return observation_goal(robot, target, 2.2, 0.5, 0, 1, 0.75, 0.3)

    assert approach((0, 0, 0), (4, 0)) == pytest.approx((0.5, 0, 0))
    assert approach((0, 0, 0), (2.4, 0)) == pytest.approx((0.2, 0, 0))
    assert approach((0, 0, 0), (2, 0)) is None
    assert approach((0, 0, 0), (-4, 0)) is None


def test_valid_plan_follows_the_identical_path_once(acquire):
    handle = start_plan(acquire)
    goal_request = acquire._planner.goals[0]
    frozen_stamp = acquire._sequence["observation"][0]
    # A newer same-position detection must not replace the admitted target's stamp.
    acquire._target.header.stamp = Stamp(sec=10)
    path = make_path()
    path.header = copy.deepcopy(goal_request.goal.header)
    path.poses = [copy.deepcopy(goal_request.start), copy.deepcopy(goal_request.goal)]
    finish_plan(acquire, handle, path)
    assert len(acquire._validator.requests) == 1
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert len(acquire._controller.goals) == 1
    assert acquire._validator.requests[0].path is path
    goal = acquire._controller.goals[0]
    assert goal.path is path
    assert goal.controller_id == "FollowPath" and goal.goal_checker_id == "acquire_goal_checker"
    assert acquire._sequence["observation"][0] == frozen_stamp
    assert frozen_stamp != AcquireNode._stamp(acquire._target.header.stamp)
    acquire._tick()
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == 1




def test_long_detour_rejected_even_with_close_goal(acquire):
    handle = start_plan(acquire)
    path = make_path((0, 0.8, 0.0, 0.5))
    assert path_length(path, "map") == pytest.approx(2.1)
    finish_plan(acquire, handle, path)
    assert not acquire._validator.requests and not acquire._controller.goals


def test_path_length_includes_height(acquire):
    handle = start_plan(acquire)
    path = make_path((0, 0.25, 0.5))
    path.poses[1].pose.position.z = 1.0
    assert path_length(path, "map") > 1.0
    finish_plan(acquire, handle, path)
    assert not acquire._validator.requests and not acquire._controller.goals


@pytest.mark.parametrize("case", ["empty", "nan", "frame", "orientation"])
def test_invalid_path_is_rejected(acquire, case):
    handle = start_plan(acquire)
    path = make_path()
    if case == "empty":
        path.poses = []
    elif case == "nan":
        path.poses[1].pose.position.x = math.nan
    elif case == "frame":
        path.poses[1].header.frame_id = "other"
    else:
        path.poses[1].pose.orientation.w = 0.0
    finish_plan(acquire, handle, path)
    assert not acquire._validator.requests and not acquire._controller.goals


def test_robot_moves_after_planning_so_path_cannot_start(acquire):
    handle = start_plan(acquire)
    finish_plan(acquire, handle, make_path())
    acquire._tf.map_tf.transform.translation.x = 0.2
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals


def test_target_moves_after_plan_so_path_cannot_start(acquire):
    handle = start_plan(acquire)
    finish_plan(acquire, handle, make_path())
    acquire._target.point.x += 0.30
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals


def test_nav2_collision_validation_rejects_path(acquire):
    handle = start_plan(acquire)
    finish_plan(acquire, handle, make_path())
    acquire._validator.response.set_result(SimpleNamespace(is_valid=False))
    assert not acquire._controller.goals


def test_plan_becomes_stale_while_waiting_for_validator(acquire):
    handle = start_plan(acquire)
    finish_plan(acquire, handle, make_path())
    acquire.clock.nanoseconds = 12_500_000_000
    stamp = Stamp(sec=12, nanosec=500_000_000)
    acquire._tf.map_tf.header.stamp = stamp
    acquire._tf.odom_tf.header.stamp = stamp
    acquire._odom_stamp = acquire._scan_stamp = 12_500_000_000
    acquire._odom_info_history = deque([12_500_000_000], maxlen=10)
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals


def test_follow_path_rejection_does_not_start_another_plan(acquire):
    start_follow(acquire)
    acquire._controller.response.set_result(FakeHandle(accepted=False))
    acquire._tick()
    assert acquire._operation is None and len(acquire._planner.goals) == 1
    assert len(acquire._controller.goals) == 1


def test_failed_planning_never_sends_follow(acquire):
    handle = start_plan(acquire)
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
    acquire._tick()
    assert not acquire._validator.requests and not acquire._controller.goals
    assert len(acquire._planner.goals) == 1


def test_disable_during_planning_blocks_late_result(acquire):
    acquire._tick()
    response = acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success
    handle = FakeHandle()
    acquire._planner.response.set_result(handle)
    assert handle.cancel_count == 1
    finish_plan(acquire, handle, make_path())
    assert not acquire._validator.requests and not acquire._controller.goals
    assert acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success


def test_disable_during_validation_blocks_late_service_response(acquire):
    handle = start_plan(acquire)
    finish_plan(acquire, handle, make_path())
    assert acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals


def stationary_odom(ns):
    msg = Odometry()
    msg.header.frame_id = "vo_odom"
    msg.child_frame_id = "base_link"
    msg.header.stamp = Stamp(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    return msg


def advance_terminal_samples(node, count=21):
    for _ in range(count):
        ns = node.clock.nanoseconds + 100_000_000
        node.clock.nanoseconds = ns
        node.wall[0] += 0.1
        for transform in (node._tf.map_tf, node._tf.odom_tf):
            transform.header.stamp = Stamp(sec=ns // 1_000_000_000,
                                            nanosec=ns % 1_000_000_000)
        node._odom_info_history.append(ns)
        node._scan_stamp, node._scan_received = ns, node.wall[0]
        node._on_odom(stationary_odom(ns))


@pytest.mark.parametrize(
    ("offset_ns", "stationary"),
    [(0, True), (37_000_000, True), (200_000_000, True),
     (200_000_001, False), (-1_000_000_000, True), (-1_000_000_001, False)],
)
def test_stationarity_odometry_timestamp_lead_is_bounded(acquire, offset_ns, stationary):
    acquire._odom_stamp = None
    acquire._on_odom(stationary_odom(acquire.clock.nanoseconds + offset_ns))
    assert acquire._stationary(acquire.clock.nanoseconds) is stationary


@pytest.mark.parametrize(
    ("offset_ns", "follows"),
    [(37_000_000, True), (200_000_000, True), (200_000_001, False)],
)
def test_new_odometry_during_path_validation_has_bounded_lead(acquire, offset_ns, follows):
    handle = start_plan(acquire)
    path = make_path()
    finish_plan(acquire, handle, path)
    acquire._on_odom(stationary_odom(acquire.clock.nanoseconds + offset_ns))
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert bool(acquire._controller.goals) is follows
    if follows:
        assert acquire._controller.goals[0].path is path


@pytest.mark.parametrize(
    ("velocity", "receipt_age_s"),
    [((0.021, 0.0, 0.0), 0.0), ((0.0, 0.021, 0.0), 0.0),
     ((0.0, 0.0, 0.051), 0.0), ((0.0, 0.0, 0.0), 1.001)],
)
def test_future_odometry_cannot_hide_motion_or_stale_receipt(acquire, velocity, receipt_age_s):
    handle = start_plan(acquire)
    finish_plan(acquire, handle, make_path())
    odom = stationary_odom(acquire.clock.nanoseconds + 37_000_000)
    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = velocity
    acquire._on_odom(odom)
    acquire._odom_received -= receipt_age_s
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals


def test_disable_follow_waits_for_terminal_result_and_two_second_pose_window(acquire):
    start_follow(acquire)
    response = acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    assert handle.cancel_count == 1
    handle.cancel_result.set_result(SimpleNamespace(return_code=0, goals_canceling=[SimpleNamespace(goal_id=handle.goal_id)]))
    acquire.get_logger().info.assert_any_call("Acquire cancellation accepted; awaiting terminal result")
    assert not acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    acquire.clock.nanoseconds = 10_100_000_000
    for transform in (acquire._tf.map_tf, acquire._tf.odom_tf):
        transform.header.stamp = Stamp(sec=10, nanosec=100_000_000)
    acquire._odom_info_history.append(10_100_000_000)
    acquire._on_odom(stationary_odom(10_100_000_000))
    assert not acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success
    advance_terminal_samples(acquire)
    assert acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success


def test_no_tf_means_stop_cannot_be_confirmed(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    acquire.clock.nanoseconds = 10_200_000_000
    acquire._tf.fail = True
    acquire._on_odom(stationary_odom(10_100_000_000))
    acquire._on_odom(stationary_odom(10_200_000_000))
    assert not acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success


def test_repeated_old_tf_cannot_confirm_stop(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    acquire.clock.nanoseconds = 10_400_000_000
    acquire._on_odom(stationary_odom(10_100_000_000))
    acquire._on_odom(stationary_odom(10_400_000_000))
    assert not acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success


def test_moving_tf_cannot_confirm_stop(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    acquire.clock.nanoseconds = 10_400_000_000
    acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=100_000_000)
    acquire._on_odom(stationary_odom(10_100_000_000))
    acquire.wall[0] += 0.3
    acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=400_000_000)
    acquire._tf.odom_tf.transform.translation.x = 0.02
    acquire._on_odom(stationary_odom(10_400_000_000))
    assert not acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success


def test_rejected_cancel_does_not_release_action_ownership(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    handle.cancel_result.set_result(SimpleNamespace(return_code=1, goals_canceling=[]))
    acquire.get_logger().warning.assert_any_call("Acquire cancellation not acknowledged; awaiting terminal result")
    assert not acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success


@pytest.mark.parametrize(
    ("goal_indices", "return_code", "acknowledged"),
    [([0], 0, True), ([1], 0, False), ([1, 0], 0, True),
     ([], 0, False), ([0], 1, False)],
)
def test_cancel_response_checks_full_native_uuid(acquire, goal_indices, return_code, acknowledged):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    operation = acquire._operation
    ids = [UUID(uuid=list(range(16))), UUID(uuid=list(range(15)) + [255])]
    response = CancelGoal.Response(
        return_code=return_code,
        goals_canceling=[GoalInfo(goal_id=ids[i]) for i in goal_indices],
    )
    handle.cancel_result.set_result(response)
    if acknowledged:
        acquire.get_logger().info.assert_any_call("Acquire cancellation accepted; awaiting terminal result")
    else:
        acquire.get_logger().warning.assert_any_call("Acquire cancellation not acknowledged; awaiting terminal result")
    assert acquire._operation is operation
    acquire.get_logger().error.assert_not_called()


def test_late_cancel_ack_after_terminal_result_does_not_release_or_restart(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    operation = acquire._operation
    handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    handle.cancel_result.set_result(CancelGoal.Response(
        return_code=0, goals_canceling=[GoalInfo(goal_id=UUID(uuid=list(range(16))))],
    ))
    acquire.get_logger().info.assert_any_call("Acquire cancellation accepted; awaiting terminal result")
    assert operation["terminal_follow"] is True
    assert acquire._operation is operation
    assert not acquire._set_enabled(SimpleNamespace(data=True), SimpleNamespace()).success
    acquire._tick()
    assert not acquire._enabled
    assert acquire._operation is operation
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == 1


@pytest.mark.parametrize("start_new_plan", [False, True])
def test_late_cancel_ack_for_released_goal_does_not_change_current_operation(acquire, start_new_plan):
    start_follow(acquire)
    old_handle = FakeHandle()
    acquire._controller.response.set_result(old_handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    old_handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    acquire.clock.nanoseconds = 10_100_000_000
    for transform in (acquire._tf.map_tf, acquire._tf.odom_tf):
        transform.header.stamp = Stamp(sec=10, nanosec=100_000_000)
    acquire._odom_info_history.append(10_100_000_000)
    acquire._on_odom(stationary_odom(10_100_000_000))
    advance_terminal_samples(acquire)
    assert acquire._operation is None
    if start_new_plan:
        assert acquire._set_enabled(SimpleNamespace(data=True), SimpleNamespace()).success
        acquire._planner = FakeAction()
        new_handle = start_plan(acquire)
        new_handle.goal_id = UUID(uuid=[255] * 16)
    current = acquire._operation
    goal_counts = (len(acquire._planner.goals), len(acquire._controller.goals))
    old_handle.cancel_result.set_result(CancelGoal.Response(
        return_code=0, goals_canceling=[GoalInfo(goal_id=UUID(uuid=list(range(16))))],
    ))
    acquire._tick()
    assert acquire._operation is current
    assert acquire._enabled is start_new_plan
    assert (len(acquire._planner.goals), len(acquire._controller.goals)) == goal_counts
    if current is not None:
        assert current["handle"] is new_handle


def test_distance_limit_uses_odom_and_requests_cancel_with_margin(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire.clock.nanoseconds = 10_100_000_000
    acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=100_000_000)
    acquire._tf.odom_tf.transform.translation.x = 1.06
    acquire._tick()
    assert handle.cancel_count == 1
    assert acquire._sequence["traveled"] == pytest.approx(1.06)


def test_distance_is_sum_of_segments_not_distance_from_start(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    for offset, x in ((100_000_000, 0.4), (200_000_000, 0.7), (300_000_000, 0.3)):
        acquire.clock.nanoseconds = 10_000_000_000 + offset
        acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=offset)
        acquire._tf.odom_tf.transform.translation.x = x
        acquire._tick()
    assert acquire._sequence["traveled"] == pytest.approx(1.1)
    assert handle.cancel_count == 1


def test_distance_keeps_accumulating_after_disable_until_result(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    acquire.clock.nanoseconds = 10_100_000_000
    acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=100_000_000)
    acquire._tf.odom_tf.transform.translation.x = 0.3
    acquire._tick()
    assert acquire._sequence["traveled"] == pytest.approx(0.3)


def test_sequence_timeout_requests_cancel(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire.wall[0] = 130.0
    acquire._odom_received = acquire._scan_received = 130.0
    acquire._tick()
    assert handle.cancel_count == 1
    assert acquire._operation["cancel_reason"] == "sequence time limit"


@pytest.mark.parametrize("missing", ["target", "tf", "nav2", "stale_target", "localization", "scan"])
def test_missing_or_stale_inputs_do_not_plan(acquire, missing):
    if missing == "target":
        acquire._target = None
    elif missing == "tf":
        acquire._tf.fail = True
    elif missing == "nav2":
        acquire._planner.ready = False
    elif missing == "stale_target":
        acquire._target.header.stamp = Stamp(sec=1)
    elif missing == "localization":
        acquire._odom_info_history.clear()
    else:
        acquire._scan_stamp = None
    acquire._tick()
    assert not acquire._planner.goals


@pytest.mark.parametrize(
    ("scan_offset_ns", "plans"),
    [(50_000_000, True), (200_000_000, True), (201_000_000, False),
     (-1_100_000_000, False)],
)
def test_scan_timestamp_lead_is_bounded(acquire, scan_offset_ns, plans):
    acquire._scan_stamp = acquire.clock.nanoseconds + scan_offset_ns
    acquire._tick()
    assert bool(acquire._planner.goals) is plans


def test_small_future_scan_survives_plan_validation(acquire):
    acquire._scan_stamp = acquire.clock.nanoseconds + 50_000_000
    start_follow(acquire)
    assert len(acquire._controller.goals) == 1


def test_expired_recovery_after_plan_does_not_start_follow(acquire):
    handle = start_plan(acquire)
    acquire.wall[0] += 31.0
    acquire.clock.nanoseconds = 14_000_000_000
    acquire._tf.map_tf.header.stamp = Stamp(sec=14)
    acquire._tf.odom_tf.header.stamp = Stamp(sec=14)
    acquire._odom_stamp = acquire._scan_stamp = 14_000_000_000
    acquire._odom_info_history = deque([14_000_000_000], maxlen=10)
    finish_plan(acquire, handle, make_path())
    assert not acquire._validator.requests and not acquire._controller.goals


def test_observation_is_remembered_while_disabled(acquire):
    acquire._enabled = False
    point = PointStamped()
    point.header.frame_id = "camera"
    point.header.stamp = Stamp(sec=9, nanosec=950_000_000)
    point.point.x = 4.0
    acquire._on_point(point)
    assert acquire._target.header.frame_id == "vo_odom"
    assert acquire._target.header.stamp.nanosec == 950_000_000
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert acquire._target.header.stamp.nanosec == 950_000_000


@pytest.mark.parametrize("optical_x, expected", [(-0.3, 1), (0.3, -1), (0.0, 1)])
def test_last_observed_side_uses_base_axes_and_original_stamp_while_disabled(acquire, optical_x, expected):
    acquire._enabled = False
    point = PointStamped()
    point.header.frame_id = "camera_optical_frame"
    point.header.stamp = Stamp(sec=9, nanosec=950_000_000)
    point.point.x, point.point.z = optical_x, 4.0

    base_from_camera = TransformStamped()
    base_from_camera.header.frame_id = "base_link"
    base_from_camera.header.stamp = point.header.stamp
    # Optical right/down/forward becomes robot forward/left/up.
    q = base_from_camera.transform.rotation
    q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    odom_from_base = TransformStamped()
    odom_from_base.header.frame_id = "vo_odom"
    odom_from_base.header.stamp = point.header.stamp
    odom_from_base.transform.translation.y = 5.0
    odom_from_base.transform.rotation.w = 1.0
    requests = []

    def transform(message, frame):
        requests.append((frame, copy.deepcopy(message.header.stamp)))
        in_base = tf2_geometry_msgs.do_transform_point(message, base_from_camera)
        return (in_base if frame == "base_link" else
                tf2_geometry_msgs.do_transform_point(in_base, odom_from_base))

    acquire._tf.transform = transform
    acquire._on_point(point)
    assert acquire._target.point.y > 0  # Global Y alone cannot distinguish these sides.
    assert acquire._target_turn_direction == expected
    assert acquire._target.header.stamp == point.header.stamp
    assert requests == [("vo_odom", point.header.stamp), ("base_link", point.header.stamp)]
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert acquire._target_turn_direction == expected


@pytest.mark.parametrize("failure", ["missing_base_tf", "nonfinite_base_point"])
def test_invalid_base_transform_preserves_whole_previous_observation(acquire, failure):
    previous = acquire._target
    receipt = acquire._target_received
    acquire._target_turn_direction = -1
    point = copy.deepcopy(previous)
    point.header.stamp = Stamp(sec=9, nanosec=950_000_000)
    point.point.y = 0.2

    def transform(message, frame):
        result = copy.deepcopy(message)
        result.header.frame_id = frame
        if frame == "base_link":
            if failure == "missing_base_tf":
                raise TransformException("missing historical base TF")
            result.point.y = math.nan
        return result

    acquire._tf.transform = transform
    acquire._on_point(point)
    assert acquire._target is previous
    assert acquire._target_received == receipt
    assert acquire._target_turn_direction == -1
    assert acquire._target.header.stamp == Stamp(sec=9, nanosec=900_000_000)


def test_localization_uses_fresh_info_and_tf_but_stationarity_keeps_odom_gate(acquire):
    header = lambda sec, nanosec=0: SimpleNamespace(stamp=Stamp(sec=sec, nanosec=nanosec))
    acquire._on_info(SimpleNamespace(lost=True, header=header(10)))
    assert acquire._target is None
    acquire._on_info(SimpleNamespace(lost=False, header=header(9, 900_000_000)))
    assert not acquire._odom_info_history
    acquire.clock.nanoseconds = 10_200_000_000
    acquire._on_info(SimpleNamespace(lost=False, header=header(10, 100_000_000)))
    acquire._tf.map_tf.header.stamp = Stamp(sec=10, nanosec=150_000_000)
    acquire._odom_stamp = acquire._odom_received = acquire._odom_velocity = None
    assert acquire._localization_fresh(acquire.clock.nanoseconds)
    assert not acquire._stationary(acquire.clock.nanoseconds)
    acquire._on_odom(stationary_odom(10_150_000_000))
    assert acquire._stationary(acquire.clock.nanoseconds)


def test_close_target_does_not_send_plan(acquire):
    acquire._target.point.x = 2.0
    acquire._tick()
    assert not acquire._planner.goals


def test_configured_nav2_ids_and_ackermann_restrictions_match_client():
    src = FilePath(__file__).parents[2]
    config = yaml.safe_load((src / "piracer_navigation/config/nav2_params.yaml").read_text())
    planner = config["planner_server"]["ros__parameters"]
    controller = config["controller_server"]["ros__parameters"]
    assert "Hybrid" in planner["planner_plugins"]
    assert planner["Hybrid"]["motion_model_for_search"] == "DUBIN"
    assert planner["Hybrid"]["allow_reverse_expansion"] is False
    assert "FollowPath" in controller["controller_plugins"]
    assert "acquire_goal_checker" in controller["goal_checker_plugins"]
    assert controller["acquire_goal_checker"]["plugin"] == "nav2_controller::PositionGoalChecker"
    assert controller["acquire_goal_checker"]["xy_goal_tolerance"] == 0.05
    acquire_config = yaml.safe_load((src / "piracer_state_manager/config/acquire.yaml").read_text())["piracer_acquire"]["ros__parameters"]
    assert acquire_config["stop_confirmation_s"] == 2.0
    assert acquire_config["stop_position_tolerance_m"] == 0.01
    assert acquire_config["stop_linear_mps"] == 0.02
    assert acquire_config["stop_angular_radps"] == 0.05
    assert acquire_config["turn_radius_m"] >= planner["Hybrid"]["minimum_turning_radius"]
    assert controller["FollowPath"]["use_rotate_to_heading"] is False
    assert controller["FollowPath"]["allow_reversing"] is False
    assert controller["FollowPath"]["use_collision_detection"] is True



def validation_at_dispatch_pose(node, x, stamp_ns=10_200_000_000):
    handle = start_plan(node)
    finish_plan(node, handle, make_path())
    assert node._validation is not None
    node.clock.nanoseconds = stamp_ns
    node.wall[0] += 0.2
    for transform in (node._tf.map_tf, node._tf.odom_tf):
        transform.header.stamp = Stamp(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9)
        transform.transform.translation.x = x
    node._odom_info_history.append(stamp_ns)
    return handle


def test_tf_regression_relative_to_dispatch_pose_cancels(acquire):
    validation_at_dispatch_pose(acquire, 0.05)
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    follower = FakeHandle()
    acquire._controller.response.set_result(follower)
    assert acquire._sequence["last_tf_stamp"] == 10_200_000_000
    acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=100_000_000)
    acquire._tick()
    assert follower.cancel_count == 1
    assert acquire._operation["cancel_reason"] == "odometry TF timestamp regressed"


def test_direction_reversal_across_dispatch_counts_both_segments(acquire):
    validation_at_dispatch_pose(acquire, 0.08)
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert acquire._sequence["traveled"] == pytest.approx(0.08)
    acquire.clock.nanoseconds = 10_300_000_000
    acquire._tf.odom_tf.header.stamp = Stamp(sec=10, nanosec=300_000_000)
    acquire._tf.odom_tf.transform.translation.x = 0.0
    acquire._tick()
    assert acquire._sequence["traveled"] == pytest.approx(0.16)
    for _ in range(3):
        acquire._tick()
    assert acquire._sequence["traveled"] == pytest.approx(0.16)


def test_dispatch_sample_crossing_travel_limit_prevents_follow(acquire):
    validation_at_dispatch_pose(acquire, 0.08)
    acquire._sequence["traveled"] = 1.0
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals
    assert acquire._sequence["traveled"] == pytest.approx(1.08)
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "blocked"
    assert "sequence travel limit with braking margin" in result["reason"]


def test_path_validation_uses_budget_after_dispatch_sample(acquire):
    validation_at_dispatch_pose(acquire, 0.08)
    acquire._sequence["traveled"] = 0.5
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals
    assert acquire._sequence["traveled"] == pytest.approx(0.58)
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert "remaining sequence travel budget" in result["reason"]


def test_sequence_deadline_during_validation_does_not_dispatch(acquire):
    validation_at_dispatch_pose(acquire, 0.0)
    acquire._sequence["started"] -= 31.0
    acquire._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert not acquire._controller.goals
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert "sequence time limit" in result["reason"]



def test_tf_failure_reason_wins_at_deadline_and_stays_latched(acquire):
    start_follow(acquire)
    handle = FakeHandle()
    acquire._controller.response.set_result(handle)
    acquire.wall[0] = 131.0
    acquire._odom_received = acquire._scan_received = 131.0
    acquire._tf.odom_tf.header.stamp = Stamp(sec=9, nanosec=900_000_000)
    acquire._tick()
    assert acquire._operation["cancel_reason"] == "odometry TF timestamp regressed"
    acquire._tick()
    assert handle.cancel_count == 1
    assert acquire._operation["cancel_reason"] == "odometry TF timestamp regressed"
