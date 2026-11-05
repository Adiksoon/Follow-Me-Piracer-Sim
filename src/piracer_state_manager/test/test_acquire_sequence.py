"""Offline sequence tests: real messages, deterministic clocks and Nav2 futures."""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import tf2_geometry_msgs
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time as Stamp
from nav_msgs.msg import Path
from geometry_msgs.msg import TransformStamped, Twist

from piracer_state_manager.acquire_geometry import observation_goal
from test_acquire_path import acquire, FakeHandle, stationary_odom, finish_plan, start_follow, start_plan


def advance(node, seconds, pose=None, future_tf_ns=0, twist=(0.0, 0.0, 0.0)):
    node.clock.nanoseconds += round(seconds * 1e9)
    node.wall[0] += seconds
    now = node.clock.nanoseconds
    for transform in (node._tf.map_tf, node._tf.odom_tf):
        stamp = now + future_tf_ns
        transform.header.stamp = Stamp(sec=stamp // 10**9, nanosec=stamp % 10**9)
        if pose is not None:
            transform.transform.translation.x, transform.transform.translation.y = pose[:2]
            transform.transform.rotation.z = math.sin(pose[2] / 2)
            transform.transform.rotation.w = math.cos(pose[2] / 2)
    node._odom_info_history.append(now)
    node._scan_stamp, node._scan_received = now, node.wall[0]
    odom = stationary_odom(now)
    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = twist
    node._on_odom(odom)


def advance_cached_odom(node, seconds=0.1):
    node.clock.nanoseconds += round(seconds * 1e9)
    node.wall[0] += seconds
    now = node.clock.nanoseconds
    node._odom_info_history.append(now)
    node._scan_stamp, node._scan_received = now, node.wall[0]
    node._on_odom(stationary_odom(now))


def dispatch_leg(node):
    count = len(node._planner.goals)
    node._tick()
    assert len(node._planner.goals) == count + 1
    planner = FakeHandle()
    node._planner.response.set_result(planner)
    request = node._planner.goals[-1]
    path = Path()
    path.header = copy.deepcopy(request.goal.header)
    path.poses = [copy.deepcopy(request.start), copy.deepcopy(request.goal)]
    finish_plan(node, planner, path)
    node._validator.response.set_result(SimpleNamespace(is_valid=True))
    assert node._controller.goals[-1].path is path
    follower = FakeHandle()
    node._controller.response.set_result(follower)
    q = request.goal.pose.orientation
    goal = (request.goal.pose.position.x, request.goal.pose.position.y,
            math.atan2(2*q.w*q.z, 1-2*q.z*q.z))
    return follower, goal


def finish_leg(node, follower, goal, status=GoalStatus.STATUS_SUCCEEDED, future_tf_ns=0, zero_command=True):
    advance(node, 4.0, goal)
    node._tick()
    operation = node._operation
    follower.result.set_result(SimpleNamespace(status=status))
    if zero_command:
        node._on_command(Twist())
    assert node._operation is operation
    advance(node, 0.1, future_tf_ns=future_tf_ns)
    assert node._operation is operation
    for _ in range(21):
        advance(node, 0.1, future_tf_ns=future_tf_ns)
    return operation


def begin_terminal_cancel(node):
    follower, goal = dispatch_leg(node)
    advance(node, 4.0, goal)
    follower.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    node._on_command(Twist())
    return goal


def restart_attempt(node):
    assert node._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success
    assert node._set_enabled(SimpleNamespace(data=True), SimpleNamespace()).success
    node._target.header.stamp = node.clock.to_msg()
    node._target_received = node.wall[0]
    for transform in (node._tf.map_tf, node._tf.odom_tf):
        transform.header.stamp = node.clock.to_msg()
    node._odom_info_history.clear()
    node._odom_info_history.append(node.clock.nanoseconds)
    node._scan_stamp, node._scan_received = node.clock.nanoseconds, node.wall[0]


def test_geometry_reserves_clearance_and_alternates_forward_turns():
    robot = (0.0, 0.0, 0.0)
    target = (2.84, 0.0)
    turns = []
    for index in range(3):
        goal = observation_goal(robot, target, 2.2, 0.5, index, 3, 0.75, 0.3)
        assert goal is not None
        dx, dy = goal[0] - robot[0], goal[1] - robot[1]
        assert 0.1 < math.hypot(dx, dy) <= 0.5
        assert dx * math.cos(robot[2]) + dy * math.sin(robot[2]) > 0
        assert math.dist(goal[:2], target) >= 2.2
        turn = math.atan2(math.sin(goal[2]-robot[2]), math.cos(goal[2]-robot[2]))
        turns.append(turn)
        if index:
            # Recover the physical arc sweep from its displacement chord,
            # independently of the requested final orientation.
            chord_turn = math.atan2(math.sin(math.atan2(dy, dx) - robot[2]),
                                    math.cos(math.atan2(dy, dx) - robot[2]))
            arc_turn = 2 * chord_turn
            side = 1 if arc_turn > 0 else -1
            center = (robot[0]-side*0.75*math.sin(robot[2]),
                      robot[1]+side*0.75*math.cos(robot[2]))
            assert math.dist(center, goal[:2]) == pytest.approx(0.75)
            assert 0.75*abs(arc_turn) <= 0.5
        robot = goal
    assert turns[0] == pytest.approx(0)
    assert turns[1] > 0 > turns[2]


def test_recorded_second_arc_keeps_xy_and_requests_chord_heading():
    robot = (0.17163683474063873, 0.014778018929064274, 0.2105276514251448)
    target = (2.840, 0.004)
    goal = observation_goal(robot, target, 2.2, 0.5, 1, 3, 0.75, 0.3)

    distance = math.hypot(target[0] - robot[0], target[1] - robot[1])
    step = min(0.5, (distance - 2.2) / 2)
    end_yaw = robot[2] + 0.3
    expected_x = robot[0] + 0.75 * (math.sin(end_yaw) - math.sin(robot[2]))
    expected_y = robot[1] + 0.75 * (math.cos(robot[2]) - math.cos(end_yaw))
    expected_heading = math.atan2(expected_y - robot[1], expected_x - robot[0])

    assert step / 0.75 > 0.3  # The historical case does not clip its arc angle.
    assert goal[:2] == pytest.approx((expected_x, expected_y), abs=1e-12)
    assert goal[:2] == pytest.approx((0.38138314992394917, 0.09385350818279334), abs=1e-12)
    assert goal[2] == pytest.approx(expected_heading, abs=1e-12)
    assert goal[2] == pytest.approx(0.3605276514251448, abs=1e-12)
    assert math.hypot(goal[0] - robot[0], goal[1] - robot[1]) <= 0.5
    assert (goal[0] - robot[0]) * math.cos(robot[2]) + (goal[1] - robot[1]) * math.sin(robot[2]) > 0


@pytest.mark.parametrize("robot, expected", [
    ((0.628598690032959, -0.028608115389943123, 0.17610561069745545),
     (0.852679511189536, -0.022757019468037876, -0.04889438930254453)),
    ((0.5223232507705688, -0.024924803525209427, 0.0818040017047737),
     (0.7459594081714678, -0.04019958131912771, -0.14319599829522628)),
])
def test_recorded_final_goals_keep_xy_and_face_between_chord_and_tangent(robot, expected):
    goal = observation_goal(robot, (4.0, 0.0), 2.2, 0.5, 2, 3, 0.75, 0.3)
    assert goal == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("yaw", [math.pi - 0.02, -math.pi + 0.02])
def test_capped_final_heading_wraps_with_rotated_forward_geometry(yaw):
    robot = (1.2, -0.7, yaw)
    target = (robot[0] + 5 * math.cos(yaw), robot[1] + 5 * math.sin(yaw))
    goal = observation_goal(robot, target, 2.2, 0.15, 2, 3, 0.75, 0.3)
    expected_heading = math.atan2(math.sin(yaw - 0.15), math.cos(yaw - 0.15))
    assert goal[2] == pytest.approx(expected_heading, abs=1e-12)
    assert math.dist(robot[:2], goal[:2]) == pytest.approx(2 * 0.75 * math.sin(0.1))
    assert (goal[0] - robot[0]) * math.cos(yaw) + (goal[1] - robot[1]) * math.sin(yaw) > 0


def test_last_two_arcs_mirror_in_rotated_local_coordinates_with_step_clipping():
    robot = (1.2, -0.7, 0.83)
    target = (robot[0] + 5 * math.cos(robot[2]), robot[1] + 5 * math.sin(robot[2]))

    def local_goal(index, max_step):
        goal = observation_goal(robot, target, 2.2, max_step, index, 3, 0.75, 0.3)
        assert goal is not None
        dx, dy = goal[0] - robot[0], goal[1] - robot[1]
        x = math.cos(robot[2]) * dx + math.sin(robot[2]) * dy
        y = -math.sin(robot[2]) * dx + math.cos(robot[2]) * dy
        heading = math.atan2(math.sin(goal[2] - robot[2]), math.cos(goal[2] - robot[2]))
        return x, y, heading

    second = local_goal(1, 0.5)
    third = local_goal(2, 0.5)
    assert second[0] == pytest.approx(third[0], abs=1e-12)
    assert second[1] == pytest.approx(-third[1], abs=1e-12)
    assert third[2] == pytest.approx(-1.5 * second[2], abs=1e-12)
    assert math.hypot(*second[:2]) == pytest.approx(2 * 0.75 * math.sin(0.3 / 2), abs=1e-12)

    clipped_second = local_goal(1, 0.15)
    clipped_third = local_goal(2, 0.15)
    assert clipped_second[0] == pytest.approx(clipped_third[0], abs=1e-12)
    assert clipped_second[1] == pytest.approx(-clipped_third[1], abs=1e-12)
    assert clipped_third[2] == pytest.approx(-1.5 * clipped_second[2], abs=1e-12)
    assert math.hypot(*clipped_second[:2]) == pytest.approx(2 * 0.75 * math.sin(0.2 / 2), abs=1e-12)


@pytest.mark.parametrize("target", [(2.2, 0), (2.25, 0), (-4, 0)])
def test_geometry_does_not_force_motion_without_forward_clearance(target):
    assert observation_goal((0, 0, 0), target, 2.2, 0.5, 0, 3, 0.75, 0.3) is None


@pytest.mark.parametrize("yaw", [0.0, 0.83, math.pi - 0.02, -math.pi + 0.02])
@pytest.mark.parametrize("max_step", [0.5, 0.15])
@pytest.mark.parametrize("index", [1, 2])
def test_directional_arcs_mirror_and_preserve_forward_clearance(yaw, max_step, index):
    robot = (1.2, -0.7, yaw)
    target = (robot[0] + 5 * math.cos(yaw), robot[1] + 5 * math.sin(yaw))
    endpoints = []
    for first_turn in (1, -1):
        goal = observation_goal(robot, target, 2.2, max_step, index, 3, 0.75, 0.3, first_turn)
        dx, dy = goal[0] - robot[0], goal[1] - robot[1]
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading = math.atan2(math.sin(goal[2] - yaw), math.cos(goal[2] - yaw))
        expected_side = first_turn if index == 1 else -first_turn
        assert forward > 0 and lateral * expected_side > 0 and heading * expected_side > 0
        assert math.hypot(dx, dy) <= max_step
        assert math.dist(goal[:2], target) >= 2.2
        assert -math.pi <= goal[2] <= math.pi
        endpoints.append((forward, lateral, heading))
    assert endpoints[0] == pytest.approx((endpoints[1][0], -endpoints[1][1], -endpoints[1][2]))


@pytest.mark.parametrize("count", [1, 3])
def test_turn_direction_does_not_change_initial_approach(count):
    args = ((1.0, -2.0, 0.4), (5.0, 1.0), 2.2, 0.5, 0, count, 0.75, 0.3)
    assert observation_goal(*args, 1) == observation_goal(*args, -1)


@pytest.mark.parametrize("first_turn", [0, 2, -2, None])
def test_invalid_first_turn_is_rejected(first_turn):
    with pytest.raises(ValueError, match="first turn"):
        observation_goal((0, 0, 0), (4, 0), 2.2, 0.5, 1, 3, 0.75, 0.3, first_turn)


def test_observation_side_does_not_follow_robot_rotation_before_activation(acquire):
    point = copy.deepcopy(acquire._target)
    point.header.stamp = Stamp(sec=9, nanosec=950_000_000)
    point.point.y = 0.1
    acquire._enabled = False
    acquire._on_point(point)
    assert acquire._target_turn_direction == 1
    advance(acquire, 0.2, pose=(0.0, 0.0, 1.0))
    assert -math.sin(1.0) * point.point.x + math.cos(1.0) * point.point.y < 0
    acquire._set_enabled(SimpleNamespace(data=True), SimpleNamespace())
    acquire._tick()
    assert len(acquire._planner.goals) == 1
    assert acquire._sequence["turn_direction"] == 1


def test_side_is_frozen_across_approach_and_new_opposite_observation(acquire):
    acquire.params["max_maneuvers"] = 3
    point = copy.deepcopy(acquire._target)
    point.header.stamp = Stamp(sec=9, nanosec=950_000_000)
    point.point.y = -0.1
    acquire._on_point(point)
    first, first_goal = dispatch_leg(acquire)
    assert acquire._sequence["turn_direction"] == -1
    finish_leg(acquire, first, first_goal)

    newer = copy.deepcopy(point)
    newer.header.stamp = acquire.clock.to_msg()
    newer.point.y = 0.1  # Within the existing 0.25 m observation-shift allowance.
    acquire._on_point(newer)
    assert acquire._target_turn_direction == 1
    assert acquire._sequence["turn_direction"] == -1
    second, second_goal = dispatch_leg(acquire)
    dx, dy = second_goal[0] - first_goal[0], second_goal[1] - first_goal[1]
    assert -math.sin(first_goal[2]) * dx + math.cos(first_goal[2]) * dy < 0
    finish_leg(acquire, second, second_goal)

    third, third_goal = dispatch_leg(acquire)
    dx, dy = third_goal[0] - second_goal[0], third_goal[1] - second_goal[1]
    assert -math.sin(second_goal[2]) * dx + math.cos(second_goal[2]) * dy > 0
    finish_leg(acquire, third, third_goal)
    assert json.loads(acquire._result_pub.publish.call_args.args[0].data)["completed_maneuvers"] == 3

    restart_attempt(acquire)
    acquire._tick()
    assert acquire._sequence["turn_direction"] == 1


@pytest.mark.parametrize("count", [1, 2, 3])
def test_sequence_completes_once_after_each_terminal_result_and_confirmed_stop(acquire, count):
    acquire.params["max_maneuvers"] = count
    goals = []
    stamp = acquire._target.header.stamp
    for index in range(count):
        follower, goal = dispatch_leg(acquire)
        goals.append(goal)
        for _ in range(3):
            acquire._tick()
        assert len(acquire._planner.goals) == len(acquire._controller.goals) == index + 1
        assert acquire._sequence["completed"] == index
        finish_leg(acquire, follower, goal)
        assert acquire._operation is None
        assert acquire._sequence["completed"] == index + 1
    assert acquire._target.header.stamp == stamp
    assert acquire.clock.nanoseconds - acquire._stamp(stamp) > 3e9
    acquire._result_pub.publish.assert_called_once()
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "exhausted" and result["completed_maneuvers"] == count
    assert result["attempt_id"] == "fixture-attempt"
    assert result["travel_m"] < 1.05
    for _ in range(5):
        acquire._tick()
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == count


def test_disable_after_success_waits_for_stop_and_reports_canceled_once(acquire):
    follower, goal = dispatch_leg(acquire)
    advance(acquire, 4.0, goal)
    acquire._tick()
    operation = acquire._operation
    follower.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED))
    acquire._on_command(Twist())

    response = acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert response.success is False
    assert acquire._operation is operation
    assert follower.cancel_count == 0
    acquire._result_pub.publish.assert_not_called()

    advance(acquire, 0.1)
    assert acquire._operation is operation
    for _ in range(21):
        advance(acquire, 0.1)
    assert acquire._operation is None
    acquire._result_pub.publish.assert_called_once()
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["attempt_id"] == "fixture-attempt"
    assert result["outcome"] == "canceled"
    assert result["reason"] == "Acquire disabled"
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == 1
    assert follower.cancel_count == 0

    response = acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert response.success is True
    acquire._tick()
    acquire._result_pub.publish.assert_called_once()
    assert len(acquire._controller.goals) == 1


@pytest.mark.parametrize("missing", ["tf", "scan", "localization", "nav2", "stationary", "time", "travel", "target_shift"])
def test_next_leg_rechecks_readiness_and_whole_sequence_limits(acquire, missing):
    acquire.params["max_maneuvers"] = 3
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    if missing == "tf":
        acquire._tf.fail = True
    elif missing == "scan":
        acquire._scan_stamp = None
    elif missing == "localization":
        acquire._odom_info_history.clear()
    elif missing == "nav2":
        acquire._planner.ready = False
    elif missing == "stationary":
        acquire._odom_velocity = (0.1, 0.0, 0.0)
    elif missing == "time":
        acquire._sequence["started"] -= 31.0
    elif missing == "travel":
        acquire._sequence["traveled"] = 1.06
    else:
        acquire._target.point.x += 0.3
    acquire._tick()
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == 1


def test_disable_between_legs_prevents_next_goal(acquire):
    acquire.params["max_maneuvers"] = 3
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    assert acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace()).success
    acquire._tick()
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == 1
    assert json.loads(acquire._result_pub.publish.call_args.args[0].data)["outcome"] == "canceled"


def test_disable_during_second_plan_blocks_late_success(acquire):
    acquire.params["max_maneuvers"] = 3
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    acquire._tick()
    assert len(acquire._planner.goals) == 2
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    late_handle = FakeHandle()
    acquire._planner.response.set_result(late_handle)
    assert late_handle.cancel_count == 1
    finish_plan(acquire, late_handle, Path())
    acquire._tick()
    assert acquire._operation is None
    assert len(acquire._controller.goals) == 1


def test_disable_second_follow_prevents_third_even_if_nav2_reports_success(acquire):
    acquire.params["max_maneuvers"] = 3
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    second, second_goal = dispatch_leg(acquire)
    response = acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success and second.cancel_count == 1
    finish_leg(acquire, second, second_goal)
    acquire._tick()
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "canceled" and result["completed_maneuvers"] == 1
    assert len(acquire._planner.goals) == len(acquire._controller.goals) == 2


def test_goal_transforms_vo_memory_to_map_with_translation_and_rotation(acquire):
    map_from_odom = TransformStamped()
    map_from_odom.header.frame_id = "map"
    map_from_odom.header.stamp = Stamp(sec=10)
    map_from_odom.child_frame_id = "vo_odom"
    map_from_odom.transform.translation.x = 5.0
    map_from_odom.transform.translation.y = -2.0
    map_from_odom.transform.rotation.z = math.sin(math.pi / 4)
    map_from_odom.transform.rotation.w = math.cos(math.pi / 4)
    acquire._tf.map_tf = copy.deepcopy(map_from_odom)
    acquire._tf.transform = lambda pose, frame: tf2_geometry_msgs.do_transform_pose_stamped(pose, map_from_odom)
    follower, goal = dispatch_leg(acquire)
    assert goal == pytest.approx((5.0, -1.5, math.pi / 2))
    start = acquire._planner.goals[-1].start
    assert (start.pose.position.x, start.pose.position.y) == pytest.approx((5.0, -2.0))
    assert acquire._target.header.frame_id == "vo_odom"
    assert acquire._target.header.stamp == Stamp(sec=9, nanosec=900_000_000)


def test_aborted_leg_reports_failure_only_after_stop_and_never_advances(acquire):
    acquire.params["max_maneuvers"] = 3
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal, GoalStatus.STATUS_ABORTED)
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "failed" and result["completed_maneuvers"] == 0
    acquire._tick()
    assert len(acquire._planner.goals) == 1


@pytest.mark.parametrize("kind", ["plan", "follow"])
@pytest.mark.parametrize("terminal_first", [False, True])
@pytest.mark.parametrize("status", [GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED])
def test_terminal_status_survives_disable_ordering(acquire, kind, terminal_first, status):
    acquire.params["max_maneuvers"] = 3
    if kind == "follow":
        handle, goal = dispatch_leg(acquire)
        advance(acquire, 4.0, goal)
    else:
        handle = start_plan(acquire)
    operation = acquire._operation

    def disable():
        return acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())

    if not terminal_first:
        assert not disable().success
        assert handle.cancel_count == 1
    handle.result.set_result(SimpleNamespace(status=status))
    response = disable()
    # An unrequested planning cancellation already reports failure.
    fault = status == GoalStatus.STATUS_ABORTED or (kind == "plan" and terminal_first)
    assert acquire._fault_latched is fault
    assert f"fault_latched={str(fault).lower()}" in response.message

    if kind == "follow":
        assert not response.success and acquire._operation is operation
        acquire._result_pub.publish.assert_not_called()
        acquire._on_command(Twist())
        advance(acquire, 0.1)
        assert acquire._operation is operation
        for _ in range(21):
            advance(acquire, 0.1)
    else:
        assert response.success
    assert acquire._operation is None
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == ("failed" if fault else "canceled")
    assert result["completed_maneuvers"] == 0
    acquire._tick()
    assert len(acquire._planner.goals) == 1


@pytest.mark.parametrize("future_tf_ns, releases", [(37_000_000, True), (200_000_000, True), (200_000_001, False)])
def test_stop_confirmation_uses_existing_bounded_tf_lead(acquire, future_tf_ns, releases):
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal, future_tf_ns=future_tf_ns)
    assert (acquire._operation is None) is releases
    assert acquire._result_pub.publish.called is releases


@pytest.mark.parametrize("command", [None, (99.0, True), (100.0, False)])
def test_result_and_stationary_odom_cannot_release_without_zero_drive_command(acquire, command):
    follower, goal = dispatch_leg(acquire)
    acquire._last_command = command
    finish_leg(acquire, follower, goal, zero_command=False)
    assert acquire._operation is not None
    acquire._result_pub.publish.assert_not_called()
    acquire._on_command(Twist())
    advance(acquire, 0.1)
    assert acquire._operation is not None
    for _ in range(21):
        advance(acquire, 0.1)
    assert acquire._operation is None


def test_new_nonzero_command_resets_stop_confirmation(acquire):
    follower, goal = dispatch_leg(acquire)
    advance(acquire, 4.0, goal)
    follower.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    acquire._on_command(Twist())
    advance(acquire, 0.1)
    command = Twist()
    command.linear.x = 0.01
    acquire._on_command(command)
    advance(acquire, 0.3)
    assert acquire._operation is not None and acquire._operation["stop_window"] is None
    acquire._on_command(Twist())
    advance(acquire, 0.1)
    assert acquire._operation["stop_window"]["samples"] == 1


def test_terminal_stop_accepts_noisy_twist_after_full_pose_window(acquire):
    goal = begin_terminal_cancel(acquire)
    for index in range(22):
        twist = (0.08, -0.03, 0.07) if index % 2 else (-0.06, 0.04, -0.06)
        advance(acquire, 0.1, goal, twist=twist)
    assert acquire._operation is None, acquire._operation.get("stop_window")
    acquire._result_pub.publish.assert_called_once()


def test_terminal_stop_rejects_sustained_creep_until_a_new_full_window(acquire):
    goal = begin_terminal_cancel(acquire)
    for index in range(22):
        advance(acquire, 0.1, (goal[0] + 0.0006 * (index + 1), goal[1], goal[2]))
    assert acquire._operation is not None
    assert acquire._result_pub.publish.call_count == 0
    held = (goal[0] + 0.0006 * 22, goal[1], goal[2])
    for _ in range(21):
        advance(acquire, 0.1, held)
    assert acquire._operation is None
    acquire._result_pub.publish.assert_called_once()


def test_terminal_stop_resets_fixed_xy_anchor_when_translation_returns(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    advance(acquire, 0.1, (goal[0] + 0.012, goal[1], goal[2]))
    advance(acquire, 0.1, goal)
    for _ in range(19):
        advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    acquire._result_pub.publish.assert_not_called()
    advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_bounds_sustained_yaw_rate_and_requires_new_window(acquire):
    goal = begin_terminal_cancel(acquire)
    for index in range(1, 19):
        advance(acquire, 0.1, (goal[0], goal[1], goal[2] + 0.006 * index))
    assert acquire._operation is not None
    acquire._result_pub.publish.assert_not_called()
    stable_yaw = goal[2] + 0.006 * 18
    for _ in range(22):
        advance(acquire, 0.1, (goal[0], goal[1], stable_yaw))
    assert acquire._operation is None


def test_terminal_stop_angular_variation_rejects_out_and_back_yaw(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    for _ in range(6):
        advance(acquire, 0.1, goal)
    advance(acquire, 0.1, (goal[0], goal[1], goal[2] + 0.06))
    advance(acquire, 0.1, goal)
    for _ in range(12):
        advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    acquire._result_pub.publish.assert_not_called()
    for _ in range(22):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_repeated_stamp_and_gap_restart_the_window(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    duplicate = stationary_odom(acquire.clock.nanoseconds)
    acquire._on_odom(duplicate)
    for _ in range(19):
        advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    assert acquire._operation is None

    restart_attempt(acquire)
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    advance(acquire, 1.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    for _ in range(19):
        advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    assert acquire._operation is not None
    advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_invalid_odom_and_localization_loss_restart_window(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    invalid = stationary_odom(acquire.clock.nanoseconds + 100_000_000)
    invalid.twist.twist.linear.x = float("nan")
    acquire.clock.nanoseconds += 100_000_000
    acquire.wall[0] += 0.1
    acquire._on_odom(invalid)
    assert acquire._operation is not None
    for _ in range(22):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None

    restart_attempt(acquire)
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    stamp = acquire.clock.to_msg()
    acquire._on_info(SimpleNamespace(lost=True, header=SimpleNamespace(stamp=stamp)))
    assert acquire._operation is not None
    acquire.clock.nanoseconds += 1
    acquire.wall[0] += 1e-9
    for transform in (acquire._tf.map_tf, acquire._tf.odom_tf):
        transform.header.stamp = Stamp(sec=acquire.clock.nanoseconds // 1_000_000_000,
                                        nanosec=acquire.clock.nanoseconds % 1_000_000_000)
    acquire._on_info(SimpleNamespace(lost=False, header=SimpleNamespace(
        stamp=Stamp(sec=acquire.clock.nanoseconds // 1_000_000_000,
                    nanosec=acquire.clock.nanoseconds % 1_000_000_000))))
    for _ in range(22):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_clock_regression_clears_window_and_requires_reconfirmation(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    advance(acquire, 0.1, goal)
    terminal_stamp = acquire._operation["terminal_stamp"]
    acquire.clock.nanoseconds = terminal_stamp - 100_000_000
    acquire._now()
    assert acquire._operation is not None
    assert acquire._operation["stop_window"] is None
    acquire.clock.nanoseconds = terminal_stamp + 100_000_000
    acquire._tf.map_tf.header.stamp = Stamp(sec=acquire.clock.nanoseconds // 1_000_000_000,
                                            nanosec=acquire.clock.nanoseconds % 1_000_000_000)
    acquire._tf.odom_tf.header.stamp = copy.deepcopy(acquire._tf.map_tf.header.stamp)
    acquire._odom_info_history.clear()
    acquire._odom_info_history.append(acquire.clock.nanoseconds)
    acquire._odom_stamp = None
    acquire._on_command(Twist())
    acquire._on_odom(stationary_odom(acquire.clock.nanoseconds))
    for _ in range(23):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_ignores_transient_cached_tf_without_resetting_or_crediting(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    window = acquire._operation["stop_window"]
    start_wall = window["start_wall"]
    start_samples = window["samples"]
    advance_cached_odom(acquire)
    assert acquire._operation["stop_window"]["start_wall"] == start_wall
    assert acquire._operation["stop_window"]["samples"] == start_samples
    advance(acquire, 0.1, goal)
    for _ in range(20):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_cached_tf_alone_cannot_release_and_stale_gap_restarts(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    for _ in range(12):
        advance_cached_odom(acquire)
    assert acquire._operation is not None
    assert acquire._operation["stop_window"] is None
    # A fresh TF starts a new window; it cannot inherit the stale cached span.
    advance(acquire, 0.1, goal)
    assert acquire._operation["stop_window"]["samples"] == 1
    for _ in range(21):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_terminal_stop_same_tf_stamp_changed_pose_and_backward_tf_reset(acquire):
    goal = begin_terminal_cancel(acquire)
    advance(acquire, 0.1, goal)
    stamp = copy.deepcopy(acquire._tf.odom_tf.header.stamp)
    acquire.clock.nanoseconds += 100_000_000
    acquire.wall[0] += 0.1
    for transform in (acquire._tf.map_tf, acquire._tf.odom_tf):
        transform.header.stamp = copy.deepcopy(stamp)
        transform.transform.translation.x += 0.012
    acquire._odom_info_history.append(acquire.clock.nanoseconds)
    acquire._on_odom(stationary_odom(acquire.clock.nanoseconds))
    assert acquire._operation["stop_window"] is None

    advance(acquire, 0.1, goal)
    assert acquire._operation["stop_window"]["samples"] == 1
    previous = acquire._operation["stop_window"]["last_tf_stamp"]
    acquire.clock.nanoseconds += 100_000_000
    acquire.wall[0] += 0.1
    backward = previous - 100_000_000
    acquire._tf.odom_tf.header.stamp = Stamp(sec=backward // 1_000_000_000,
                                             nanosec=backward % 1_000_000_000)
    acquire._odom_info_history.append(acquire.clock.nanoseconds)
    acquire._on_odom(stationary_odom(acquire.clock.nanoseconds))
    assert acquire._operation["stop_window"] is None
    for _ in range(22):
        advance(acquire, 0.1, goal)
    assert acquire._operation is None


def test_completed_entry_requires_disable_enable_to_restart(acquire):
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    original_id = acquire._attempt_id
    acquire._set_enabled(SimpleNamespace(data=True), SimpleNamespace())
    acquire._tick()
    assert acquire._attempt_id == original_id and len(acquire._planner.goals) == 1
    acquire._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    acquire._set_enabled(SimpleNamespace(data=True), SimpleNamespace())
    assert acquire._attempt_id != original_id and acquire._sequence is None
    assert not acquire._finished


def test_path_cannot_spend_budget_reserved_for_the_entire_sequence(acquire):
    from test_acquire_path import start_plan, make_path
    handle = start_plan(acquire)
    acquire._sequence["traveled"] = 0.8
    finish_plan(acquire, handle, make_path())
    assert not acquire._controller.goals
    assert "remaining sequence travel budget" in json.loads(acquire._result_pub.publish.call_args.args[0].data)["reason"]


def test_recorded_third_path_is_rejected_at_historical_remaining_budget(acquire):
    from test_acquire_path import start_plan, make_path

    handle = start_plan(acquire)
    acquire._sequence["traveled"] = 0.6309455216925647
    path_length_m = 0.46418399728422055
    path = make_path((0.0, path_length_m))
    finish_plan(acquire, handle, path)

    assert 1.2 - 0.15 - acquire._sequence["traveled"] == pytest.approx(0.41905447830743536)
    assert path_length_m > 1.2 - 0.15 - acquire._sequence["traveled"]
    assert not acquire._validator.requests
    assert not acquire._controller.goals
    result = json.loads(acquire._result_pub.publish.call_args.args[0].data)
    assert result["outcome"] == "blocked"
    assert result["completed_maneuvers"] == 0
    assert "remaining sequence travel budget" in result["reason"]


@pytest.mark.parametrize("limit", ["time", "travel"])
def test_second_leg_keeps_first_leg_budget_and_first_cancel_reason(acquire, limit):
    acquire.params["max_maneuvers"] = 3
    first, goal = dispatch_leg(acquire)
    finish_leg(acquire, first, goal)
    started = acquire._sequence["started"]
    spent = acquire._sequence["traveled"]
    second, _ = dispatch_leg(acquire)
    assert acquire._sequence["started"] == started
    assert acquire._sequence["traveled"] == spent
    if limit == "time":
        acquire._sequence["started"] -= 31.0
        expected = "sequence time limit"
    else:
        advance(acquire, 0.1, (1.06, 0.0, 0.0))
        expected = "sequence travel limit with braking margin"
    acquire._tick()
    assert second.cancel_count == 1
    assert acquire._operation["cancel_reason"] == expected
    acquire._cancel("later failure must not overwrite the first cause")
    assert acquire._operation["cancel_reason"] == expected


def test_sequence_travel_includes_terminal_stop_window(acquire):
    follower, goal = dispatch_leg(acquire)
    advance(acquire, 0.1, (0.2, 0.0, 0.0))
    acquire._tick()
    follower.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED))
    acquire._on_command(Twist())
    advance(acquire, 0.1, (0.25, 0.0, 0.0))
    assert acquire._sequence["traveled"] == pytest.approx(0.25)
    assert acquire._operation is not None
    assert not acquire._result_pub.publish.called
