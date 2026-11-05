"""One startup setpoint drives control, Nav2 and the derived STOP distance."""
from types import SimpleNamespace as NS

import pytest
import rclpy
from rclpy.parameter import Parameter

from piracer_state_manager.follow_node import PiracerFollowNode
from test_follow_nav2_integration import _grid, _integrated_cycle_node, _stamp_grid


@pytest.fixture(params=[2.25, 1.0])
def follow(request):
    assert not rclpy.ok()
    rclpy.init(args=["--ros-args", "-p", f"target_distance_m:={request.param}"])
    node = None
    try:
        node = PiracerFollowNode()
        yield node
    finally:
        if node is not None:
            node.stop_and_destroy()
        rclpy.shutdown()


def test_startup_setpoint_controls_distance_and_preserves_motion_gate(follow):
    desired = follow.get_parameter("target_distance_m").value
    controller = follow._controller
    assert controller.target_distance_m == desired
    assert controller.min_distance_m == {2.25: 2.0, 1.0: .75}[desired]
    assert controller.update(desired, 0.).linear_x == 0
    assert controller.update(desired + .5, 0.).linear_x > 0
    assert controller.update(controller.min_distance_m, 0.).linear_x == 0
    assert not follow._publish_cmd_vel


def test_live_update_is_rejected_without_changing_effective_setpoint(follow):
    desired = follow._controller.target_distance_m
    result = follow.set_parameters([Parameter("target_distance_m", value=3.0)])[0]
    assert not result.successful
    assert follow.get_parameter("target_distance_m").value == desired
    assert follow._controller.target_distance_m == desired


def test_factory_rejects_invalid_startup_setpoints(follow, monkeypatch):
    original = follow.get_parameter
    for invalid in (0.25, 0., -1., float("nan"), float("inf")):
        monkeypatch.setattr(follow, "get_parameter", lambda name, value=invalid:
                            NS(value=value) if name == "target_distance_m" else original(name))
        with pytest.raises(ValueError):
            follow._make_controller()


def test_nav2_goal_uses_the_same_configured_standoff(follow):
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(width=120), 5_000_000_000, "map"))
    node._controller = follow._controller
    desired = node._controller.target_distance_m
    node._test_target.point.x, node._test_target.point.y = desired + 1., 0.
    node._tf_buffer.transform = lambda message, _frame: message
    node._latest_scan.ranges[50] = .5
    command = node._controller.update(desired + 1., 0.)
    node._integrated_cycle(command, node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert len(node._nav2_action.goals) == 1
    goal = node._nav2_action.goals[0].pose.pose.position
    assert node._test_target.point.x - goal.x == pytest.approx(desired)
    assert goal.y == pytest.approx(0.)


def test_scan_stops_at_the_configured_standoff(follow):
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    node._controller = follow._controller
    desired = node._controller.target_distance_m
    node._test_target.point.x, node._test_target.point.y = desired + .0233, 0.
    node._tf_buffer.transform = lambda message, _frame: message
    node._latest_scan.ranges[50] = desired - .05
    command = node._controller.update(desired + .0233, 0.)
    node._integrated_cycle(command, node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert not node._nav2_action.goals and not node._fault_latched
    assert node._last_selected_source == "direct"


def test_integrated_stop_uses_the_derived_minimum(follow):
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    node._controller = follow._controller
    node._test_target.point.x = node._controller.min_distance_m
    node._test_target.point.y = 0.
    node._integrated_cycle(NS(linear_x=.05, angular_z=0.), node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert node._mode_reason == "target_inside_min_distance"
    assert node._last_selected_twist.linear.x == 0
    assert not node._nav2_action.goals
