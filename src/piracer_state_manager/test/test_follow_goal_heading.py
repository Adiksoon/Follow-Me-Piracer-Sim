"""Update the planner's approach heading without moving its committed goal."""
import math
from types import SimpleNamespace as NS

import pytest
from geometry_msgs.msg import TransformStamped, Twist

from piracer_state_manager.follow_nav2_helpers import NAV2, footprint_clear
from test_follow_nav2_integration import (
    _grid, _integrated_cycle_node, _set_world_cell, _stamp_grid,
)


@pytest.fixture
def owned_goal():
    grid = _stamp_grid(_grid(width=180, height=120, origin=(-2., -3., 0.)),
                       5_000_000_000, "map")
    node = _integrated_cycle_node(mode=NAV2, local_grid=grid, global_grid=grid)
    node._goal_state, node._goal_unresolved = "active", True
    node._goal_handle = NS(goal_id=NS(uuid=[1] * 16), cancel_goal_async=lambda: None)
    node._sent_goal_stamp_ns = node._test_now_ns
    node._active_actions = {
        "navigate_to_pose": {bytes([1] * 16): (2, node._test_now_ns)},
        "follow_path": {b"owned-path": (2, node._test_now_ns)},
    }
    node._nav2_goal_target_map = (4.84, 0.01)
    node._nav2_goal_offset = (-2.25, 0.)
    node._nav2_latest_command = Twist()
    node._nav2_latest_command.linear.x = .05
    node._nav2_latest_command_wall = node._test_monotonic_now
    node._nav2_has_moved = True
    return node


def forward(node, robot_xy):
    node._forward_nav_command((0., 0., 0.), node._test_monotonic_now, map_xy=robot_xy)


def test_planner_heading_changes_while_committed_xy_and_action_remain(owned_goal):
    node = owned_goal
    handle = node._goal_handle
    offset = node._nav2_goal_offset
    for robot in ((1.65, -.88), (2., -.8)):
        forward(node, robot)
        msg = node._nav2_goal_update_publisher.messages[-1]
        assert (msg.pose.position.x, msg.pose.position.y) == pytest.approx((2.59, .01))
        yaw = 2 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        assert yaw == pytest.approx(math.atan2(.01 - robot[1], 2.59 - robot[0]))
        assert msg.header.frame_id == "map" and msg.header.stamp.sec == 5
    assert len(node._nav2_goal_update_publisher.messages) == 2
    assert node._goal_handle is handle and node._nav2_goal_offset == offset
    assert not node._nav2_action.goals and node._goal_state == "active"


def test_target_loss_keeps_the_committed_observation_xy(owned_goal):
    node = owned_goal
    node._memory_goal = (2.59, .01, 0.)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.
    node._tf_buffer.lookup_transform = lambda *_: tf
    forward(node, (2., -.8))
    msg = node._nav2_goal_update_publisher.messages[-1]
    assert (msg.pose.position.x, msg.pose.position.y) == pytest.approx((2.59, .01))


def test_new_heading_is_rejected_if_endpoint_footprint_becomes_occupied(owned_goal):
    node = owned_goal
    node._nav2_goal_target_map = (4.84, 0.)
    _set_world_cell(node._global_grid, 2.59, .20, 100)
    assert footprint_clear(node._global_grid, 2.59, 0., 0.)
    assert not footprint_clear(node._global_grid, 2.59, 0., math.pi / 2)
    forward(node, (2.59, -1.))
    assert not node._nav2_goal_update_publisher.messages
    assert node._nav2_goal_offset == (-2.25, 0.)


@pytest.mark.parametrize("failure", ["foreign_owner", "zero_command", "stale_command"])
def test_rejected_nav_command_cannot_update_heading(owned_goal, failure):
    node = owned_goal
    if failure == "foreign_owner":
        node._active_actions["navigate_to_pose"][bytes([2] * 16)] = (2, node._test_now_ns)
    elif failure == "zero_command":
        node._nav2_latest_command.linear.x = 0.
    else:
        node._nav2_latest_command_wall -= 1.
    forward(node, (1.65, -.88))
    assert not node._nav2_goal_update_publisher.messages
    assert node._last_selected_twist.linear.x == 0.


@pytest.mark.parametrize("blocked", ["disabled", "dry", "stale_data"])
def test_disabled_dry_and_stale_inputs_cannot_update_heading(owned_goal, blocked):
    node = owned_goal
    if blocked == "disabled":
        node._enabled = False
    elif blocked == "dry":
        node._publish_cmd_vel = False
    else:
        node._can_hold_data_wait = lambda *_: False
        node._integrated_inputs_fresh = lambda *_: (False, "stale_scan")
    node._integrated_cycle(NS(linear_x=.05, angular_z=0.), node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert not node._nav2_goal_update_publisher.messages
