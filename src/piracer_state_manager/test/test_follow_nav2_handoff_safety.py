"""Safety regressions retained after removing geometric observation arcs."""
import math
from types import SimpleNamespace as NS
import pytest
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from rclpy.time import Time
from piracer_state_manager import follow_node as module
from test_follow_nav2_integration import _grid, _set_world_cell
from test_follow_last_seen import episode, Handle, reacquire_point


def tick(n, t):
    stamp = n._stamp_ns(n._latest_target_point)
    receipt = n._latest_target_received_monotonic
    if receipt is None or t.wall-receipt > .5 or t.ros-stamp > 500_000_000:
        return n._try_last_seen(t.ros, t.wall, False)
    return n._integrated_cycle(NS(linear_x=.04, angular_z=0.), n._test_target, t.ros, t.wall)


@pytest.fixture
def observation(episode):
    n,t,_=episode
    n._goal_handle=None
    n._goal_state,n._goal_unresolved='idle',False
    n._goal_cancel_reason=None
    n._mode=module.NAV2_PENDING
    n._nav2_goal_offset=n._nav2_goal_target_map=None
    n._standoff_id=None
    n._active_actions={}
    n._nav2_action.goals=[]
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._global_grid.data[:]=[0]*len(n._global_grid.data)
    n._latest_scan.ranges[20]=1.8
    t.ros,t.wall=5_000_000_000,100.
    n._latest_detections=NS(header=NS(stamp=Time(nanoseconds=t.ros).to_msg()),detections=[object()])
    n._detection_receipt=t.wall
    return n,t


@pytest.fixture
def waiting_observation(observation):
    n,t=observation
    original=n.get_parameter
    n.get_parameter=lambda key: NS(value={'local_costmap_max_age_s':.6,'global_costmap_max_age_s':2.}[key]) if key in ('local_costmap_max_age_s','global_costmap_max_age_s') else original(key)
    n._odom_info=NS(lost=False,header=NS(stamp=Time(nanoseconds=t.ros).to_msg()))
    n._local_grid_received_monotonic=n._global_grid_received_monotonic=t.wall
    n._latest_scan_received_monotonic=n._odom_info_received_monotonic=t.wall
    n._start_nav2_goal()
    h=Handle(10)
    n._on_nav2_goal_response(NS(result=lambda:h))
    n._active_actions['navigate_to_pose']={bytes(h.goal_id.uuid):(2,t.ros)}
    n._active_actions['follow_path']={bytes([10]):(2,t.ros)}
    command=Twist();command.linear.x=.04
    n._on_nav2_command(command)
    tick(n,t)
    assert n._last_selected_twist.linear.x==pytest.approx(.04)
    return n,t,h


@pytest.mark.parametrize('reason', ['stale_scan', 'stale_local_costmap', 'stale_global_costmap', 'ros_clock_stalled'])
def test_temporary_gap_keeps_one_owned_arc_zero_then_resumes(waiting_observation, reason):
    (n, t, h) = waiting_observation
    n._safe_stop(reason)
    assert n._last_selected_twist.linear.x == 0
    assert h.cancels == 0 and (not n._fault_latched)
    assert n._goal_handle is h and n._nav2_goal_offset is not None
    assert n._nav2_latest_command is None
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0
    fresh = Twist()
    fresh.linear.x = 0.04
    n._on_nav2_command(fresh)
    tick(n, t)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0 and len(n._nav2_action.goals) == 1

def test_fresh_owned_arc_crosses_unknown_footprint_without_cancelling(waiting_observation):
    (n, t, h) = waiting_observation
    n._local_grid.data[:] = [-1] * len(n._local_grid.data)
    n._integrated_cycle(NS(linear_x=0.04, angular_z=0.0), n._test_target, t.ros, t.wall)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0 and n._nav2_goal_offset is not None
    n._on_odom_info(NS(lost=True))
    assert h.cancels == 1 and n._last_selected_twist.linear.x == 0

@pytest.mark.parametrize('occupied', [False, True])
def test_confirmation_gap_keeps_zero_even_when_the_costmap_changes(waiting_observation, occupied):
    (n, t, h) = waiting_observation
    n._last_seen = (*n._last_seen[:2], t.ros - 900000000, t.wall - 0.9)
    n._latest_target_point.header.stamp = Time(nanoseconds=n._last_seen[2]).to_msg()
    n._latest_detections = None
    if occupied:
        _set_world_cell(n._local_grid, 0.1, 0.0, 100)
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0
    assert h.cancels == 0 and (not n._fault_latched)
    assert n._nav2_goal_offset is not None and n._nav2_latest_command is None

@pytest.mark.parametrize('fault', ['future_scan', 'future_detection', 'future_target', 'invalid_receipt', 'regression', 'expired', 'lost', 'tf', 'foreign'])
def test_invalid_inputs_never_hold_an_observation_action(waiting_observation, fault):
    (n, t, h) = waiting_observation
    if fault == 'future_scan':
        n._latest_scan.header.stamp = Time(nanoseconds=t.ros + 300000000).to_msg()
    elif fault == 'future_detection':
        n._latest_detections.header.stamp = Time(nanoseconds=t.ros + 1).to_msg()
    elif fault == 'future_target':
        n._latest_target_point.header.stamp = Time(nanoseconds=t.ros + 1).to_msg()
    elif fault == 'invalid_receipt':
        n._latest_scan_received_monotonic = float('nan')
    elif fault == 'regression':
        n._last_integrated_ros_ns = t.ros + 1
    elif fault == 'expired':
        t.wall += 120.1
    elif fault == 'lost':
        n._odom_info.lost = True
    elif fault == 'tf':
        n._tf_pose = lambda *_: (_ for _ in ()).throw(module.TransformException('invalid TF'))
    else:
        n._active_actions['navigate_to_pose'][b'foreign'] = (2, t.ros)
    n._safe_stop('stale_scan')
    assert h.cancels == 1 and n._last_seen is None
    assert n._last_selected_twist.linear.x == 0 and n._last_seen is None

def test_real_control_cycle_holds_clock_pause_and_recovers_same_goal(waiting_observation):
    from piracer_state_manager.follow_control import FollowPController
    from rclpy.clock import ClockType
    (n, t, h) = waiting_observation
    n._controller = FollowPController(max_speed_mps=0.075)
    n._cycle_counts = {'timer_calls': 0, 'ros_time_zero_cycles': 0}
    n._record_cycle = lambda *_: None
    n._publish_control_sample = lambda *_: None
    n._last_ros_time_ns = None
    n._last_ros_time_advance_monotonic = None
    n.get_clock = lambda : NS(now=lambda : Time(nanoseconds=t.ros), clock_type=ClockType.SYSTEM_TIME)
    original_parameter = n.get_parameter
    n.get_parameter = lambda name: NS(value='base_link') if name == 'base_frame' else original_parameter(name)
    n._latest_target_point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._control_cycle()
    t.wall += 0.6
    n._latest_target_received_monotonic = t.wall
    n._control_cycle()
    assert n._mode_reason == 'ros_clock_stalled'
    assert n._last_selected_twist.linear.x == 0 and h.cancels == 0
    t.ros += 50000000
    t.wall += 0.05
    n._latest_target_point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._latest_target_received_monotonic = t.wall
    fresh = Twist()
    fresh.linear.x = 0.04
    n._on_nav2_command(fresh)
    n._control_cycle()
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert n._goal_handle is h and h.cancels == 0 and (len(n._nav2_action.goals) == 1)

@pytest.mark.parametrize('which', ['local', 'global'])
def test_real_cycle_confirmation_gap_clears_motion_while_nav2_owns_collision_checks(waiting_observation, which):
    (n, t, h) = waiting_observation
    n._memory_goal = (*n._last_seen[:2], 0.0)
    n._last_seen = (*n._last_seen[:2], t.ros - 900000000, t.wall - 0.9)
    n._latest_target_point.header.stamp = Time(nanoseconds=n._last_seen[2]).to_msg()
    n._latest_target_received_monotonic = t.wall - 0.9
    n._latest_detections = None
    _set_world_cell(n._local_grid if which == 'local' else n._global_grid, n._memory_goal[0] + 0.1, n._memory_goal[1], 100)
    n._cycle_counts = {'timer_calls': 0, 'ros_time_zero_cycles': 0}
    n._record_cycle = lambda *_: None
    n._last_ros_time_ns = None
    n._last_ros_time_advance_monotonic = None
    n._control_cycle()
    assert n._last_selected_twist.linear.x == 0
    assert h.cancels == 0 and (not n._fault_latched)
    assert n._nav2_goal_offset is not None and n._nav2_latest_command is None

@pytest.mark.parametrize('kind', ['point', 'detection'])
def test_sampled_clock_future_messages_do_not_refresh_accepted_cache(waiting_observation, kind):
    import copy
    (n, t, h) = waiting_observation
    (old_point, old_detection) = (n._latest_target_point, n._latest_detections)
    old_receipts = (n._latest_target_received_monotonic, n._detection_receipt)
    memory = n._last_seen
    message = copy.deepcopy(old_point if kind == 'point' else old_detection)
    message.header.stamp = Time(nanoseconds=t.ros + 9000000).to_msg()
    (n._on_target_point if kind == 'point' else n._on_detections)(message)
    assert n._latest_target_point is old_point and n._latest_detections is old_detection
    assert (n._latest_target_received_monotonic, n._detection_receipt) == old_receipts
    assert n._last_seen == memory and h.cancels == 0

@pytest.mark.parametrize('order', ['point_first', 'positive_first'])
def test_future_observations_do_not_reacquire_or_replace_owned_memory(waiting_observation, order):
    import copy
    (n, t, h) = waiting_observation
    n._memory_goal = (*n._last_seen[:2], 0.0)
    t.ros += 900000000
    t.wall += 0.9
    n._latest_detections = NS(header=NS(stamp=Time(nanoseconds=t.ros).to_msg()), detections=[])
    n._detection_receipt = t.wall
    point = copy.deepcopy(n._latest_target_point)
    point.header.stamp = Time(nanoseconds=t.ros + 9000000).to_msg()
    detection = NS(header=NS(stamp=point.header.stamp), detections=[object()])
    calls = [(n._on_target_point, point), (n._on_detections, detection)]
    if order == 'positive_first':
        calls.reverse()
    for (callback, message) in calls:
        callback(message)
    n._cycle_counts = {'timer_calls': 0, 'ros_time_zero_cycles': 0}
    n._record_cycle = lambda *_: None
    n._last_ros_time_ns = None
    n._last_ros_time_advance_monotonic = None
    fresh = Twist()
    fresh.linear.x = 0.04
    n._on_nav2_command(fresh)
    n._control_cycle()
    assert h.cancels == 0 and n._memory_goal is not None
    assert n._nav2_goal_offset is not None and n._mode == module.NAV2

def test_accepted_point_regression_returns_without_replacing_cache(waiting_observation):
    import copy
    (n, t, h) = waiting_observation
    old = n._latest_target_point
    point = copy.deepcopy(old)
    point.header.stamp = Time(nanoseconds=t.ros - 1).to_msg()
    n._on_target_point(point)
    assert n._latest_target_point is old and h.cancels == 1

def test_pending_owned_arc_can_wait_without_cancelling_its_send(waiting_observation):
    (n, t, _) = waiting_observation
    n._goal_handle = None
    (n._goal_state, n._mode) = ('sending', module.NAV2_PENDING)
    n._goal_send_future = object()
    n._active_actions.clear()
    n._safe_stop('stale_scan')
    assert n._goal_state == 'sending' and n._goal_cancel_reason is None
    assert n._last_selected_twist.linear.x == 0 and n._nav2_goal_offset is not None
    accepted(n, t, 11)
    assert len(n._nav2_action.goals) == 1

@pytest.mark.parametrize('missing', ['navigate_to_pose', 'follow_path', 'both'])
def test_incomplete_action_status_holds_zero_until_ownership_and_new_twist(waiting_observation, missing):
    (n, t, h) = waiting_observation
    complete = {name: dict(status) for (name, status) in n._active_actions.items()}
    for name in ('navigate_to_pose', 'follow_path'):
        if missing in (name, 'both'):
            n._active_actions[name] = {}
    n._safe_stop('awaiting_target_confirmation')
    assert h.cancels == 0 and (not n._fault_latched)
    assert n._last_selected_twist.linear.x == 0 and n._nav2_goal_offset is not None
    command = Twist()
    command.linear.x = 0.04
    n._on_nav2_command(command)
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0 and n._nav2_latest_command is None
    n._active_actions = complete
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0
    n._on_nav2_command(command)
    tick(n, t)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0 and len(n._nav2_action.goals) == 1

@pytest.mark.parametrize('foreign', [False, True])
def test_early_status_waits_zero_and_is_checked_immediately_at_acceptance(waiting_observation, foreign):
    (n, t, _) = waiting_observation
    h = Handle(11)
    n._goal_handle = None
    (n._goal_state, n._mode) = ('sending', module.NAV2_PENDING)
    n._goal_send_future = object()
    observed_id = b'foreign' if foreign else bytes(h.goal_id.uuid)
    n._active_actions['navigate_to_pose'] = {observed_id: (2, t.ros)}
    n._safe_stop('awaiting_target_confirmation')
    assert n._goal_cancel_reason is None and (not n._fault_latched)
    assert n._last_selected_twist.linear.x == 0
    n._on_nav2_goal_response(NS(result=lambda : h))
    if foreign:
        assert h.cancels == 1 and n._last_seen is None
        assert n._goal_cancel_reason == 'foreign_nav2_goal'
        assert n._last_selected_twist.linear.x == 0
    else:
        assert h.cancels == 0 and n._goal_state == 'active'
        tick(n, t)
        assert n._last_selected_twist.linear.x == 0
        command = Twist()
        command.linear.x = 0.04
        n._on_nav2_command(command)
        tick(n, t)
        assert n._last_selected_twist.linear.x == pytest.approx(0.04)

def test_two_navigate_goals_before_acceptance_cannot_hold_an_arc(waiting_observation):
    (n, t, _) = waiting_observation
    n._goal_handle = None
    (n._goal_state, n._mode) = ('sending', module.NAV2_PENDING)
    n._goal_send_future = object()
    n._active_actions['navigate_to_pose'] = {b'one': (2, t.ros), b'two': (2, t.ros)}
    n._safe_stop('awaiting_target_confirmation')
    assert n._last_seen is None and n._last_selected_twist.linear.x == 0
    assert n._goal_cancel_reason == 'awaiting_target_confirmation'
    assert n._last_selected_twist.linear.x == 0

@pytest.mark.parametrize('kind', ['point', 'detection', 'regression', 'reacquire_point', 'reacquire_detection'])
def test_hard_stamp_rejection_cannot_restart_direct_from_old_cache(waiting_observation, kind):
    import copy
    from piracer_state_manager.follow_control import FollowPController
    from rclpy.clock import ClockType
    (n, t, _) = waiting_observation
    n._mode = module.DIRECT
    n._memory_goal = n._last_seen = None
    n._goal_handle = None
    (n._goal_state, n._goal_unresolved) = ('idle', False)
    n._active_actions.clear()
    n._last_blocked_stamp_ns = None
    n._controller = FollowPController(max_speed_mps=0.075)
    n._cycle_counts = {'timer_calls': 0, 'ros_time_zero_cycles': 0}
    n._record_cycle = n._publish_control_sample = lambda *_: None
    n._last_ros_time_ns = n._last_ros_time_advance_monotonic = None
    n.get_clock = lambda : NS(now=lambda : Time(nanoseconds=t.ros), clock_type=ClockType.SYSTEM_TIME)
    original_parameter = n.get_parameter
    n.get_parameter = lambda name: NS(value='base_link') if name == 'base_frame' else original_parameter(name)
    n._latest_scan.ranges = [2.8]*len(n._latest_scan.ranges)
    valid = copy.deepcopy(n._latest_target_point)
    valid.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._on_target_point(valid)
    n._control_cycle()
    assert n._last_selected_twist.linear.x > 0
    if kind.startswith('reacquire'):
        n._mode = module.NAV2
        n._last_seen = (valid.point.x, valid.point.y, t.ros, t.wall)
        n._memory_goal = (valid.point.x, valid.point.y, 0.0)
    invalid = copy.deepcopy(n._latest_detections if 'detection' in kind else valid)
    invalid.header.stamp = Time(nanoseconds=t.ros + (1000000000 if kind != 'regression' else -1)).to_msg()
    (n._on_detections if 'detection' in kind else n._on_target_point)(invalid)
    n._control_cycle()
    assert n._last_selected_twist.linear.x == 0
    assert n._latest_target_received_monotonic is None
    n._on_target_point(valid)
    n._control_cycle()
    assert n._last_selected_twist.linear.x == 0
    t.ros += 1000000
    t.wall += 0.001
    valid.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._on_target_point(valid)
    n._control_cycle()
    if kind.startswith('reacquire'):
        assert n._fault_latched and n._last_selected_twist.linear.x == 0
        assert n._latest_target_received_monotonic is None
    else:
        assert n._last_selected_twist.linear.x > 0

@pytest.mark.parametrize('scan_kind', ['nan', 'inf', 'off_axis'])
def test_known_free_sweep_does_not_require_a_finite_forward_scan_return(waiting_observation, scan_kind):
    (n, t, h) = waiting_observation
    if scan_kind == 'off_axis':
        n._latest_scan.angle_min = 1.0
    else:
        n._latest_scan.ranges = [float(scan_kind)] * 41
    n._forward_nav_command((0.0, 0.0, 0.0), t.wall, blocked_cost=99)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0

@pytest.mark.parametrize('cost', [-1, 100])
def test_missing_scan_returns_preserve_occupied_stop_and_allow_unknown(waiting_observation, cost):
    (n, t, h) = waiting_observation
    n._latest_scan.ranges = [float('nan')] * 41
    _set_world_cell(n._local_grid, 0.3, 0.0, cost)
    command = Twist()
    (command.linear.x, command.angular.z) = (0.04, 1.0)
    n._on_nav2_command(command)
    n._forward_nav_command((0.0, 0.0, 0.0), t.wall, blocked_cost=99)
    if cost == 100:
        assert n._last_selected_twist.linear.x == 0 and h.cancels == 1
    else:
        assert n._last_selected_twist.linear.x > 0 and h.cancels == 0

def test_feasible_nav2_arc_uses_nav2_preview_instead_of_a_second_fixed_horizon(waiting_observation):
    (n, t, h) = waiting_observation
    _set_world_cell(n._local_grid, 0.3, 0.0, -1)
    command = Twist()
    (command.linear.x, command.angular.z) = (0.04, 0.08)
    n._on_nav2_command(command)
    n._forward_nav_command((0.0, 0.0, 0.0), t.wall, blocked_cost=99)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0

def test_follow_goal_uses_position_only_ackermann_tree(observation):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import yaml
    from ament_index_python.packages import get_package_share_directory
    (n, t) = observation
    n._latest_scan.ranges[20] = .5  # Obstacle before the intended stand-off.
    tick(n, t)
    goal = n._nav2_action.goals[-1]
    expected = Path(get_package_share_directory('piracer_navigation')) / 'behavior_trees/follow_navigate_to_pose.xml'
    assert goal.behavior_tree == str(expected)
    source_tree = Path(__file__).parents[2] / 'piracer_navigation/behavior_trees/follow_navigate_to_pose.xml'
    checker = ET.parse(source_tree).find('.//FollowPath').attrib['goal_checker_id']
    source = Path(__file__).parents[2] / 'piracer_navigation/config/nav2_params.yaml'
    config = yaml.safe_load(source.read_text())['controller_server']['ros__parameters'][checker]
    assert config['plugin'] == 'nav2_controller::PositionGoalChecker'
    assert config['xy_goal_tolerance'] == 0.15

def test_bbox_without_valid_new_point_keeps_the_owned_arc(waiting_observation):
    import copy
    (n, t, h) = waiting_observation
    n._memory_goal = (*n._last_seen[:2], 0.0)
    t.ros += 100000000
    t.wall += 0.1
    n._on_detections(NS(header=NS(stamp=Time(nanoseconds=t.ros).to_msg()), detections=[object()]))
    old = copy.deepcopy(n._latest_target_point)
    receipt = n._latest_target_received_monotonic
    n._on_target_point(old)
    assert h.cancels == 0 and n._latest_target_received_monotonic == receipt
    fresh = Twist()
    fresh.linear.x = 0.04
    n._on_nav2_command(fresh)
    tick(n, t)
    assert h.cancels == 0 and n._mode == module.NAV2
    reacquire_point(n, t)
    assert h.cancels == 0 and n._memory_goal is None
    assert n._goal_state == 'active'

def test_stale_new_point_cannot_replace_last_valid_observation(waiting_observation):
    import copy
    (n, t, h) = waiting_observation
    (old, receipt, remembered) = (n._latest_target_point, n._latest_target_received_monotonic, n._last_seen)
    t.ros += 800000000
    t.wall += 0.8
    point = copy.deepcopy(old)
    point.header.stamp = Time(nanoseconds=t.ros - 600000000).to_msg()
    n._on_target_point(point)
    assert n._latest_target_point is old and n._latest_target_received_monotonic == receipt
    assert n._last_seen == remembered and h.cancels == 0
    assert n._last_selected_twist.linear.x == 0 and n._nav2_latest_command is None

def test_exact_time_tf_delay_keeps_cache_zero_then_recovers_same_arc(waiting_observation):
    import copy
    from tf2_ros import ExtrapolationException
    (n, t, h) = waiting_observation
    (old, receipt) = (n._latest_target_point, n._latest_target_received_monotonic)
    t.ros += 600000000
    t.wall += 0.6
    point = copy.deepcopy(old)
    point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    transform = n._tf_buffer.transform
    n._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(ExtrapolationException('TF not at image stamp yet'))
    n._on_target_point(point)
    assert n._latest_target_point is old and n._latest_target_received_monotonic == receipt
    assert h.cancels == 0 and (not n._fault_latched)
    assert n._last_selected_twist.linear.x == 0 and n._nav2_latest_command is None
    n._tf_buffer.transform = transform
    n._retry_target_point()
    assert n._latest_target_point is point and n._last_seen[2] == t.ros
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0
    command = Twist()
    command.linear.x = 0.04
    n._on_nav2_command(command)
    tick(n, t)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0 and len(n._nav2_action.goals) == 1

@pytest.mark.parametrize('invalid', ['lost', 'tf'])
def test_measurement_tf_wait_cannot_override_robot_localization_failure(waiting_observation, invalid):
    import copy
    from tf2_ros import ExtrapolationException
    (n, t, h) = waiting_observation
    point = copy.deepcopy(n._latest_target_point)
    t.ros += 100000000
    t.wall += 0.1
    point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(ExtrapolationException('not ready'))
    if invalid == 'lost':
        n._odom_info.lost = True
    else:
        n._tf_pose = lambda *_: (_ for _ in ()).throw(module.TransformException('stale current robot TF'))
    n._on_target_point(point)
    assert h.cancels == 1 and n._last_seen is None
    assert n._last_selected_twist.linear.x == 0

def test_rejected_memory_point_stays_ineligible_after_tf_recovery_and_drain(waiting_observation):
    import copy
    (n, t, h) = waiting_observation
    n._memory_goal = (*n._last_seen[:2], 0.0)
    t.ros += 100000000
    t.wall += 0.1
    point = copy.deepcopy(n._latest_target_point)
    point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._latest_detections = NS(header=NS(stamp=point.header.stamp), detections=[object()])
    transform = n._tf_buffer.transform
    n._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(module.TransformException('missing map TF'))
    n._on_target_point(point)
    assert n._latest_target_received_monotonic is None and h.cancels == 1
    n._tf_buffer.transform = transform
    n._on_nav2_goal_result(NS(result=lambda : NS(status=GoalStatus.STATUS_CANCELED)))
    n._active_actions.clear()
    t.ros += 400000000
    t.wall += 0.4
    n._finish_goal_drain(t.ros, t.wall)
    n._cycle_counts = {'timer_calls': 0, 'ros_time_zero_cycles': 0}
    n._record_cycle = lambda *_: None
    n._last_ros_time_ns = n._last_ros_time_advance_monotonic = None
    n._control_cycle()
    assert n._last_selected_twist.linear.x == 0
    assert n._latest_target_received_monotonic is None
    point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    n._on_target_point(point)
    assert n._latest_target_received_monotonic is None and n._fault_latched


def test_target_data_wait_stays_zero_when_the_approach_map_changes(waiting_observation):
    (n, t, h) = waiting_observation
    n._memory_goal = (0.6, 0.0, 0.0)
    _set_world_cell(n._global_grid, *n._last_seen[:2], 100)
    n._safe_stop('ros_clock_stalled')
    assert h.cancels == 0
    _set_world_cell(n._global_grid, 0.65, 0.0, 100)
    n._safe_stop('ros_clock_stalled')
    assert h.cancels == 0 and (not n._fault_latched)
    assert n._last_selected_twist.linear.x == 0 and n._nav2_latest_command is None

def test_owned_nav2_route_uses_controller_collision_stop_in_unknown_cells(waiting_observation):
    (n, t, h) = waiting_observation
    n._memory_goal = (0.6, 0.0, 0.0)
    t.ros += 100000000
    t.wall += 0.1
    n._on_detections(NS(header=NS(stamp=Time(nanoseconds=t.ros).to_msg()), detections=[]))
    n._local_grid.data[:] = [-1] * len(n._local_grid.data)
    tick(n, t)
    assert n._last_selected_twist.linear.x == pytest.approx(0.04)
    assert h.cancels == 0 and len(n._nav2_action.goals) == 1
    n._on_nav2_command(Twist())
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0 and h.cancels == 0
    n._active_actions['navigate_to_pose'] = {b'foreign': (2, t.ros)}
    tick(n, t)
    assert n._last_selected_twist.linear.x == 0 and h.cancels == 1



def accepted(n, t, number):
    handle=Handle(number)
    n._on_nav2_goal_response(NS(result=lambda:handle))
    n._active_actions['navigate_to_pose']={bytes(handle.goal_id.uuid):(2,t.ros)}
    n._active_actions['follow_path']={bytes([number]):(2,t.ros)}
    command=Twist();command.linear.x=.04
    n._on_nav2_command(command)
    tick(n,t)
    assert n._last_selected_twist.linear.x==pytest.approx(.04)
    return handle


@pytest.mark.parametrize('x,z',[(1.5,0.),(-1.,0.),(3.,float('nan')),(3.,0.)])
def test_pending_measurement_is_rechecked_after_base_tf_arrives(waiting_observation,x,z):
    import copy
    from tf2_ros import ExtrapolationException
    n,t,h=waiting_observation
    endpoint=n._memory_goal=(.9,1.1,-.5)
    t.ros+=50_000_000
    t.wall+=.05
    message=copy.deepcopy(n._latest_target_point)
    message.header.stamp=Time(nanoseconds=t.ros).to_msg()
    first_receipt=t.wall
    available=False
    def transform(m,frame):
        if frame=='base_link' and not available:
            raise ExtrapolationException('base transform not ready')
        return NS(header=m.header,point=NS(x=x if frame=='base_link' else 4.,y=0.,z=z if frame=='base_link' else 0.))
    n._tf_buffer.transform=transform
    n._on_target_point(message)
    assert n._pending_target_point is not None
    t.ros+=50_000_000
    t.wall+=.05
    available=True
    n._retry_target_point()
    if x<=2 or not math.isfinite(z):
        assert n._fault_latched and h.cancels==1
        assert n._latest_target_received_monotonic is None
        assert n._last_selected_twist.linear.x==0
    else:
        assert h.cancels==0 and n._memory_goal is None
        assert n._latest_target_received_monotonic==first_receipt
        assert n._latest_target_point is message
        n._latest_detections=NS(header=NS(stamp=message.header.stamp),detections=[object()])
        n._integrated_cycle(NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)
        assert n._memory_goal is None and n._goal_handle is h
        assert n._last_selected_source!='direct'
