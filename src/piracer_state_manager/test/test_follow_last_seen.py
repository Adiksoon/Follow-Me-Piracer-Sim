"""Exercise the real FOLLOW action handoff with controlled ROS and wall clocks."""
from types import SimpleNamespace as NS

import pytest
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import TransformStamped, Twist, PointStamped
from sensor_msgs.msg import LaserScan
from rclpy.time import Time

from piracer_state_manager import follow_node as module
from piracer_state_manager.follow_nav2_helpers import DIRECT, NAV2, SAFE_STOP
from test_follow_nav2_integration import _grid, _integrated_cycle_node, _stamp_grid, _set_world_cell


class Handle:
    accepted = True

    def __init__(self, identifier):
        self.goal_id = NS(uuid=list(bytes([identifier]) * 16))
        self.cancels = 0

    def get_result_async(self):
        return NS(add_done_callback=lambda callback: None)

    def cancel_goal_async(self):
        self.cancels += 1


@pytest.fixture
def episode(monkeypatch):
    grid = _stamp_grid(_grid(width=180, height=120, origin=(-2., -3., 0.)), 5_000_000_000)
    world = _stamp_grid(_grid(width=180, height=120, origin=(-2., -3., 0.)), 5_000_000_000, 'map')
    node = _integrated_cycle_node(local_grid=grid, global_grid=world)
    timer = NS(ros=5_000_000_000, wall=100.)
    monkeypatch.setattr(module, 'time', NS(monotonic=lambda: timer.wall))
    node.get_clock = lambda: NS(now=lambda: Time(nanoseconds=timer.ros))
    node.get_logger = lambda: NS(info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None)
    node._controller.max_speed_mps = 0.075
    expected = {'/cmd_vel': 'velocity_smoother', '/cmd_vel_nav': 'controller_server',
                '/cmd_vel_follow': 'piracer_follow'}
    node.get_publishers_info_by_topic = lambda t: [NS(node_namespace='/', node_name=expected[t])]
    node.get_subscriptions_info_by_topic = lambda t: [
        NS(node_namespace='/', node_name='bt_navigator_navigate_to_pose_rclcpp_node')]
    node._active_actions = {}
    node._last_seen = node._memory_goal = None

    # Two distinct costmaps block forward travel and trigger one standoff action.
    _set_world_cell(grid, 0.48, 0., 100)
    direct = NS(linear_x=.05, angular_z=0.)
    node._integrated_cycle(direct, node._test_target, timer.ros, timer.wall)
    grid.header.stamp.nanosec += 1
    world.header.stamp.nanosec += 1
    node._integrated_cycle(direct, node._test_target, timer.ros, timer.wall)
    assert len(node._nav2_action.goals) == 1 and node._goal_state == 'sending'
    old = Handle(1)
    node._on_nav2_goal_response(NS(result=lambda: old))
    node._active_actions['navigate_to_pose'] = {bytes(old.goal_id.uuid): (2, timer.ros)}
    node._active_actions['follow_path'] = {b'first-path': (2, timer.ros)}
    scan = LaserScan()
    scan.header.frame_id = 'sim_camera_link'
    scan.header.stamp = Time(nanoseconds=timer.ros).to_msg()
    scan.angle_min, scan.angle_increment = -.4, .02
    scan.range_min, scan.range_max = .05, 3.
    scan.ranges = [2.8] * 41
    node._latest_scan = scan
    static_tf = TransformStamped()
    static_tf.transform.rotation.w = 1.
    static_tf.transform.translation.x = .236
    node._tf_buffer.lookup_transform = lambda *_: static_tf
    command = Twist()
    command.linear.x = .04
    node._on_nav2_command(command)
    node._forward_nav_command((0., 0., 0.), timer.wall)
    assert node._last_selected_twist.linear.x == pytest.approx(.04)
    assert node._mode == NAV2
    timer.ros += 900_000_000
    timer.wall += .9
    node._latest_detections = NS(header=NS(stamp=Time(nanoseconds=timer.ros).to_msg()), detections=[])
    node._detection_receipt = timer.wall
    return node, timer, old


def tick(n, t):
    return n._try_last_seen(t.ros, t.wall, False)


def reacquire_point(n, t):
    t.ros += 1_000_000
    t.wall += .001
    point = PointStamped()
    point.header.frame_id = "sim_camera_optical_frame"
    point.header.stamp = Time(nanoseconds=t.ros).to_msg()
    point.point.x, point.point.y, point.point.z = .01, .01, 3.
    n._on_target_point(point)


def valid_cycle(n, t, seconds=.1, command=None):
    """Deliver a new image-time point and current maps/Nav2 command."""
    import copy
    t.ros+=round(seconds*1e9);t.wall+=seconds
    point=copy.deepcopy(n._latest_target_point)
    point.header.stamp=Time(nanoseconds=t.ros).to_msg()
    n._on_target_point(point)
    for grid in (n._local_grid,n._global_grid):
        grid.header.stamp=Time(nanoseconds=t.ros).to_msg()
    nav=Twist();nav.linear.x=.04
    n._on_nav2_command(nav)
    n._integrated_cycle(command or NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)


def prepare_control_cycle(n, t):
    from piracer_state_manager.follow_control import FollowPController
    from rclpy.clock import ClockType
    n._controller = FollowPController(max_speed_mps=.075)
    n._cycle_counts = {'timer_calls': 0, 'ros_time_zero_cycles': 0}
    n._record_cycle = lambda *_: None
    n._publish_control_sample = lambda *_: None
    n._last_ros_time_ns = n._last_ros_time_advance_monotonic = None
    n.get_clock = lambda: NS(now=lambda: Time(nanoseconds=t.ros), clock_type=ClockType.SYSTEM_TIME)
    original = n.get_parameter
    n.get_parameter = lambda key: NS(value='base_link') if key == 'base_frame' else original(key)
    original_transform = n._tf_buffer.transform
    n._tf_buffer.transform = lambda message, frame: NS(
        header=NS(frame_id=frame, stamp=message.header.stamp),
        point=original_transform(message, frame).point)


def test_normal_goal_success_holds_without_a_false_failure_latch(episode):
    n, t, _ = episode
    n._on_nav2_goal_result(NS(result=lambda: NS(status=GoalStatus.STATUS_SUCCEEDED)))
    n._active_actions.clear()
    t.ros += 400_000_000
    t.wall += .4
    n._finish_goal_drain(t.ros, t.wall)
    assert not n._fault_latched and n._mode == module.NAV2_PENDING
    assert n._nav2_latest_command is None and not n._goal_unresolved

@pytest.mark.parametrize('mode,state',[(DIRECT,'idle'),(module.NAV2_PENDING,'sending'),(SAFE_STOP,'unknown')])
def test_detection_loss_does_not_start_new_episode_without_active_detour(episode,mode,state):
    n,t,old=episode;n._mode=mode;n._goal_state=state
    assert not tick(n,t) and n._memory_goal is None
    assert old.cancels==0 and len(n._nav2_action.goals)==1


@pytest.mark.parametrize("planning", ["no_motion", "no_path", "accepted_only"])
def test_planning_without_an_executing_detour_cannot_arm_memory(episode, planning):
    node, timer, old = episode
    if planning == "no_motion":
        node._detour_forwarded = False
    elif planning == "no_path":
        node._active_actions["follow_path"].clear()
    else:
        node._active_actions["navigate_to_pose"][bytes(old.goal_id.uuid)] = (1, timer.ros)
    tick(node, timer)
    assert node._memory_goal is None and node._last_selected_twist.linear.x == 0
    assert old.cancels == 0 and len(node._nav2_action.goals) == 1
    assert node._nav2_latest_command is None


@pytest.mark.parametrize("gap", ["clock", "detection", "point"])
def test_short_confirmation_gap_keeps_normal_owned_goal_zero_then_lost_cancels(episode, gap):
    n, t, old = episode
    if gap == "detection":
        n._latest_detections = None
    elif gap == "point":
        n._latest_detections.detections = [object()]
    if gap == "point":
        command=Twist();command.linear.x=.04
        n._on_nav2_command(command)
    n._try_last_seen(t.ros, t.wall, gap == "clock")
    assert old.cancels == 0
    if gap == "point":
        assert n._memory_goal is not None and n._last_selected_twist.linear.x==pytest.approx(.04)
    else:
        assert n._memory_goal is None
        assert n._last_selected_twist.linear.x == 0 and n._nav2_latest_command is None
    n._on_odom_info(NS(lost=True))
    assert old.cancels == 1 and n._last_selected_twist.linear.x == 0


def test_local_endpoint_requires_full_grid_coverage_and_free_footprint():
    from piracer_state_manager.follow_nav2_helpers import footprint_clear
    grid = _grid(width=40, height=40, origin=(-1., -1., 0.))
    for x in (1.8, 1.01, .95, -1.):
        assert not footprint_clear(grid, x, 0., 0., blocked_cost=99)
    assert footprint_clear(grid, .5, 0., 0., blocked_cost=99)
    _set_world_cell(grid, .7, 0., 99)
    assert not footprint_clear(grid, .5, 0., 0., blocked_cost=99)

@pytest.mark.parametrize('path_state', ['absent', 'old', 'ambiguous'])
def test_refreshed_normal_goal_rejects_previous_smoother_tail(episode, path_state):
    n, t, _ = episode
    n._detour_forwarded = False
    n._sent_goal_stamp_ns = t.ros
    paths = {} if path_state == 'absent' else {b'old': (2, t.ros - 1)}
    if path_state == 'ambiguous':
        paths = {b'a': (2, t.ros), b'b': (2, t.ros)}
    n._active_actions['follow_path'] = paths
    tail = Twist()
    tail.linear.x = .04
    n._on_nav2_command(tail)
    n._forward_nav_command((0., 0., 0.), t.wall)
    assert n._nav2_latest_command is None and n._last_selected_twist.linear.x == 0
    assert not n._detour_forwarded
    n._active_actions['follow_path'] = {b'new': (2, t.ros)}
    n._forward_nav_command((0., 0., 0.), t.wall)
    assert n._last_selected_twist.linear.x == 0 and not n._detour_forwarded
    n._on_nav2_command(tail)
    n._forward_nav_command((0., 0., 0.), t.wall)
    assert n._last_selected_twist.linear.x == pytest.approx(.04) and n._detour_forwarded


@pytest.mark.parametrize('motion_forwarded', [False, True])
def test_normal_abort_stops_without_a_second_side_after_partial_motion(episode, motion_forwarded):
    n, t, _ = episode
    n._detour_forwarded = motion_forwarded
    n._on_nav2_goal_result(NS(result=lambda: NS(status=GoalStatus.STATUS_ABORTED)))
    assert n._fault_latched and n._last_selected_twist.linear.x == 0
    assert len(n._nav2_action.goals) == 1


@pytest.mark.parametrize('x', [2., 0., -1.])
@pytest.mark.parametrize('stage', ['arc', 'target'])
def test_frozen_observation_target_still_stops_for_live_minimum_distance(episode, x, stage):
    n, t, old = episode
    n._observation_stage = stage
    n._memory_goal = (*n._last_seen[:2], 0.)
    n._tf_buffer.transform = lambda *_: NS(point=NS(x=x, y=0., z=0.))
    reacquire_point(n, t)
    assert n._mode == SAFE_STOP and n._last_selected_twist.linear.x == 0
    assert n._mode_reason == 'target_inside_min_distance'
    assert old.cancels == 1 and n._memory_goal is None


@pytest.mark.parametrize('with_overlap', [False, True])
def test_current_path_replacement_discards_cache_before_any_forward_tick(episode, with_overlap):
    from action_msgs.msg import GoalStatus, GoalStatusArray
    n, t, handle = episode
    old = GoalStatus()
    old.goal_info.goal_id.uuid = [3] * 16
    old.goal_info.stamp = Time(nanoseconds=t.ros).to_msg()
    old.status = GoalStatus.STATUS_EXECUTING
    initial = GoalStatusArray(status_list=[old])
    n._on_action_status('follow_path', initial)
    twist = Twist()
    twist.linear.x = .04
    n._on_nav2_command(twist)
    assert n._nav2_latest_command is twist
    new = GoalStatus()
    new.goal_info.goal_id.uuid = [4] * 16
    new.goal_info.stamp = Time(nanoseconds=t.ros).to_msg()
    new.status = GoalStatus.STATUS_ACCEPTED
    if with_overlap:
        n._on_action_status('follow_path', GoalStatusArray(status_list=[old, new]))
        assert n._exclusive_nav_stream() and not n._memory_command_owned()
    new.status = GoalStatus.STATUS_EXECUTING
    n._on_action_status('follow_path', GoalStatusArray(status_list=[new]))
    assert n._memory_command_owned() and n._nav2_latest_command is None
    assert n._nav2_latest_command_wall is None
    n._forward_nav_command((0., 0., 0.), t.wall)
    assert n._last_selected_twist.linear.x == 0 and handle.cancels == 0
    n._on_nav2_command(twist)
    n._forward_nav_command((0., 0., 0.), t.wall)
    assert n._last_selected_twist.linear.x == pytest.approx(.04)



def forward_memory(n,t):
    command=Twist();command.linear.x=.04
    n._on_nav2_command(command)
    return tick(n,t)


@pytest.mark.parametrize('positive',[False,True])
def test_point_gap_keeps_exactly_one_owned_goal_for_either_detection_result(episode,positive):
    n,t,h=episode
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._latest_scan.ranges=[2.8]*len(n._latest_scan.ranges)
    n._latest_detections.detections=[object()] if positive else []
    original=n._last_seen
    assert forward_memory(n,t)
    assert n._memory_goal[:2]==pytest.approx((n._nav2_action.goals[0].pose.pose.position.x,
                                                     n._nav2_action.goals[0].pose.pose.position.y))
    assert n._goal_handle is h and h.cancels==0 and len(n._nav2_action.goals)==1
    assert n._last_seen==original and n._last_selected_source=='nav2'
    assert n._last_selected_twist.linear.x==pytest.approx(.04)


def test_occupied_remembered_point_does_not_replace_committed_observation_goal(episode):
    n,t,h=episode
    n._last_seen=(5.,0.,*n._last_seen[2:])
    _set_world_cell(n._global_grid, 5.,0.,100)
    forward_memory(n,t)
    assert n._memory_goal[:2]==pytest.approx((n._nav2_action.goals[0].pose.pose.position.x,
                                                     n._nav2_action.goals[0].pose.pose.position.y))
    assert n._goal_handle is h and h.cancels==0 and len(n._nav2_action.goals)==1


@pytest.mark.parametrize('clock',['ros','wall'])
def test_detection_packets_never_renew_the_original_memory_budget(episode,clock):
    n,t,h=episode
    forward_memory(n,t)
    original=n._last_seen
    if clock=='ros':t.ros=original[2]+120_100_000_000
    else:t.wall=original[3]+120.1
    n._on_detections(NS(header=NS(stamp=Time(nanoseconds=t.ros).to_msg()),detections=[object()]))
    tick(n,t)
    assert n._last_seen is None and n._memory_goal is None
    assert h.cancels==1 and n._last_selected_twist.linear.x==0


def test_safe_reacquisition_keeps_nav2_while_the_direct_path_is_blocked(episode):
    n,t,h=episode
    forward_memory(n,t)
    reacquire_point(n,t)
    assert n._latest_target_received_monotonic==t.wall
    assert n._memory_goal is None and h.cancels==0 and len(n._nav2_action.goals)==1
    n._integrated_cycle(NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)
    assert n._goal_state=='active' and n._goal_unresolved and h.cancels==0
    assert n._last_selected_source=='nav2'


@pytest.mark.parametrize('fault',['lost','tf','stalled','future_detection','future_point',
                                  'regression','foreign_goal','foreign_publisher','blocked_scan','stale_twist','disable'])
def test_memory_continuation_still_stops_on_real_faults(episode,fault):
    n,t,h=episode
    forward_memory(n,t)
    if fault=='lost':n._on_odom_info(NS(lost=True))
    elif fault=='tf':n._tf_pose=lambda *_:(_ for _ in ()).throw(module.TransformException('lost TF'))
    elif fault=='future_detection':n._latest_detections.header.stamp=Time(nanoseconds=t.ros+1).to_msg()
    elif fault=='future_point':n._latest_target_point.header.stamp=Time(nanoseconds=t.ros+1).to_msg()
    elif fault=='regression':n._last_integrated_ros_ns=t.ros+1
    elif fault=='foreign_goal':n._active_actions['navigate_to_pose'][b'foreign']=(2,t.ros)
    elif fault=='foreign_publisher':n.get_publishers_info_by_topic=lambda _:[NS(node_namespace='/',node_name='foreign')]
    elif fault=='blocked_scan':n._latest_scan.ranges=[.1]*len(n._latest_scan.ranges)
    elif fault=='stale_twist':n._nav2_latest_command_wall=t.wall-.3
    elif fault=='disable':n._set_enabled(NS(data=False),NS())
    n._try_last_seen(t.ros,t.wall,fault=='stalled')
    assert n._command_publisher.messages[-1].linear.x==0
    if fault!='disable':
        assert n._last_selected_twist.linear.x==0 and n._last_selected_source=='stop'
    assert len(n._nav2_action.goals)==1


@pytest.mark.parametrize('status',[GoalStatus.STATUS_ABORTED,GoalStatus.STATUS_UNKNOWN])
def test_failed_committed_memory_goal_never_retries(episode,status):
    n,t,h=episode
    forward_memory(n,t)
    n._on_nav2_goal_result(NS(result=lambda:NS(status=status)))
    assert n._fault_latched and n._last_selected_twist.linear.x==0
    tick(n,t)
    assert len(n._nav2_action.goals)==1


def test_observation_success_drains_both_clocks_without_another_goal(episode):
    n, t, h = episode
    prepare_control_cycle(n, t)
    forward_memory(n, t)
    _set_world_cell(n._global_grid, *n._last_seen[:2], 100)
    n._on_nav2_goal_result(NS(result=lambda: NS(status=GoalStatus.STATUS_SUCCEEDED)))
    assert n._memory_goal is None and n._last_seen is None
    assert n._nav2_goal_offset is None and n._nav2_goal_target_map is None
    assert n._goal_finish_action == 'recover' and n._behavior_status() == 'AVOIDING'
    assert n._last_selected_twist.linear.x == 0
    n._finish_goal_drain(t.ros, t.wall + .4)
    assert n._goal_unresolved and n._behavior_status() == 'AVOIDING'
    t.ros += 400_000_000
    n._finish_goal_drain(t.ros, t.wall)
    assert n._goal_unresolved and n._behavior_status() == 'AVOIDING'
    t.wall += .4
    n._active_actions.clear()
    n._finish_goal_drain(t.ros, t.wall)
    assert not n._goal_unresolved and not n._fault_latched
    assert n._behavior_status() == 'DIRECT'
    for _ in range(4):
        n._control_cycle()
        assert n._last_selected_twist.linear.x == 0
    assert len(n._nav2_action.goals) == 1 and h.cancels == 0


@pytest.mark.parametrize('result',[GoalStatus.STATUS_CANCELED,GoalStatus.STATUS_SUCCEEDED])
def test_clear_path_recovery_wins_over_racing_terminal_result(episode,result):
    n,t,h=episode
    forward_memory(n,t)
    reacquire_point(n,t)
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._latest_scan.ranges=[2.8]*len(n._latest_scan.ranges)
    for _ in range(4):
        valid_cycle(n,t,.2)
    assert n._goal_cancel_reason=='direct_path_clear' and h.cancels==1
    n._on_nav2_goal_result(NS(result=lambda:NS(status=result)))
    assert not n._fault_latched and n._goal_finish_action=='recover'
    assert n._memory_goal is None and n._last_selected_twist.linear.x==0
    n._active_actions.clear()
    t.ros+=400_000_000;t.wall+=.4
    n._finish_goal_drain(t.ros,t.wall)
    assert not n._goal_unresolved and len(n._nav2_action.goals)==1


@pytest.mark.parametrize('stage', ['active', 'draining', 'idle'])
def test_reacquisition_uses_normal_control_without_a_historical_leg(episode, stage):
    n, t, h = episode
    prepare_control_cycle(n, t)
    forward_memory(n, t)
    if stage != 'active':
        n._on_nav2_goal_result(NS(result=lambda: NS(status=GoalStatus.STATUS_SUCCEEDED)))
    if stage == 'idle':
        n._active_actions.clear()
        t.ros += 400_000_000
        t.wall += .4
        n._finish_goal_drain(t.ros, t.wall)
    reacquire_point(n, t)
    assert n._memory_goal is None and n._latest_target_received_monotonic == t.wall
    n._control_cycle()
    assert not n._fault_latched and n._last_seen is not None
    if stage == 'active':
        assert n._goal_handle is h and h.cancels == 0
    elif stage == 'draining':
        assert n._goal_unresolved and n._goal_finish_action == 'recover'
        assert n._last_selected_twist.linear.x == 0
    for goal in n._nav2_action.goals:
        assert (goal.pose.pose.position.x, goal.pose.pose.position.y) != pytest.approx(n._last_seen[:2])


@pytest.mark.parametrize('interference', [None, 'client', 'namespace', 'navigate', 'paths', 'publisher'])
def test_idle_acquire_client_preserves_owned_memory_and_foreign_rejection(episode, interference):
    n, t, handle = episode
    clients = [NS(node_namespace='/', node_name='bt_navigator_navigate_to_pose_rclcpp_node'),
               NS(node_namespace='/', node_name='piracer_acquire')]
    n.get_subscriptions_info_by_topic = lambda _: clients
    n._local_grid.data[:] = [0] * len(n._local_grid.data)
    n._latest_scan.ranges = [2.8] * len(n._latest_scan.ranges)
    if interference == 'client':
        clients.append(NS(node_namespace='/', node_name='foreign_client'))
    elif interference == 'namespace':
        clients[1].node_namespace = '/foreign'
    elif interference == 'navigate':
        n._active_actions['navigate_to_pose'][b'foreign'] = (2, t.ros)
    elif interference == 'paths':
        n._active_actions['follow_path'][b'foreign'] = (2, t.ros)
    elif interference == 'publisher':
        original = n.get_publishers_info_by_topic
        n.get_publishers_info_by_topic = lambda topic: original(topic) + [
            NS(node_namespace='/', node_name='foreign_publisher')]
    assert forward_memory(n, t)
    assert n._goal_handle is handle and len(n._nav2_action.goals) == 1
    if interference is None:
        assert n._memory_goal is not None and n._last_selected_source == 'nav2'
        assert n._last_selected_twist.linear.x == pytest.approx(.04)
        assert handle.cancels == 0
    else:
        assert n._last_selected_twist.linear.x == 0
        assert n._memory_goal is None


def test_disable_ack_waits_for_observation_result_drain(episode):
    n, t, _ = episode
    forward_memory(n, t)
    n._on_nav2_goal_result(NS(result=lambda: NS(status=GoalStatus.STATUS_SUCCEEDED)))
    assert not n._set_enabled(NS(data=False), NS()).success
    n._finish_goal_drain(t.ros, t.wall + .4)
    assert not n._set_enabled(NS(data=False), NS()).success
    t.ros += 400_000_000
    t.wall += .4
    n._finish_goal_drain(t.ros, t.wall)
    assert n._set_enabled(NS(data=False), NS()).success
    assert not n._goal_unresolved and n._last_selected_twist.linear.x == 0


@pytest.mark.parametrize('can_acquire', [True, False])
def test_observation_completion_releases_normal_state_handoff(episode, can_acquire):
    from piracer_state_manager.state_machine import ACQUIRE, FOLLOW, SEARCH, StateMachine, BehaviorEnableCoordinator
    from test_state_machine import frame
    n, t, _ = episode
    forward_memory(n, t)
    machine = StateMachine()
    for stamp in range(1, 8):
        frame(machine, stamp, True, stamp / 10)
    assert machine.state == FOLLOW
    for stamp in range(8, 13):
        machine.observe(stamp, False, stamp / 10, can_acquire=can_acquire,
                        target_visible=False, hold_follow=n._behavior_status() == 'AVOIDING')
    assert machine.state == FOLLOW and machine.miss_streak == 0
    n._on_nav2_goal_result(NS(result=lambda: NS(status=GoalStatus.STATUS_SUCCEEDED)))
    assert n._behavior_status() == 'AVOIDING'
    t.ros += 400_000_000
    t.wall += .4
    n._finish_goal_drain(t.ros, t.wall)
    assert n._behavior_status() == 'DIRECT'
    for stamp in range(13, 16):
        machine.observe(stamp, False, stamp / 10, can_acquire=can_acquire,
                        target_visible=False, hold_follow=n._behavior_status() == 'AVOIDING')
        if stamp < 15:
            assert machine.state == FOLLOW
    expected = ACQUIRE if can_acquire else SEARCH
    assert machine.state == expected
    coordinator = BehaviorEnableCoordinator(('search', 'acquire', 'follow'))
    coordinator.confirmed.update(search=False, acquire=False, follow=True)
    ready = {name: True for name in coordinator.behavior_names}
    target = expected.lower()
    assert coordinator.next_request(target, ready) == ('follow', False)
    assert coordinator.begin('follow', False)
    assert coordinator.next_request(target, ready) is None
    assert not coordinator.complete('follow', False, False)
    assert coordinator.next_request(target, ready) == ('follow', False)
    assert coordinator.begin('follow', False)
    response = n._set_enabled(NS(data=False), NS())
    assert response.success and coordinator.complete('follow', False, response.success)
    assert coordinator.next_request(target, ready) == (target, True)
    assert len(n._nav2_action.goals) == 1
