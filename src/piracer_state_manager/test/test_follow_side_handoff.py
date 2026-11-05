"""Stand-off selection and a confirmed safe return from Nav2 to DIRECT."""
import math
from types import SimpleNamespace as NS

import pytest
from action_msgs.msg import GoalStatus
from rclpy.time import Time

from piracer_state_manager import follow_node as module
from piracer_state_manager.follow_nav2_helpers import clear_stand_off_goal, footprint_clear, stand_off_goal
from test_follow_last_seen import episode, valid_cycle
from test_follow_nav2_integration import _grid, _rasterize_box_bounds, _set_world_cell


def test_first_danger_keeps_the_radial_endpoint_when_it_is_free():
    grid=_grid(width=160,height=140,origin=(-2.,-3.,0.))
    _rasterize_box_bounds(grid,(1.7,2.,-.6,.6))
    target=(2.839554,.013049)
    radial=stand_off_goal(0.,0.,*target)
    assert footprint_clear(grid,*radial)
    side=clear_stand_off_goal(grid,radial,*target,0.,approach_from=(0.,0.))
    assert side==pytest.approx(radial) and abs(side[1])<.01
    assert math.hypot(side[0]-target[0],side[1]-target[1])==pytest.approx(2.25)
    assert math.atan2(side[1],side[0])==pytest.approx(side[2])
    assert side==clear_stand_off_goal(grid,radial,*target,0.,approach_from=(0.,0.))


@pytest.fixture
def approach(episode):
    n,t,h=episode
    t.ros,t.wall=5_000_000_000,100.
    n._goal_handle=None
    n._goal_state,n._goal_unresolved='idle',False
    n._goal_cancel_reason=None
    n._nav2_goal_offset=n._nav2_goal_target_map=None
    n._nav2_action.goals=[]
    n._active_actions.clear()
    n._global_grid.data[:]=[0]*len(n._global_grid.data)
    target=[2.839554,.013049]
    n._tf_buffer.transform=lambda message,frame:NS(header=message.header,point=NS(x=target[0],y=target[1],z=0.))
    n._start_nav2_goal()
    return n,t,target


def finish(n,t,h):
    n._goal_handle=h
    n._goal_state,n._goal_unresolved='active',True
    n._on_nav2_goal_result(NS(result=lambda:NS(status=GoalStatus.STATUS_SUCCEEDED)))
    n._active_actions.clear()
    t.ros+=400_000_000;t.wall+=.4
    n._finish_goal_drain(t.ros,t.wall)


def fresh(n,t):
    n._latest_target_point.header.stamp=Time(nanoseconds=t.ros).to_msg()
    n._latest_target_received_monotonic=t.wall
    for grid in (n._local_grid,n._global_grid):
        grid.header.stamp=Time(nanoseconds=t.ros).to_msg()


def test_stationary_success_holds_zero_without_repeating_goal(approach):
    n,t,target=approach
    from test_follow_last_seen import Handle
    finish(n,t,Handle(10))
    for step in range(4):
        t.ros+=1_000_000_000;t.wall+=1.
        fresh(n,t)
        n._maybe_refresh_goal(target,t.wall)
        n._start_nav2_goal()
        assert len(n._nav2_action.goals)==1
        assert not n._fault_latched and n._last_selected_twist.linear.x==0


def test_refresh_reselects_standoff_for_the_current_target(approach):
    n,t,target=approach
    from test_follow_last_seen import Handle
    first=n._nav2_action.goals[0].pose.pose
    assert 2 * math.atan2(first.orientation.z, first.orientation.w) == pytest.approx(
        math.atan2(first.position.y, first.position.x))
    finish(n,t,Handle(10))
    for dx in [.1,.1,.1]:
        t.ros+=4_000_000_000;t.wall+=4.
        target[0]+=dx
        fresh(n,t)
        n._maybe_refresh_goal(target,t.wall)
        n._start_nav2_goal()
    assert len(n._nav2_action.goals)==2
    goal=n._nav2_action.goals[-1].pose.pose
    assert math.hypot(goal.position.x-target[0],goal.position.y-target[1])==pytest.approx(2.25)
    assert goal.position.x-first.position.x==pytest.approx(.3,abs=1e-4)
    assert 2*math.atan2(goal.orientation.z,goal.orientation.w)==pytest.approx(
        math.atan2(goal.position.y,goal.position.x))


def test_moving_target_can_use_fallback_then_return_to_radial_goal(approach):
    from test_follow_last_seen import Handle
    n,t,target=approach
    _rasterize_box_bounds(n._global_grid,(1.7,2.,-.6,.6))
    for value in (3.9,5.):
        previous=n._nav2_action.goals[-1].pose.pose
        pose=(previous.position.x,previous.position.y,
              2*math.atan2(previous.orientation.z,previous.orientation.w))
        finish(n,t,Handle(10))
        n._tf_pose=lambda *_,p=pose:p
        target[0]=value
        t.ros+=4_000_000_000;t.wall+=4.
        fresh(n,t)
        n._maybe_refresh_goal(target,t.wall)
        radial=stand_off_goal(*pose[:2],*target)
        n._start_nav2_goal()
        goal=n._nav2_action.goals[-1].pose.pose
        actual=(goal.position.x,goal.position.y,
                2*math.atan2(goal.orientation.z,goal.orientation.w))
        assert not n._fault_latched and footprint_clear(n._global_grid,*actual)
        if value==3.9:
            assert not footprint_clear(n._global_grid,*radial)
            assert abs(goal.position.y)>.6
        else:
            assert actual==pytest.approx(radial)
    assert len(n._nav2_action.goals)==3


def test_duplicate_point_and_timer_ticks_cannot_confirm_direct_return(episode):
    import copy
    from geometry_msgs.msg import Twist
    n,t,h=episode
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._latest_scan.ranges=[2.8]*len(n._latest_scan.ranges)
    valid_cycle(n,t,0.)
    first_stamp=n._direct_clear_since_ns
    image=copy.deepcopy(n._latest_target_point)
    first_receipt=n._latest_target_received_monotonic
    ros,wall=t.ros,t.wall
    for step in range(1,6):
        t.ros=ros+step*100_000_000;t.wall=wall+step*.1
        n._on_target_point(copy.deepcopy(image))
        nav=Twist();nav.linear.x=.04;n._on_nav2_command(nav)
        n._integrated_cycle(NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)
        assert n._direct_clear_since_ns==first_stamp and h.cancels==0
        assert n._latest_target_received_monotonic==first_receipt
    valid_cycle(n,t,0.)
    assert h.cancels==1 and n._goal_cancel_reason=='direct_path_clear'
    assert n._last_selected_twist.linear.x==0


def test_danger_resets_confirmation_and_point_loss_during_cancel_cannot_restart_memory(episode):
    n,t,h=episode
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._latest_scan.ranges=[2.8]*len(n._latest_scan.ranges)
    valid_cycle(n,t,0.)
    valid_cycle(n,t,.2)
    _set_world_cell(n._global_grid,.5,0.,100)
    valid_cycle(n,t,.2)
    assert n._direct_clear_since_ns is None and h.cancels==0
    n._global_grid.data[:]=[0]*len(n._global_grid.data)
    valid_cycle(n,t,.2)
    valid_cycle(n,t,.4)
    assert h.cancels==0
    valid_cycle(n,t,.1)
    assert h.cancels==1
    t.ros+=600_000_000;t.wall+=.6
    assert not n._try_last_seen(t.ros,t.wall,False)
    n._stop_for_input_error('stale_received_target')
    assert n._direct_clear_since_ns is None and n._last_selected_twist.linear.x==0
    assert n._memory_goal is None and len(n._nav2_action.goals)==1
    n._on_nav2_goal_result(NS(result=lambda:NS(status=GoalStatus.STATUS_CANCELED)))
    n._active_actions.clear()
    t.ros+=400_000_000;t.wall+=.4
    n._finish_goal_drain(t.ros,t.wall)
    n._stop_for_input_error('stale_received_target')
    valid_cycle(n,t,0.)
    assert n._mode==module.NAV2_PENDING and n._last_selected_twist.linear.x==0
    for _ in range(3):valid_cycle(n,t,.2)
    assert n._mode==module.DIRECT and len(n._nav2_action.goals)==1
    valid_cycle(n,t,.1)
    assert n._last_selected_source=='direct'


@pytest.mark.parametrize('side_sign', [-1, 1])
def test_side_approach_heading_follows_endpoint_on_either_side(side_sign):
    grid=_grid(width=160,height=140,origin=(-2.,-3.,0.))
    _set_world_cell(grid,.8,0.,100)
    if side_sign < 0:
        _rasterize_box_bounds(grid,(-1.,2.,.1,2.5))
    target=(3.,0.)
    radial=stand_off_goal(0.,0.,*target)
    goal=clear_stand_off_goal(grid,radial,*target,0.,approach_from=(0.,0.))
    assert goal is not None and goal[1] * side_sign > 0
    assert math.hypot(goal[0]-target[0],goal[1]-target[1]) == pytest.approx(2.25)
    assert goal[2] == pytest.approx(math.atan2(goal[1],goal[0]))
    assert footprint_clear(grid,*goal)


def test_candidate_footprint_is_checked_at_sent_approach_heading():
    grid=_grid(width=160,height=140,origin=(-2.,-3.,0.))
    _set_world_cell(grid,.8,0.,100)
    radial=stand_off_goal(0.,0.,3.,0.)
    legacy=clear_stand_off_goal(grid,radial,3.,0.,0.)
    approach=clear_stand_off_goal(grid,radial,3.,0.,0.,approach_from=(0.,0.))
    # This cell is under the front of the approach footprint, above the old one.
    _set_world_cell(grid,1.14,1.32,100)
    assert footprint_clear(grid,*legacy)
    assert not footprint_clear(grid,*approach)
    goal=clear_stand_off_goal(grid,radial,3.,0.,0.,approach_from=(0.,0.))
    assert goal is not None and goal[1]<0 and footprint_clear(grid,*goal)


@pytest.mark.parametrize('forwarded',[False,True])
def test_abort_never_retries_a_second_side(episode,forwarded):
    n,t,h=episode
    n._detour_forwarded=forwarded
    n._on_nav2_goal_result(NS(result=lambda:NS(status=GoalStatus.STATUS_ABORTED)))
    n._active_actions.clear()
    t.ros+=400_000_000;t.wall+=.4
    n._finish_goal_drain(t.ros,t.wall)
    n._integrated_cycle(NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)
    assert n._fault_latched and n._last_selected_twist.linear.x==0
    assert len(n._nav2_action.goals)==1


def test_no_side_endpoint_stops_without_radial_fallback(approach):
    n,t,target=approach
    n._goal_state,n._goal_unresolved='idle',False
    n._nav2_goal_offset=n._nav2_goal_target_map=None
    n._global_grid.data[:]=[100]*len(n._global_grid.data)
    n._start_nav2_goal()
    assert n._fault_latched and n._last_selected_twist.linear.x==0
    assert len(n._nav2_action.goals)==1


def test_zero_requested_speed_stops_and_clears_pending_refresh(episode):
    n,t,h=episode
    n._goal_refresh_pending=True
    n._integrated_cycle(NS(linear_x=0.,angular_z=0.),n._test_target,t.ros,t.wall)
    assert n._last_selected_twist.linear.x==0 and h.cancels==1
    assert not n._goal_refresh_pending
    assert n._goal_cancel_reason=='target_distance_reached'


@pytest.mark.parametrize('y',[-.0001,0.,.0001])
def test_radial_goal_stays_straight_under_tiny_bearing_noise(y):
    grid=_grid(width=160,height=140,origin=(-2.,-3.,0.))
    radial=stand_off_goal(0.,0.,2.84,y)
    goal=clear_stand_off_goal(grid,radial,2.84,y,0.)
    assert goal is not None
    assert goal==pytest.approx(radial)
    assert math.hypot(goal[0]-2.84,goal[1]-y) == pytest.approx(2.25)


def test_clear_path_cancels_nav2_and_waits_for_both_clock_drains(episode):
    from geometry_msgs.msg import Twist
    n,t,h=episode
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._latest_scan.ranges=[2.8]*len(n._latest_scan.ranges)
    command=NS(linear_x=.05,angular_z=0.)
    nav=Twist();nav.linear.x=.04
    for _ in range(7):
        t.ros+=100_000_000;t.wall+=.1
        n._local_grid.header.stamp=Time(nanoseconds=t.ros).to_msg()
        n._latest_target_point.header.stamp=Time(nanoseconds=t.ros).to_msg()
        n._latest_target_received_monotonic=t.wall
        n._on_nav2_command(nav)
        n._integrated_cycle(command,n._test_target,t.ros,t.wall)
        assert n._goal_handle is h and n._goal_unresolved
        if h.cancels:
            assert h.cancels==1 and n._last_selected_twist.linear.x==0
        else:
            assert n._last_selected_source=='nav2'
    assert n._goal_cancel_reason=='direct_path_clear'
    n._on_nav2_goal_result(NS(result=lambda:NS(status=GoalStatus.STATUS_CANCELED)))
    n._active_actions.clear()
    assert n._last_selected_twist.linear.x==0
    n._finish_goal_drain(t.ros,t.wall+.4)
    assert n._goal_unresolved and n._last_selected_twist.linear.x==0
    t.ros+=400_000_000;t.wall+=.4
    n._finish_goal_drain(t.ros,t.wall)
    assert not n._goal_unresolved
    n._local_grid.header.stamp=Time(nanoseconds=t.ros).to_msg()
    fresh(n,t)
    n._integrated_cycle(command,n._test_target,t.ros,t.wall)
    assert n._mode==module.DIRECT, n._mode_reason
    assert n._last_selected_twist.linear.x==0
    valid_cycle(n,t,.1)
    assert n._last_selected_source=='direct' and n._last_selected_twist.linear.x==pytest.approx(.05)


def test_clear_path_discards_pending_refresh_instead_of_starting_another_goal(episode):
    from geometry_msgs.msg import Twist
    n,t,h=episode
    n._local_grid.data[:]=[0]*len(n._local_grid.data)
    n._latest_scan.ranges=[2.8]*len(n._latest_scan.ranges)
    n._goal_refresh_pending=True
    n._request_goal_cancel('target_moved','refresh')
    n._on_nav2_goal_result(NS(result=lambda:NS(status=GoalStatus.STATUS_CANCELED)))
    n._active_actions.clear()
    t.ros+=400_000_000;t.wall+=.4
    n._finish_goal_drain(t.ros,t.wall)
    requested=[]
    n._start_nav2_goal=lambda:requested.append(True)
    n._local_grid.header.stamp=Time(nanoseconds=t.ros).to_msg()
    n._integrated_cycle(NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)
    assert requested==[] and n._mode==module.NAV2_PENDING
    assert n._last_selected_twist.linear.x==0
    t.ros+=600_000_000;t.wall+=.6
    fresh(n,t)
    n._integrated_cycle(NS(linear_x=.05,angular_z=0.),n._test_target,t.ros,t.wall)
    assert requested==[] and n._mode==module.DIRECT
    assert n._last_blocked_stamp_ns is None and not n._goal_refresh_pending
    assert n._last_selected_twist.linear.x==0
