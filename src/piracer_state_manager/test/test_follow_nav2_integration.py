from math import cos, sin
from pathlib import Path
from types import SimpleNamespace

import math

import pytest

from piracer_state_manager.follow_nav2_helpers import (
    DIRECT,
    NAV2,
    NAV2_PENDING,
    SAFE_STOP,
    bound_nav_twist,
    clear_stand_off_goal,
    footprint_clear,
    goal_refresh_due,
    localization_is_fresh,
    nav2_command_policy,
    require_distinct_command_topics,
    stand_off_goal,
    swept_footprint_clear,
    stamped_input_is_fresh,
)


def _grid(width=60, height=80, resolution=0.05, origin=(-1.0, -2.0, 0.0), fill=0):
    yaw = origin[2]
    return SimpleNamespace(
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=origin[0], y=origin[1]),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=sin(yaw / 2), w=cos(yaw / 2)
                ),
            ),
        ),
        data=[fill] * (width * height),
    )


def _set_world_cell(grid, x, y, value):
    ox = grid.info.origin.position.x
    oy = grid.info.origin.position.y
    q = grid.info.origin.orientation
    yaw = 2.0 * __import__("math").atan2(q.z, q.w)
    dx, dy = x - ox, y - oy
    lx = cos(yaw) * dx + sin(yaw) * dy
    ly = -sin(yaw) * dx + cos(yaw) * dy
    i = int(lx / grid.info.resolution)
    j = int(ly / grid.info.resolution)
    grid.data[j * grid.info.width + i] = value


def _rasterize_box_bounds(grid, bounds, value=100):
    """Mark every cell with positive-area overlap with world-axis box bounds."""
    xmin, xmax, ymin, ymax = bounds
    ox = grid.info.origin.position.x
    oy = grid.info.origin.position.y
    resolution = grid.info.resolution
    for j in range(grid.info.height):
        cell_ymin = oy + j * resolution
        cell_ymax = cell_ymin + resolution
        if max(cell_ymin, ymin) >= min(cell_ymax, ymax):
            continue
        for i in range(grid.info.width):
            cell_xmin = ox + i * resolution
            cell_xmax = cell_xmin + resolution
            if max(cell_xmin, xmin) < min(cell_xmax, xmax):
                grid.data[j * grid.info.width + i] = value


def test_fixed_obstacle_blocks_swept_corridor_but_goal_and_start_are_free():
    grid = _grid()
    # Rasterize the complete physical box bounds at the actual 5 cm cell size.
    _rasterize_box_bounds(grid, (0.39, 0.47, 0.09, 0.21))

    assert footprint_clear(grid, 0.0, 0.0, 0.0)
    clear, collision_distance = swept_footprint_clear(
        grid, 0.0, 0.0, 0.0, 0.05, 0.0, 0.6
    )
    assert not clear
    assert 0.10 < collision_distance < 0.25
    # Initial observed RGB-D target point, not the nominal model spawn center.
    goal = stand_off_goal(0.0, 0.0, 2.839554, 0.013049, 2.25)
    assert goal == pytest.approx((0.589584, 0.0027, 0.0046), abs=2e-4)
    assert footprint_clear(grid, goal[0], goal[1], goal[2])


def test_same_fixture_box_off_corridor_is_clear_at_actual_target_goal():
    grid = _grid()
    _rasterize_box_bounds(grid, (0.39, 0.47, 0.29, 0.41))
    clear, _ = swept_footprint_clear(grid, 0.0, 0.0, 0.0, 0.05, 0.0, 0.6)
    assert clear
    goal = stand_off_goal(0.0, 0.0, 2.839554, 0.013049, 2.25)
    assert footprint_clear(grid, goal[0], goal[1], goal[2])


def test_barrier_rejects_radial_goal_but_allows_side_observation_pose():
    grid = _grid(width=150, height=120, origin=(-1.0, -3.0, 0.0))
    target = (4.116, -0.149)
    radial = stand_off_goal(0.838, -0.007, *target)
    assert clear_stand_off_goal(grid, radial, *target) == radial
    _rasterize_box_bounds(grid, (1.7, 2.0, -0.9, 0.9))
    assert not footprint_clear(grid, *radial)
    first = clear_stand_off_goal(grid, radial, *target)
    assert first is not None and footprint_clear(grid, *first)
    _set_world_cell(grid, first[0], first[1], 100)
    other = clear_stand_off_goal(grid, radial, *target)
    assert other is not None and footprint_clear(grid, *other)
    assert (first[1] - target[1]) * (other[1] - target[1]) < 0
    for x, y, yaw in (first, other):
        assert math.hypot(target[0] - x, target[1] - y) == pytest.approx(2.25)
        assert math.atan2(target[1] - y, target[0] - x) == pytest.approx(yaw)
    grid.data[:] = [-1] * len(grid.data)
    assert clear_stand_off_goal(grid, radial, *target) == radial
    grid.data[:] = [100] * len(grid.data)
    assert clear_stand_off_goal(grid, radial, *target) is None


def test_obstacle_outside_corridor_and_rotated_grid_are_handled():
    grid = _grid(origin=(-1.0, -1.0, 0.4))
    _set_world_cell(grid, 0.43, 0.35, 100)
    clear, _ = swept_footprint_clear(grid, 0.0, 0.0, 0.0, 0.05, 0.0, 0.6)
    assert clear
    _set_world_cell(grid, 0.43, 0.10, 100)
    clear, _ = swept_footprint_clear(grid, 0.0, 0.0, 0.0, 0.05, 0.0, 0.6)
    assert not clear


def test_unknown_is_traversable_but_out_of_map_and_stale_inputs_still_stop():
    grid = _grid(fill=-1)
    assert footprint_clear(grid, 0.0, 0.0, 0.0)
    assert swept_footprint_clear(grid, 0.0, 0.0, 0.0, 0.05, 0.0, 0.6)[0]
    grid = _grid(origin=(20.0, 20.0, 0.0))
    assert not footprint_clear(grid, 0.0, 0.0, 0.0)
    assert stamped_input_is_fresh(1_000_000_000, 1_400_000_000, 0.3, 0.6)
    assert not stamped_input_is_fresh(1_000_000_000, 1_700_000_000, 0.3, 0.6)
    assert not stamped_input_is_fresh(2_000_000_000, 1_000_000_000, 0.1, 0.6)
    assert stamped_input_is_fresh(1_000_000_000, 1_300_000_000, 0.0, 1.0)
    assert not stamped_input_is_fresh(1_000_000_000, 2_100_000_000, 0.0, 1.0)
    assert localization_is_fresh(False, 1_000_000_000, 1_500_000_000, 0.4)
    assert not localization_is_fresh(True, 1_000_000_000, 1_100_000_000, 0.1)
    assert not localization_is_fresh(False, 1_000_000_000, 2_100_000_000, 0.1)
    # A fresh costmap does not mask a stopped scan converter.
    assert stamped_input_is_fresh(1_000_000_000, 1_100_000_000, 0.1, 0.6)
    assert not stamped_input_is_fresh(1_000_000_000, 1_100_000_000, 0.7, 0.5)


@pytest.mark.parametrize("blocked_cost", [99, 100])
@pytest.mark.parametrize("cell", [-2, -1, 100])
def test_unknown_policy_preserves_invalid_and_occupied_cost_rejection(blocked_cost, cell):
    grid = _grid(fill=cell)
    assert footprint_clear(grid, 0.0, 0.0, 0.0, blocked_cost=blocked_cost) is (cell == -1)
    grid.data[:] = [blocked_cost] * len(grid.data)
    assert not footprint_clear(grid, 0.0, 0.0, 0.0, blocked_cost=blocked_cost)


def test_suitcase_at_standoff_does_not_create_an_avoidance_goal():
    goal = stand_off_goal(0.0, 0.0, 3.0, 0.0, 2.25)
    assert goal == pytest.approx((0.75, 0.0, 0.0))
    assert stand_off_goal(0.75, 0.0, 3.0, 0.0, 2.25) is None
    assert swept_footprint_clear(_grid(), 0.75, 0.0, 0.0, 0.0, 0.0, 0.0)[0]


def test_new_scan_danger_stops_direct_even_with_the_same_map_stamp():
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    command = SimpleNamespace(linear_x=.05, angular_z=0.)
    node._integrated_cycle(command, node._test_target, node._test_now_ns, node._test_monotonic_now)
    assert node._last_selected_source == "direct"
    node._latest_scan.ranges[len(node._latest_scan.ranges)//2] = .5
    node._integrated_cycle(command, node._test_target, node._test_now_ns, node._test_monotonic_now)
    assert node._last_selected_source == "stop"
    assert node._last_selected_twist.linear.x == 0
    assert node._mode == NAV2_PENDING


@pytest.mark.parametrize('mode',[DIRECT,NAV2])
def test_rotated_robot_cannot_release_nav2_while_corridor_to_suitcase_is_blocked(mode,monkeypatch):
    from geometry_msgs.msg import Twist
    from piracer_state_manager.follow_control import FollowPController
    from piracer_state_manager import follow_node as control_module
    from rclpy.time import Time
    from test_follow_last_seen import Handle

    # Turned pose observed at the premature handoff; target is beyond the barrier.
    pose=(.193,.027,.424)
    local=_stamp_grid(_grid(width=40,height=40,origin=(-.8,-1.,0.)),5_000_000_000)
    global_map=_stamp_grid(_grid(width=180,height=140,origin=(-2.,-3.,0.)),5_000_000_000,'map')
    _rasterize_box_bounds(global_map,(1.7,2.,-.6,.6))
    node=_integrated_cycle_node(mode=mode,local_grid=local,global_grid=global_map)
    clock=SimpleNamespace(ros=5_000_000_000,wall=100.)
    monkeypatch.setattr(control_module,'time',SimpleNamespace(monotonic=lambda:clock.wall))
    node.get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=clock.ros))
    node._controller=FollowPController(max_speed_mps=.075)
    node._tf_pose=lambda *_:pose
    target_map=SimpleNamespace(header=node._latest_target_point.header,
                               point=SimpleNamespace(x=5.,y=0.,z=0.))
    node._tf_buffer.transform=lambda *_:target_map
    dx,dy=5.-pose[0],-pose[1]
    base=SimpleNamespace(point=SimpleNamespace(
        x=cos(pose[2])*dx+sin(pose[2])*dy,
        y=-sin(pose[2])*dx+cos(pose[2])*dy,z=0.))
    command=node._controller.update(base.point.x,base.point.y)
    assert swept_footprint_clear(local,*pose,command.linear_x,command.angular_z,.6)[0]
    from piracer_state_manager.follow_nav2_helpers import scan_clears_motion
    assert scan_clears_motion(node._latest_scan,(0.,0.,0.),0.,distance=2.)
    handle=Handle(7)
    if mode==NAV2:
        node._goal_handle=handle
        node._goal_state,node._goal_unresolved='active',True
        node._sent_goal_stamp_ns=5_000_000_000
        node._nav2_goal_target_map=(5.,0.)
        node._nav2_goal_started_wall=100.
        node._active_actions={'navigate_to_pose':{bytes(handle.goal_id.uuid):(2,5_000_000_000)},
                              'follow_path':{b'current-path':(2,5_000_000_000)}}
    for step in range(4):
        now=5_000_000_000+step*200_000_000
        clock.ros,clock.wall=now,100.+step*.2
        node._latest_target_point.header.stamp.sec=now//10**9
        node._latest_target_point.header.stamp.nanosec=now%10**9
        if mode==NAV2:
            node._test_monotonic_now=100.+step*.2
            nav=Twist();nav.linear.x=.04
            node._on_nav2_command(nav)
        node._integrated_cycle(command,base,now,100.+step*.2)
        assert node._direct_clear_since_ns is None
        assert node._last_selected_source!='direct'
    if mode==NAV2:
        assert handle.cancels==0 and node._goal_state=='active'
    else:
        assert node._mode==NAV2_PENDING and len(node._nav2_action.goals)==1


def test_nav2_forwarding_caps_speed_curvature_and_rejects_unsafe_twists():
    result = bound_nav_twist(0.10, 0.20)
    assert result is not None
    speed, yaw_rate = result
    assert speed == pytest.approx(0.05)
    assert yaw_rate / speed == pytest.approx(0.20 / 0.10)
    assert nav2_command_policy(0.10, 0.20, 0.05, True) == ("forward", result)
    assert nav2_command_policy(-0.01, 0.0, 0.01, True)[0] == "fault"
    assert nav2_command_policy(0.0, 0.2, 0.01, True)[0] == "fault"
    assert nav2_command_policy(float("nan"), 0.0, 0.01, False)[0] == "fault"
    assert nav2_command_policy(0.10, 0.1, 1.0, False)[0] == "wait"
    assert nav2_command_policy(0.10, 0.1, 1.0, True)[0] == "fault"


@pytest.mark.parametrize("cap", [0.05, 0.075])
@pytest.mark.parametrize("turn", [-1, 1])
def test_selected_follow_speed_preserves_curvature_and_steering_limit(cap, turn):
    policy, (v, w) = nav2_command_policy(0.10, turn * 0.20, 0.05, True, cap)
    assert policy == "forward" and v == pytest.approx(cap)
    assert w / v == pytest.approx(turn * 2.0)
    _, (v, w) = nav2_command_policy(0.10, turn * 2.0, 0.05, True, cap)
    assert math.atan(0.155 * w / v) == pytest.approx(turn * 0.554)


def test_goal_refresh_is_rate_and_displacement_bounded():
    # The helper compares target positions, so robot-pose changes cannot request refresh.
    assert not goal_refresh_due(0.0, 0.0, 0.0, 0.0, 10.0)
    assert not goal_refresh_due(0.2, 0.0, 0.0, 0.0, 2.99)
    assert not goal_refresh_due(0.24, 0.0, 0.0, 0.0, 3.0)
    assert goal_refresh_due(0.26, 0.0, 0.0, 0.0, 3.0)


def test_expected_modes_are_distinct():
    assert DIRECT != NAV2 and NAV2 != SAFE_STOP and SAFE_STOP != DIRECT


def test_integrated_command_topic_collision_fails_closed():
    with pytest.raises(ValueError, match="command_topic to differ from nav2_command_topic"):
        require_distinct_command_topics("/cmd_vel", "/cmd_vel")
    require_distinct_command_topics("/cmd_vel_follow", "/cmd_vel")

    source = (Path(__file__).parents[1]
              / "piracer_state_manager/follow_node.py").read_text()
    assert "if self._follow_nav2_avoidance:" in source
    assert "self.resolve_topic_name(" in source
    assert "require_distinct_command_topics(output_topic, nav2_input_topic)" in source


def test_cancel_terminal_and_smoother_drain_require_wall_and_ros_progress():
    pytest.importorskip("rclpy")
    from action_msgs.msg import GoalStatus
    from piracer_state_manager.follow_node import PiracerFollowNode

    node = object.__new__(PiracerFollowNode)
    node._memory_goal = node._last_seen = None
    node._nav2_goal_offset = None
    node._publish_cmd_vel = False
    node._goal_state = "canceling"
    node._goal_unresolved = True
    node._goal_cancel_reason = "target_moved"
    node._goal_finish_action = None
    node._goal_terminal_wall = None
    node._goal_terminal_ros_ns = None
    node._goal_handle = object()
    node._goal_send_future = None
    node._enabled = True
    node._fault_latched = False
    node._mode = SAFE_STOP
    node._mode_reason = "waiting_for_cancel"
    node._goal_terminal_status = None
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_: None, error=lambda *_: None)

    class ClockNow:
        nanoseconds = 1_000_000_000

    class Clock:
        @staticmethod
        def now():
            return ClockNow()

    node.get_clock = lambda: Clock()

    class Future:
        @staticmethod
        def result():
            return SimpleNamespace(status=GoalStatus.STATUS_CANCELED)

    node._on_nav2_goal_result(Future())
    assert node._goal_state == "draining"
    assert node._goal_unresolved

    node._finish_goal_drain(1_000_000_000, node._goal_terminal_wall + 0.4)
    assert node._goal_state == "draining"  # wall time alone cannot release ownership
    node._finish_goal_drain(1_250_000_000, node._goal_terminal_wall + 0.4)
    assert node._goal_state == "draining"  # ROS time must cross smoother timeout too
    node._finish_goal_drain(1_300_000_001, node._goal_terminal_wall + 0.4)
    assert node._goal_state == "idle"
    assert not node._goal_unresolved
    assert node._mode == NAV2_PENDING


def test_late_goal_acceptance_is_canceled_and_unknown_acceptance_stays_owned():
    pytest.importorskip("rclpy")
    from piracer_state_manager.follow_node import PiracerFollowNode

    class ResultFuture:
        def add_done_callback(self, callback):
            self.callback = callback

    class Handle:
        accepted = True

        def __init__(self):
            self.cancel_count = 0
            self.goal_id = SimpleNamespace(uuid=[1] * 16)

        def get_result_async(self):
            return ResultFuture()

        def cancel_goal_async(self):
            self.cancel_count += 1

    handle = Handle()
    node = object.__new__(PiracerFollowNode)
    node._memory_goal = node._last_seen = None
    node._nav2_goal_offset = None
    node._goal_state = "sending"
    node._goal_unresolved = True
    node._goal_send_future = object()
    node._goal_handle = None
    node._goal_cancel_reason = "disabled"
    node._goal_finish_action = None
    node._active_actions = {}
    node._enabled = False
    node._mode = NAV2_PENDING
    node._mode_reason = "disabled_while_sending"
    node._fault_latched = False
    node.get_logger = lambda: SimpleNamespace(error=lambda *_: None)

    class Future:
        @staticmethod
        def result():
            return handle

    node._on_nav2_goal_response(Future())
    assert handle.cancel_count == 1
    assert node._goal_state == "canceling"
    assert node._goal_unresolved

    unknown = object.__new__(PiracerFollowNode)
    unknown._memory_goal = unknown._last_seen = None
    unknown._nav2_goal_offset = None
    unknown._goal_state = "sending"
    unknown._goal_unresolved = True
    unknown._mode = NAV2_PENDING
    unknown._fault_latched = False
    unknown.get_logger = lambda: SimpleNamespace(error=lambda *_: None)

    class FailedFuture:
        @staticmethod
        def result():
            raise RuntimeError("transport lost")

    unknown._on_nav2_goal_response(FailedFuture())
    assert unknown._goal_state == "unknown"
    assert unknown._goal_unresolved
    assert unknown._mode == SAFE_STOP


def test_follow_disable_ack_waits_for_goal_terminal_and_stale_nav_output_is_never_forwarded():
    pytest.importorskip("rclpy")
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import Twist
    from piracer_state_manager.follow_node import PiracerFollowNode

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Controller:
        def reset(self, _reason):
            pass

    class GoalHandle:
        def __init__(self):
            self.cancel_count = 0

        def cancel_goal_async(self):
            self.cancel_count += 1

    node = object.__new__(PiracerFollowNode)
    node._memory_goal = node._last_seen = None
    node._nav2_goal_offset = None
    node._follow_nav2_avoidance = True
    node._enabled = True
    node._goal_unresolved = True
    node._goal_state = "active"
    node._goal_handle = GoalHandle()
    node._goal_cancel_reason = None
    node._goal_finish_action = None
    node._mode = NAV2
    node._mode_reason = "goal_accepted"
    node._nav2_latest_command = None
    node._publish_cmd_vel = True
    node._command_publisher = Publisher()
    node._controller = Controller()
    node._fault_latched = False
    node._latest_target_point = None
    node._latest_target_received_monotonic = None
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_: None)

    response = node._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success
    assert node._goal_handle.cancel_count == 1
    assert node._command_publisher.messages[-1].linear.x == 0.0

    class ClockNow:
        nanoseconds = 2_000_000_000

    class Clock:
        @staticmethod
        def now():
            return ClockNow()

    node.get_clock = lambda: Clock()

    class CanceledFuture:
        @staticmethod
        def result():
            return SimpleNamespace(status=GoalStatus.STATUS_CANCELED)

    node._on_nav2_goal_result(CanceledFuture())
    terminal_wall = node._goal_terminal_wall
    node._finish_goal_drain(2_000_000_000, terminal_wall + 0.4)
    assert node._goal_unresolved
    node._finish_goal_drain(2_300_000_001, terminal_wall + 0.4)
    assert not node._goal_unresolved
    final_response = node._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
    assert final_response.success

    node._nav2_latest_command = None
    node._mode = DIRECT
    old_command = Twist()
    old_command.linear.x = 0.04
    node._on_nav2_command(old_command)
    assert node._nav2_latest_command is None

    assert nav2_command_policy(0.04, 0.0, 0.21, True)[0] == "fault"


def test_dry_follow_blocks_twists_and_navigate_to_pose_requests():
    pytest.importorskip("rclpy")
    from geometry_msgs.msg import Twist
    from piracer_state_manager.follow_node import PiracerFollowNode

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    node = object.__new__(PiracerFollowNode)
    node._memory_goal = node._last_seen = None
    node._nav2_goal_offset = None
    node._publish_cmd_vel = False
    node._command_publisher = Publisher()
    node._goal_unresolved = False
    node._goal_state = "idle"
    node._nav2_action = None
    node._mode = DIRECT
    node._mode_reason = ""
    node._publish_command(Twist())
    node._start_nav2_goal()
    assert node._command_publisher.messages == []
    assert node._goal_state == "idle"
    assert node._mode == NAV2_PENDING


def _integrated_cycle_node(mode=DIRECT, local_grid=None, global_grid=None):
    pytest.importorskip("rclpy")
    from builtin_interfaces.msg import Time as RosTime
    from geometry_msgs.msg import Twist
    from piracer_state_manager.follow_node import PiracerFollowNode

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Action:
        def __init__(self):
            self.goals = []

        def server_is_ready(self):
            return True

        def send_goal_async(self, goal):
            self.goals.append(goal)
            return SimpleNamespace(add_done_callback=lambda callback: None)

    now_ns = 5_000_000_000
    monotonic_now = 100.0
    target_map = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=5, nanosec=0), frame_id="map"
        ),
        point=SimpleNamespace(x=2.839554, y=0.013049, z=0.0),
    )
    target = SimpleNamespace(point=SimpleNamespace(x=2.839554, y=0.013049, z=0.0))
    node = object.__new__(PiracerFollowNode)
    node._pending_target_point = None
    node._target_input_stamp_ns = None
    node._memory_goal = node._last_seen = None
    node._nav2_goal_offset = None
    node._enabled = True
    node._publish_cmd_vel = True
    node._follow_nav2_avoidance = True
    node._fault_latched = False
    node._mode = mode
    node._mode_reason = "test"
    node._last_integrated_ros_ns = None
    node._last_blocked_stamp_ns = None
    node._direct_clear_since_ns = None
    node._local_grid = local_grid
    node._global_grid = global_grid
    node._integrated_inputs_fresh = lambda *_: (
        node._local_grid is not None and node._global_grid is not None,
        "missing_costmap" if node._local_grid is None or node._global_grid is None else "",
    )
    node._tf_pose = lambda *_: (0.0, 0.0, 0.0)
    node._tf_buffer = SimpleNamespace(transform=lambda *_: target_map)
    node._controller = SimpleNamespace(
        max_speed_mps=0.05,
        target_distance_m=2.25,
        min_distance_m=2.0,
        reset=lambda *_: None,
    )
    node._command_publisher = Publisher()
    node._nav2_goal_update_publisher = Publisher()
    node._goal_state = "idle"
    node._goal_unresolved = False
    node._goal_handle = None
    node._goal_send_future = None
    node._goal_cancel_reason = None
    node._goal_terminal_wall = None
    node._goal_terminal_ros_ns = None
    node._goal_terminal_status = None
    node._goal_finish_action = None
    node._nav2_goal_started_wall = None
    node._nav2_goal_target_map = None
    node._nav2_has_moved = False
    node._nav2_latest_command = None
    node._nav2_latest_command_wall = None
    node._goal_refresh_pending = False
    node._nav2_action = Action()
    from sensor_msgs.msg import LaserScan
    node._latest_scan = LaserScan()
    node._latest_scan.header.frame_id = "base_link"
    node._latest_scan.header.stamp.sec = 5
    node._latest_scan.range_min, node._latest_scan.range_max = .05, 3.
    node._latest_scan.angle_min, node._latest_scan.angle_increment = -.5, .01
    node._latest_scan.ranges = [float("inf")] * 101
    node._latest_scan_received_monotonic = monotonic_now
    node._odom_info = SimpleNamespace(lost=False, header=node._latest_scan.header)
    node._odom_info_received_monotonic = monotonic_now
    node._local_grid_received_monotonic = monotonic_now
    node._global_grid_received_monotonic = monotonic_now
    node._latest_detections = None
    node._detection_receipt = None
    node._active_actions = {}
    node._latest_target_point = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=5, nanosec=0)),
        point=target.point,
    )
    node.get_parameter = lambda name: SimpleNamespace(value={
        "max_target_age_s": 0.5,
        "target_distance_m": 2.25,
    }[name])

    class ClockNow:
        nanoseconds = now_ns

        @staticmethod
        def to_msg():
            return RosTime(sec=5, nanosec=0)

    class Clock:
        @staticmethod
        def now():
            return ClockNow()

    node.get_clock = lambda: Clock()
    node.get_logger = lambda: SimpleNamespace(
        warning=lambda *_: None,
        error=lambda *_: None,
    )
    node._test_now_ns = now_ns
    node._latest_target_received_monotonic = monotonic_now
    node._test_monotonic_now = monotonic_now
    node._test_target = target
    return node


def _stamp_grid(grid, stamp_ns, frame_id="vo_odom"):
    grid.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000,
                              nanosec=stamp_ns % 1_000_000_000),
        frame_id=frame_id,
    )
    return grid


@pytest.mark.parametrize("distance,expected_nav2", [(1.5, True), (2., True), (2.02, False)])
def test_first_fresh_depth_hit_triggers_nav2_from_front_footprint(distance, expected_nav2):
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(width=120), 5_000_000_000, "map"))
    node._test_target.point.x = 5.0
    node._tf_buffer.transform = lambda message, _frame: message
    node._latest_scan.ranges[50] = .246 + distance
    node._integrated_cycle(SimpleNamespace(linear_x=.05, angular_z=0.), node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert bool(node._nav2_action.goals) is expected_nav2
    if expected_nav2:
        assert node._mode == NAV2_PENDING
        assert node._last_selected_twist.linear.x == 0
        assert not any(m.linear.x > 0 for m in node._command_publisher.messages)
    else:
        assert node._last_selected_source == "direct"


def test_depth_obstacle_does_not_start_a_goal_when_follow_wants_to_stand_still():
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    node._latest_scan.ranges[50] = 1.5
    node._integrated_cycle(SimpleNamespace(linear_x=0., angular_z=0.), node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert not node._nav2_action.goals and node._last_selected_twist.linear.x == 0


@pytest.mark.parametrize("hit,blocked", [(2.2, False), (.26, True)])
def test_near_standoff_scan_only_blocks_the_remaining_footprint(hit, blocked):
    from piracer_state_manager.follow_control import FollowPController

    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    target = node._test_target.point
    target.x, target.y = 2.27324498, .01837878
    node._tf_buffer.transform = lambda message, _frame: message
    node._latest_scan.ranges[50] = hit
    command = FollowPController().update(target.x, target.y)
    node._integrated_cycle(command, node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert bool(node._nav2_action.goals) is blocked
    assert not node._fault_latched
    if blocked:
        assert node._last_selected_twist.linear.x == 0
    else:
        assert node._mode == DIRECT and node._last_selected_source == "direct"
        assert node._last_selected_twist.linear.x == pytest.approx(command.linear_x)


def test_near_standoff_repeated_cycles_and_moving_target_need_no_rearm():
    from piracer_state_manager.follow_control import FollowPController
    from rclpy.time import Time

    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    controller = FollowPController()
    node._tf_buffer.transform = lambda message, _frame: message
    node._latest_scan.ranges[50] = 2.2
    for index, distance in enumerate((2.2733, 2.2733, 2.2733, 2.25, 3.0)):
        now_ns = node._test_now_ns + index * 100_000_000
        node._test_target.point.x, node._test_target.point.y = distance, 0.
        node._latest_target_point.header.stamp = Time(nanoseconds=now_ns).to_msg()
        node._latest_target_received_monotonic = node._test_monotonic_now + index * .1
        node._integrated_cycle(controller.update(distance, 0.), node._test_target,
                               now_ns, node._latest_target_received_monotonic)
        assert not node._nav2_action.goals and not node._fault_latched
        if distance == 2.25:
            assert node._last_selected_twist.linear.x == 0
        else:
            assert node._last_selected_source == "direct"
            assert node._last_selected_twist.linear.x > 0


def test_near_standoff_costmap_collision_still_stops_before_nav2():
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    node._test_target.point.x, node._test_target.point.y = 2.2733, 0.
    node._tf_buffer.transform = lambda message, _frame: message
    _set_world_cell(node._local_grid, .26, 0., 100)
    node._integrated_cycle(SimpleNamespace(linear_x=.00233, angular_z=0.),
                           node._test_target, node._test_now_ns, node._test_monotonic_now)
    assert node._mode_reason == "imminent_swept_footprint_collision"
    assert node._last_selected_twist.linear.x == 0
    assert not node._nav2_action.goals


def test_integrated_cycle_recovers_from_startup_data_wait_and_preserves_direct_p_output():
    from piracer_state_manager.follow_nav2_helpers import DIRECT

    node = _integrated_cycle_node(mode=SAFE_STOP)
    node._enabled = False
    response = node._set_enabled(SimpleNamespace(data=True), SimpleNamespace())
    assert response.success
    node._stop_for_input_error("no_target")
    assert node._mode == SAFE_STOP
    assert not node._fault_latched

    command = SimpleNamespace(linear_x=0.031, angular_z=-0.017)
    node._integrated_cycle(command, node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert node._mode == SAFE_STOP
    assert not node._fault_latched

    node._local_grid = _stamp_grid(_grid(), node._test_now_ns)
    node._global_grid = _stamp_grid(_grid(), node._test_now_ns, "map")
    node._integrated_cycle(command, node._test_target,
                           node._test_now_ns + 1, node._test_monotonic_now + 0.05)
    assert node._mode == DIRECT
    selected = node._command_publisher.messages[-1]
    assert selected.linear.x == pytest.approx(command.linear_x)
    assert selected.angular.z == pytest.approx(command.angular_z)
    assert node._last_selected_source == "direct"


def test_integrated_nav2_ignores_unused_direct_arc_but_stops_for_current_footprint():
    from geometry_msgs.msg import Twist
    from piracer_state_manager.follow_nav2_helpers import NAV2

    grid = _grid()
    for x in (0.40, 0.45):
        for y in (0.10, 0.15, 0.20):
            _set_world_cell(grid, x, y, 100)
    grid = _stamp_grid(grid, 5_000_000_000)
    node = _integrated_cycle_node(
        mode=NAV2,
        local_grid=grid,
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )
    node._goal_state = "active"
    node._goal_unresolved = True
    node._goal_handle = object()
    node._nav2_goal_target_map = (2.839554, 0.013049)
    node._goal_handle = SimpleNamespace(goal_id=SimpleNamespace(uuid=[1] * 16))
    node._sent_goal_stamp_ns = node._test_now_ns
    node._active_actions = {
        "navigate_to_pose": {bytes([1] * 16): (2, node._test_now_ns)},
        "follow_path": {b"own-path": (2, node._test_now_ns)},
    }
    node._nav2_goal_started_wall = 90.0
    nav_command = Twist()
    nav_command.linear.x = 0.05
    nav_command.angular.z = 0.20
    node._nav2_latest_command = nav_command
    node._nav2_latest_command_wall = node._test_monotonic_now
    node._nav2_has_moved = True
    direct = SimpleNamespace(linear_x=0.05, angular_z=0.0)

    node._integrated_cycle(direct, node._test_target,
                           node._test_now_ns, node._test_monotonic_now)
    assert node._mode == NAV2
    assert node._last_selected_source == "nav2"
    assert node._command_publisher.messages[-1].linear.x > 0.0

    for x in (0.10, 0.25, 0.30):
        for y in (0.0, 0.05):
            _set_world_cell(grid, x, y, 100)
    node._nav2_latest_command.angular.z = 0.0
    node._goal_handle = SimpleNamespace(cancel_goal_async=lambda: None)
    node._integrated_cycle(direct, node._test_target,
                           node._test_now_ns + 1, node._test_monotonic_now + 0.01)
    assert node._mode == SAFE_STOP
    assert node._last_selected_source == "stop"


def test_integrated_data_loss_cancels_nav2_then_recovers_after_clear_confirmation():
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import Twist
    node = _integrated_cycle_node(
        mode=NAV2,
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )

    class Handle:
        def __init__(self):
            self.cancel_count = 0

        def cancel_goal_async(self):
            self.cancel_count += 1

    handle = Handle()
    node._goal_state = "active"
    node._goal_unresolved = True
    node._goal_handle = handle
    node._nav2_latest_command = Twist()
    node._nav2_latest_command.linear.x = 0.05
    node._nav2_latest_command_wall = node._test_monotonic_now
    node._nav2_has_moved = True
    node._integrated_inputs_fresh = lambda *_: (False, "stale_scan")
    node._integrated_cycle(SimpleNamespace(linear_x=0.05, angular_z=0.0),
                           node._test_target, node._test_now_ns,
                           node._test_monotonic_now)
    assert node._mode == SAFE_STOP
    assert not node._fault_latched
    assert handle.cancel_count == 1
    assert node._command_publisher.messages[-1].linear.x == 0.0

    class CanceledFuture:
        @staticmethod
        def result():
            return SimpleNamespace(status=GoalStatus.STATUS_CANCELED)

    node._goal_cancel_reason = "stale_scan"
    node._on_nav2_goal_result(CanceledFuture())
    assert node._goal_state == "draining"
    assert node._goal_unresolved
    terminal_wall = node._goal_terminal_wall
    node._finish_goal_drain(node._test_now_ns + 400_000_000, terminal_wall + 0.4)
    assert not node._goal_unresolved
    assert node._mode == NAV2_PENDING
    assert not node._fault_latched

    node._integrated_inputs_fresh = lambda *_: (True, "")
    command = SimpleNamespace(linear_x=0.02, angular_z=0.01)
    for offset in range(5):
        stamp = node._test_now_ns + 500_000_000 + offset * 200_000_000
        node._latest_target_point.header.stamp.sec=stamp//10**9
        node._latest_target_point.header.stamp.nanosec=stamp%10**9
        node._local_grid = _stamp_grid(_grid(), stamp)
        node._integrated_cycle(command, node._test_target, stamp,
                               node._test_monotonic_now + 0.5 + offset * 0.05)
    assert node._mode == DIRECT
    assert node._last_selected_source == "direct"
    assert node._command_publisher.messages[-1].linear.x == pytest.approx(command.linear_x)


@pytest.mark.parametrize("goal_state", ("active", "canceling"))
def test_safe_stop_with_unresolved_active_goal_stays_stopped_without_fallthrough(goal_state):
    from piracer_state_manager.follow_nav2_helpers import SAFE_STOP

    node = _integrated_cycle_node(
        mode=SAFE_STOP,
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )
    node._goal_state = goal_state
    node._goal_unresolved = True
    node._goal_handle = object()
    node._mode_reason = "nav2_imminent_footprint_collision"
    original_reason = node._mode_reason
    node._tf_pose = lambda *_: pytest.fail("owned SAFE_STOP must return before motion evaluation")

    node._integrated_cycle(
        SimpleNamespace(linear_x=0.05, angular_z=0.0), node._test_target,
        node._test_now_ns, node._test_monotonic_now,
    )

    assert node._mode == SAFE_STOP
    assert node._mode_reason == original_reason
    assert node._goal_state == goal_state
    assert node._goal_unresolved
    assert not node._nav2_action.goals
    assert node._last_selected_source == "stop"
    assert node._command_publisher.messages[-1].linear.x == 0.0


def test_safe_stop_during_terminal_drain_preserves_drain_reason_and_ownership():
    from piracer_state_manager.follow_nav2_helpers import SAFE_STOP

    node = _integrated_cycle_node(
        mode=SAFE_STOP,
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )
    node._goal_state = "draining"
    node._goal_unresolved = True
    node._goal_terminal_status = "CANCELED"
    node._goal_terminal_wall = node._test_monotonic_now - 0.1
    node._goal_terminal_ros_ns = node._test_now_ns - 100_000_000
    node._goal_finish_action = "recover"
    node._mode_reason = "before_drain_tick"
    node._tf_pose = lambda *_: pytest.fail("terminal drain must return before motion evaluation")

    node._integrated_cycle(
        SimpleNamespace(linear_x=0.05, angular_z=0.0), node._test_target,
        node._test_now_ns, node._test_monotonic_now,
    )

    assert node._mode == SAFE_STOP
    assert node._mode_reason == "goal_terminal_canceled_tail_drain"
    assert node._goal_state == "draining"
    assert node._goal_unresolved
    assert not node._nav2_action.goals
    assert node._last_selected_source == "stop"
    assert node._command_publisher.messages[-1].linear.x == 0.0


def test_truly_unknown_integrated_mode_keeps_defensive_stop():
    from piracer_state_manager.follow_nav2_helpers import SAFE_STOP

    node = _integrated_cycle_node(
        mode="CORRUPT_MODE",
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )

    node._integrated_cycle(
        SimpleNamespace(linear_x=0.02, angular_z=0.0), node._test_target,
        node._test_now_ns, node._test_monotonic_now,
    )

    assert node._mode == SAFE_STOP
    assert node._mode_reason == "unknown_follow_mode"
    assert node._last_selected_source == "stop"
    assert node._command_publisher.messages[-1].linear.x == 0.0
    assert not node._nav2_action.goals


def test_nav2_goal_status_must_be_terminal_to_release_ownership():
    pytest.importorskip("rclpy")
    from piracer_state_manager.follow_node import PiracerFollowNode

    class ClockNow:
        nanoseconds = 5_000_000_000

    class Clock:
        @staticmethod
        def now():
            return ClockNow()

    class Logger:
        def __init__(self):
            self.errors = []
            self.warnings = []

        def error(self, message):
            self.errors.append(message)

        def warning(self, message):
            self.warnings.append(message)

    class Publisher:
        def publish(self, _message):
            pass

    class Future:
        def __init__(self, status):
            self.status = status

        def result(self):
            return SimpleNamespace(status=self.status)

    for status in (0, 1, 2, 3, 999):
        node = object.__new__(PiracerFollowNode)
        node._memory_goal = node._last_seen = None
        node._nav2_goal_offset = None
        node._goal_state = "active"
        node._goal_unresolved = True
        node._goal_cancel_reason = "disabled"
        node._goal_finish_action = None
        node._goal_handle = object()
        node._goal_send_future = None
        node._enabled = False
        node._follow_nav2_avoidance = True
        node._publish_cmd_vel = True
        node._controller = SimpleNamespace(reset=lambda *_: None)
        node._command_publisher = Publisher()
        node._fault_latched = False
        node._mode = SAFE_STOP
        node._mode_reason = "disabled"
        node._nav2_latest_command = object()
        node._goal_terminal_status = None
        node._goal_terminal_wall = None
        node._goal_terminal_ros_ns = None
        node._logger = Logger()
        node.get_logger = lambda: node._logger
        node.get_clock = lambda: Clock()
        node._on_nav2_goal_result(Future(status))
        assert node._goal_state == "unknown", status
        assert node._goal_unresolved
        assert node._fault_latched
        assert node._goal_terminal_wall is None
        assert node._logger.errors
        disabled = node._set_enabled(SimpleNamespace(data=False), SimpleNamespace())
        assert not disabled.success
        assert node._goal_unresolved

    aborted = object.__new__(PiracerFollowNode)
    aborted._memory_goal = aborted._last_seen = None
    aborted._nav2_goal_offset = None
    aborted._publish_cmd_vel = False
    aborted._goal_state = "active"
    aborted._goal_unresolved = True
    aborted._goal_cancel_reason = "disabled"
    aborted._goal_finish_action = None
    aborted._goal_handle = object()
    aborted._goal_send_future = None
    aborted._enabled = False
    aborted._fault_latched = False
    aborted._mode = SAFE_STOP
    aborted._mode_reason = "disabled"
    aborted._nav2_latest_command = None
    aborted._goal_terminal_status = None
    aborted._goal_terminal_wall = None
    aborted._goal_terminal_ros_ns = None
    aborted._logger = Logger()
    aborted.get_logger = lambda: aborted._logger
    aborted.get_clock = lambda: Clock()
    aborted._on_nav2_goal_result(Future(6))
    assert aborted._goal_state == "draining"
    assert aborted._goal_unresolved
    assert aborted._fault_latched
    assert aborted._goal_terminal_status == "ABORTED"
    assert "status=ABORTED(6)" in aborted._logger.warnings[-1]
    assert aborted._mode_reason == "goal_terminal_aborted_tail_drain"

    malformed = object.__new__(PiracerFollowNode)
    malformed._memory_goal = malformed._last_seen = None
    malformed._nav2_goal_offset = None
    malformed._publish_cmd_vel = False
    malformed._goal_state = "active"
    malformed._goal_unresolved = True
    malformed._mode = SAFE_STOP
    malformed._fault_latched = False
    malformed._goal_terminal_wall = None
    malformed._goal_terminal_ros_ns = None
    malformed._goal_terminal_status = None
    malformed._logger = Logger()
    malformed.get_logger = lambda: malformed._logger
    malformed._on_nav2_goal_result(SimpleNamespace(result=lambda: object()))
    assert malformed._goal_state == "unknown"
    assert malformed._goal_unresolved
    assert malformed._fault_latched


def test_repeated_blocked_maps_keep_first_global_costmap_barrier_until_clear_return():
    from piracer_state_manager.follow_nav2_helpers import NAV2_PENDING

    grid = _grid()
    for x in (0.40, 0.45):
        for y in (0.10, 0.15, 0.20):
            _set_world_cell(grid, x, y, 100)
    node = _integrated_cycle_node(
        mode=DIRECT,
        local_grid=_stamp_grid(grid, 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 4_900_000_000, "map"),
    )
    command = SimpleNamespace(linear_x=0.05, angular_z=0.0)
    node._integrated_cycle(command, node._test_target,
                           5_000_000_000, node._test_monotonic_now)
    node._local_grid = _stamp_grid(grid, 5_200_000_000)
    node._integrated_cycle(command, node._test_target,
                           5_200_000_000, node._test_monotonic_now + 0.05)
    assert node._last_blocked_stamp_ns == 5_000_000_000
    assert node._mode == NAV2_PENDING
    assert not node._nav2_action.goals

    # Later local observations do not move the barrier. A global frame newer
    # than the first blockage can then authorize exactly one goal.
    node._global_grid = _stamp_grid(_grid(), 5_300_000_000, "map")
    node._local_grid = _stamp_grid(grid, 5_400_000_000)
    node._integrated_cycle(command, node._test_target,
                           5_400_000_000, node._test_monotonic_now + 0.10)
    assert node._last_blocked_stamp_ns == 5_000_000_000
    assert len(node._nav2_action.goals) == 1
    assert node._goal_state == "sending"


def test_imminent_collision_stays_stopped_until_the_obstacle_is_farther_away():
    grid = _grid()
    _set_world_cell(grid, 0.30, 0.0, 100)
    assert footprint_clear(grid, 0.0, 0.0, 0.0)
    clear, collision_distance = swept_footprint_clear(
        grid, 0.0, 0.0, 0.0, 0.05, 0.0, 0.6
    )
    assert not clear
    assert 0.0 < collision_distance <= 0.10

    node = _integrated_cycle_node(
        mode=DIRECT,
        local_grid=_stamp_grid(grid, 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )
    calls = []
    node._command_publisher.publish = lambda message: calls.append(
        ("command", float(message.linear.x), float(message.angular.z),
         node._last_selected_source)
    )
    def submit_goal(goal):
        calls.append(("goal", goal))
        node._nav2_action.goals.append(goal)
        return SimpleNamespace(add_done_callback=lambda callback: None)
    node._nav2_action.send_goal_async = submit_goal
    command = SimpleNamespace(linear_x=0.05, angular_z=0.0)

    node._integrated_cycle(command, node._test_target,
                           5_000_000_000, node._test_monotonic_now)
    assert node._mode == SAFE_STOP
    assert not node._nav2_action.goals
    assert node._last_selected_source == "stop"
    assert calls == [("command", 0.0, 0.0, "stop")]

    node._integrated_cycle(command, node._test_target,
                           5_000_000_001, node._test_monotonic_now + 0.01)
    assert node._mode == SAFE_STOP
    assert not node._nav2_action.goals
    assert node._last_selected_source == "stop"
    assert calls[-1] == ("command", 0.0, 0.0, "stop")
    assert all(call[0] != "goal" for call in calls)
    assert all(call[3] != "direct" for call in calls if call[0] == "command")

    stamp = 5_200_000_000
    node._local_grid = _stamp_grid(grid, stamp)
    node._global_grid = _stamp_grid(_grid(), stamp, "map")
    node._integrated_cycle(command, node._test_target,
                           stamp, node._test_monotonic_now + 0.02)
    assert node._mode == SAFE_STOP and not node._nav2_action.goals
    assert node._last_selected_twist.linear.x == 0

    farther = _grid()
    _set_world_cell(farther, 0.45, 0.0, 100)
    stamp = 5_400_000_000
    node._local_grid = _stamp_grid(farther, stamp)
    node._global_grid = _stamp_grid(_grid(), stamp, "map")
    node._integrated_cycle(command, node._test_target,
                           stamp, node._test_monotonic_now + 0.04)
    assert node._mode == NAV2_PENDING
    assert len(node._nav2_action.goals) == 1
    assert node._goal_state == "sending"
    assert calls[-2][0] == "command"
    assert calls[-2][1:3] == (0.0, 0.0)
    assert calls[-2][3] == "stop"
    assert calls[-1][0] == "goal"
    assert all(call[3] != "direct" for call in calls if call[0] == "command")


def test_occupied_current_footprint_still_stops_before_any_goal():
    grid = _grid()
    _set_world_cell(grid, 0.10, 0.0, 100)
    assert not footprint_clear(grid, 0.0, 0.0, 0.0)
    node = _integrated_cycle_node(
        mode=DIRECT,
        local_grid=_stamp_grid(grid, 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )

    node._integrated_cycle(SimpleNamespace(linear_x=0.05, angular_z=0.0),
                           node._test_target, 5_000_000_000,
                           node._test_monotonic_now)
    assert node._mode == SAFE_STOP
    assert node._last_selected_source == "stop"
    assert not node._nav2_action.goals
    assert node._command_publisher.messages[-1].linear.x == 0.0
    assert node._command_publisher.messages[-1].angular.z == 0.0


def test_missing_depth_updates_reach_existing_stale_scan_stop():
    from piracer_state_manager.follow_node import PiracerFollowNode
    node = _integrated_cycle_node(
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"))
    node.get_parameter = lambda name: SimpleNamespace(value={
        "local_costmap_max_age_s": .6, "global_costmap_max_age_s": 2.,
        "max_localization_age_s": 1.}[name])
    node._latest_scan_received_monotonic = 99.4
    node._integrated_inputs_fresh = lambda *args: PiracerFollowNode._integrated_inputs_fresh(node, *args)
    assert node._integrated_inputs_fresh(5_000_000_000, 100.) == (False, "stale_scan")
    node._integrated_cycle(SimpleNamespace(linear_x=.05, angular_z=0.),
                           node._test_target, 5_000_000_000, 100.)
    assert node._last_selected_source == "stop"
    assert node._last_selected_twist.linear.x == 0 and not node._nav2_action.goals


def test_nav2_respects_controller_minimum_distance_stop_and_cancels_owned_goal():
    from geometry_msgs.msg import Twist
    from piracer_state_manager.follow_nav2_helpers import NAV2

    node = _integrated_cycle_node(
        mode=NAV2,
        local_grid=_stamp_grid(_grid(), 5_000_000_000),
        global_grid=_stamp_grid(_grid(), 5_000_000_000, "map"),
    )

    class Handle:
        def __init__(self):
            self.cancel_count = 0

        def cancel_goal_async(self):
            self.cancel_count += 1

    handle = Handle()
    node._goal_state = "active"
    node._goal_unresolved = True
    node._goal_handle = handle
    nav = Twist()
    nav.linear.x = 0.05
    node._nav2_latest_command = nav
    node._nav2_latest_command_wall = node._test_monotonic_now
    node._nav2_has_moved = True
    near_target = SimpleNamespace(point=SimpleNamespace(x=1.9, y=0.0, z=0.0))

    node._integrated_cycle(SimpleNamespace(linear_x=0.0, angular_z=0.0),
                           near_target, node._test_now_ns, node._test_monotonic_now)
    assert node._mode == SAFE_STOP
    assert node._last_selected_source == "stop"
    assert node._command_publisher.messages[-1].linear.x == 0.0
    assert handle.cancel_count == 1
