"""One source-time candidate waits for delayed VO without extending its lifetime."""
from collections import defaultdict
import json
from types import SimpleNamespace as NS

import pytest
from geometry_msgs.msg import PointStamped
from rclpy.clock import ClockType
from rclpy.time import Time
from tf2_ros import ExtrapolationException

from piracer_state_manager import follow_node as module
from piracer_state_manager.follow_control import FollowPController
from test_follow_last_seen import episode


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class DelayedTF:
    available_ns = 0

    def transform(self, message, frame):
        stamp = module.PiracerFollowNode._stamp_ns(message)
        if frame == "map" and stamp > self.available_ns:
            raise ExtrapolationException("VO has not processed this image")
        return NS(header=message.header, point=message.point)


@pytest.fixture
def direct(episode):
    n, t, _ = episode
    n._mode, n._goal_state, n._goal_unresolved = module.DIRECT, "idle", False
    n._goal_handle = n._memory_goal = n._last_seen = n._observation_stage = None
    n._active_actions.clear()
    n._last_blocked_stamp_ns = None
    n._latest_target_point = n._latest_target_received_monotonic = None
    n._pending_target_point = n._target_input_stamp_ns = None
    n._local_grid.data[:] = [0] * len(n._local_grid.data)
    n._global_grid.data[:] = [0] * len(n._global_grid.data)
    n._controller = FollowPController(max_speed_mps=.075)
    n._cycle_counts = defaultdict(int)
    n._record_cycle = lambda *_: None
    n._last_ros_time_ns = n._last_ros_time_advance_monotonic = None
    n.get_clock = lambda: NS(now=lambda: Time(nanoseconds=t.ros),
                             clock_type=ClockType.SYSTEM_TIME)
    original = n.get_parameter
    n.get_parameter = lambda key: NS(value="base_link") if key == "base_frame" else original(key)
    n._control_sample_publisher = Publisher()
    n._command_publisher.messages.clear()
    tf = DelayedTF()
    n._tf_buffer.transform = tf.transform
    return n, t, tf


def point(t, x=3., y=.1):
    result = PointStamped()
    result.header.frame_id = "base_link"
    result.header.stamp = Time(nanoseconds=t.ros).to_msg()
    result.point.x, result.point.y = x, y
    return result


def advance(t, seconds):
    t.ros += round(seconds*1e9)
    t.wall += seconds


@pytest.mark.parametrize("lag", [.1, .3])
@pytest.mark.parametrize("controller", ["P", "PD", "PI"])
def test_fast_point_stream_does_not_starve_delayed_source_tf(direct, lag, controller):
    n, t, tf = direct
    n._controller = FollowPController(max_speed_mps=.075,
        direction_kd=.5 if controller == "PD" else 0.,
        direction_ki=.01 if controller == "PI" else 0.,
        distance_ki=.005 if controller == "PI" else 0.)
    receipts, promoted = {}, {}
    for i in range(64):
        advance(t, .025)
        tf.available_ns = t.ros-round(lag*1e9)
        incoming = point(t, y=.1+i*.001)
        receipts[t.ros] = t.wall
        n._on_target_point(incoming)
        n._control_cycle()
        if n._latest_target_point is not None:
            stamp = n._stamp_ns(n._latest_target_point)
            promoted[stamp] = n._latest_target_received_monotonic
            assert promoted[stamp] == receipts[stamp]
            assert stamp <= tf.available_ns
        else:
            assert n._last_selected_twist.linear.x == 0
    assert len(promoted) >= 3
    assert any(m.linear.x > 0 for m in n._command_publisher.messages)
    previous = None
    for sample in n._control_sample_publisher.messages:
        value = json.loads(sample.data)
        stamp = value["target_stamp_ns"]
        assert stamp in receipts and value["sim_time_ns"]-stamp <= 500_000_000
        if value["new_measurement"] and value["measurement_dt_s"] is not None and previous is not None:
            assert value["measurement_dt_s"] == pytest.approx((stamp-previous)/1e9)
        if value["new_measurement"]:
            previous = stamp


@pytest.mark.parametrize("clock", ["ros", "wall"])
def test_expired_point_cannot_be_renewed_by_replay_or_late_tf(direct, clock):
    n, t, tf = direct
    incoming = point(t)
    n._on_target_point(incoming)
    assert n._pending_target_point[0] is incoming
    if clock == "ros":
        t.ros += 600_000_000
    else:
        t.wall += .6  # ROS clock is paused.
    n._control_cycle()
    assert n._pending_target_point is None and n._latest_target_point is None
    tf.available_ns = t.ros
    n._on_target_point(incoming)
    n._control_cycle()
    assert n._latest_target_point is None
    assert all(m.linear.x == 0 for m in n._command_publisher.messages)


@pytest.mark.parametrize("fault", ["unsafe", "disable", "rewind", "lost", "latched"])
def test_wait_cannot_resurrect_a_point_after_stop_or_invalidating_fault(direct, fault):
    n, t, tf = direct
    incoming = point(t)
    n._on_target_point(incoming)
    n._control_cycle()
    if fault == "unsafe":
        advance(t, .1)
        n._on_target_point(point(t, x=1.5))
    elif fault == "disable":
        n._set_enabled(NS(data=False), NS())
        n._set_enabled(NS(data=True), NS())
    elif fault == "rewind":
        t.ros -= 1_000_000_000
    elif fault == "lost":
        n._on_odom_info(NS(lost=True))
    else:
        n._fault_latched = True
    tf.available_ns = t.ros
    n._control_cycle()
    assert n._pending_target_point is None and n._latest_target_point is None
    assert n._last_selected_twist.linear.x == 0


def test_fresh_valid_cache_continues_but_expires_while_new_point_waits(direct):
    n, t, tf = direct
    tf.available_ns = t.ros
    first = point(t)
    receipt = t.wall
    n._on_target_point(first)
    n._control_cycle()
    assert n._last_selected_twist.linear.x > 0
    advance(t, .1)
    second = point(t)
    n._on_target_point(second)
    n._control_cycle()
    assert n._pending_target_point[0] is second
    assert n._latest_target_point is first and n._latest_target_received_monotonic == receipt
    assert n._last_selected_twist.linear.x > 0
    advance(t, .45)
    n._control_cycle()
    assert n._last_selected_twist.linear.x == 0
    tf.available_ns = t.ros
    n._control_cycle()
    assert n._latest_target_point is second
    assert n._latest_target_received_monotonic == pytest.approx(receipt+.1)
