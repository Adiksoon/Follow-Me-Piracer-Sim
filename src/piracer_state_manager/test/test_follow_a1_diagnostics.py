import json
import os
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import PointStamped
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import TransformException

from piracer_state_manager.follow_node import PiracerFollowNode
from piracer_state_manager.follow_control import FollowPController


class _PublisherSpy:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_startup_log_reports_constructed_controller_and_samples_its_process(monkeypatch, capfd):
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    # Read the constructed controller, even if a declared parameter differs.
    monkeypatch.setattr(PiracerFollowNode, "_make_controller",
                        lambda _: FollowPController(distance_kp=0.20))
    node = PiracerFollowNode()
    try:
        captured = capfd.readouterr()
        lines = [line for line in (captured.out + captured.err).splitlines()
                 if "FOLLOW_CONFIG " in line]
        assert len(lines) == 1
        config = json.loads(lines[0].split("FOLLOW_CONFIG ", 1)[1])
        assert config == {"process_id": os.getpid(), "node": "/piracer_follow",
                          "distance_kp": 0.20, "distance_ki": 0.0,
                          "distance_kd": 0.0, "distance_filter_tau_s": 0.3,
                          "distance_integral_limit_mps": 0.0,
                          "distance_kd": 0.0, "distance_filter_tau_s": 0.3,
                          "distance_integral_limit_mps": 0.0,
                          "direction_kp": 1.0, "direction_kd": 0.0,
                          "direction_ki": 0.0,
                          "direction_integral_limit_rad": 0.02,
                          "direction_filter_tau_s": 0.3,
                          "target_distance_m": 2.25, "max_speed_mps": 0.05,
                          "max_steering_rad": 0.554, "publish_cmd_vel": False}
        samples, commands = _PublisherSpy(), _PublisherSpy()
        node._control_sample_publisher = samples
        node._command_publisher = commands
        node._tf_buffer.transform = lambda message, *_: message
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        _fresh_point(node)
        node._control_cycle()
        assert len(samples.messages) == 1
        assert json.loads(samples.messages[0].data)["process_id"] == config["process_id"]
        sample = json.loads(samples.messages[0].data)
        assert sample["distance_integral_m_s"] == 0.0
        assert sample["direction_filtered_derivative_radps"] == 0.0
        assert commands.messages == []
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def _fresh_point(node):
    point = PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp = node.get_clock().now().to_msg()
    point.point.x = 3.0
    node._on_target_point(point)
    return point


def test_follow_cycle_counts_missing_stale_tf_and_valid_outcomes():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        node._control_cycle()
        assert node._cycle_counts["disabled"] == 1

        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        node._control_cycle()
        assert node._cycle_counts["no_target"] == 1

        point = _fresh_point(node)
        node._latest_target_received_monotonic = time.monotonic() - 1.0
        node._control_cycle()
        assert node._cycle_counts["stale_received_target"] == 1

        point = _fresh_point(node)
        current = node.get_clock().now()
        point.header.stamp = Time(
            nanoseconds=current.nanoseconds - 600_000_000,
            clock_type=current.clock_type,
        ).to_msg()
        node._on_target_point(point)
        node._control_cycle()
        assert node._cycle_counts["stale_ros_stamp"] == 1

        _fresh_point(node)
        node._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(
            TransformException("missing transform")
        )
        node._control_cycle()
        assert node._cycle_counts["tf_error"] == 1

        _fresh_point(node)
        node._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(
            ValueError("invalid point")
        )
        node._control_cycle()
        assert node._cycle_counts["invalid_target"] == 1

        _fresh_point(node)
        node._tf_buffer.transform = lambda message, *_: message
        node._control_cycle()
        assert node._cycle_counts["valid"] == 1
        assert node._cycle_counts["timer_calls"] == 7
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_counts_zero_ros_time_without_enabling_cmd_vel():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    command_spy = _PublisherSpy()

    class _ZeroClock:
        clock_type = node.get_clock().clock_type

        @staticmethod
        def now():
            return Time(nanoseconds=0, clock_type=_ZeroClock.clock_type)

    try:
        node._command_publisher = command_spy
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        _fresh_point(node)
        node.get_clock = lambda: _ZeroClock()
        node._control_cycle()
        assert node._cycle_counts["ros_time_zero"] == 1
        assert node._cycle_counts["ros_time_zero_cycles"] == 1
        assert command_spy.messages == []
        node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        node._control_cycle()
        assert node._cycle_counts["disabled"] == 1
        assert node._cycle_counts["ros_time_zero_cycles"] == 2
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_ros_time_stall_observation_is_separate_and_resets_on_progress_or_rewind():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        assert not node._observe_ros_time(10, 100.0, 0.5)
        assert not node._observe_ros_time(10, 100.49, 0.5)
        assert node._observe_ros_time(10, 100.5, 0.5)
        assert not node._observe_ros_time(11, 100.51, 0.5)
        assert not node._observe_ros_time(5, 100.52, 0.5)
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_cumulative_diagnostics_are_published_at_interval_not_each_cycle():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    try:
        node._control_diagnostics_publisher = spy
        start = node._last_diagnostics_monotonic
        node._cycle_counts["timer_calls"] = 2
        node._record_cycle("valid", False, 1_000_000_000, start + 0.5)
        assert spy.messages == []
        node._record_cycle("disabled", True, 1_000_000_000, start + 1.0)
        assert len(spy.messages) == 1
        assert isinstance(spy.messages[0], String)
        summary = json.loads(spy.messages[0].data)
        assert summary["sim_time_ns"] == 1_000_000_000
        assert summary["cycle_counts"]["valid"] == 1
        assert summary["process_id"] == os.getpid()
        assert summary["cycle_counts"]["disabled"] == 1
        assert summary["cycle_counts"]["ros_time_stalled_cycles"] == 1
        assert summary["unclassified_cycles"] == 0
        assert summary["input_received"] == {
            "local_costmap": False, "global_costmap": False,
            "odom_info": False, "scan": False,
        }
        node._latest_scan = object()
        node._cycle_counts["timer_calls"] += 1
        node._record_cycle("disabled", False, 2_000_000_000, start + 2.0)
        assert json.loads(spy.messages[-1].data)["input_received"] == {
            "local_costmap": False, "global_costmap": False,
            "odom_info": False, "scan": True,
        }
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_active_extension_uses_original_target_stamp_and_stop_path_resets_history(monkeypatch):
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    monkeypatch.setattr(
        PiracerFollowNode,
        "_make_controller",
        lambda _: FollowPController(distance_ki=0.01, direction_kd=0.5),
    )
    node = PiracerFollowNode()
    samples = _PublisherSpy()
    commands = _PublisherSpy()
    try:
        node._control_sample_publisher = samples
        node._command_publisher = commands
        node._tf_buffer.transform = lambda message, *_: message
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        point = _fresh_point(node)
        source_stamp_ns = point.header.stamp.sec * 1_000_000_000 + point.header.stamp.nanosec
        node._control_cycle()
        assert node._controller._last_measurement_stamp_ns == source_stamp_ns
        assert json.loads(samples.messages[-1].data)["new_measurement"] is True

        node._control_cycle()  # The same received point is not a new measurement.
        assert json.loads(samples.messages[-1].data)["new_measurement"] is False

        node._latest_target_point = None
        node._control_cycle()
        assert node._controller._last_measurement_stamp_ns is None
        assert node._controller._distance_i_mps == 0.0

        def prime_history():
            node._controller._last_measurement_stamp_ns = source_stamp_ns
            node._controller._direction_d_filtered_radps = 0.25
            node._controller._distance_i_mps = 0.01

        _fresh_point(node)
        prime_history()
        node._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(
            TransformException("test missing transform")
        )
        node._control_cycle()
        assert node._controller._last_measurement_stamp_ns is None
        assert node._controller._direction_d_filtered_radps == 0.0
        assert node._controller._distance_i_mps == 0.0

        _fresh_point(node)
        prime_history()
        node._latest_target_received_monotonic = time.monotonic() - 1.0
        node._control_cycle()
        assert node._controller._last_measurement_stamp_ns is None
        assert node._controller._distance_i_mps == 0.0

        prime_history()
        node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert node._controller._last_measurement_stamp_ns is None
        assert node._controller._direction_d_filtered_radps == 0.0
        assert node._controller._distance_i_mps == 0.0
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert node._controller._last_measurement_stamp_ns is None

        prime_history()
        node._last_ros_time_ns = source_stamp_ns + 10_000
        assert not node._observe_ros_time(source_stamp_ns, time.monotonic(), 0.5)
        assert node._controller._last_measurement_stamp_ns is None
        assert node._controller._direction_d_filtered_radps == 0.0
        assert node._controller._distance_i_mps == 0.0

        node._tf_buffer.transform = lambda message, *_: message
        reacquired = _fresh_point(node)
        node._control_cycle()
        reacquired_sample = json.loads(samples.messages[-1].data)
        assert reacquired_sample["direction_d_rad"] == 0.0
        assert reacquired_sample["distance_i_mps"] == 0.0
        assert reacquired_sample["reset_reason"] == "ros_clock_regression"
        assert commands.messages == []  # Default publish_cmd_vel remains false with extensions active.

        controller = FollowPController(distance_ki=0.005)
        controller.update(2.45, 0.0, measurement_stamp_ns=1_000_000_000)
        command = controller.update(2.45, 0.0, measurement_stamp_ns=2_000_000_000)
        node._controller = controller
        reacquired.point.x = 2.45
        reacquired.point.y = 0.0
        node._latest_target_point = reacquired
        node._publish_control_sample(command, reacquired, 123)
        integral_sample = json.loads(samples.messages[-1].data)
        assert integral_sample["distance_i_mps"] == pytest.approx(0.001)
        assert integral_sample["distance_integral_m_s"] == pytest.approx(0.2)
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()




@pytest.mark.parametrize("stamp_ns, outcome, expected", [
    (None, "disabled", None),
    (1_100_000_000, "stale_ros_stamp", -.1),
    (400_000_000, "stale_ros_stamp", .6),
    (800_000_000, "disabled", .2),
])
def test_diagnostics_report_the_point_and_clocks_used_by_the_cycle(stamp_ns, outcome, expected):
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        spy = _PublisherSpy()
        node._control_diagnostics_publisher = spy
        node._last_diagnostics_monotonic = 90.
        if stamp_ns is not None:
            point = PointStamped()
            point.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
            node._latest_target_point = point
            node._latest_target_received_monotonic = 99.95
        node._cycle_counts["timer_calls"] = 1
        node._record_cycle(outcome, False, 1_000_000_000, 100.)
        value = json.loads(spy.messages[-1].data)
        assert value["cycle_outcome"] == outcome
        assert value["sim_time_ns"] == 1_000_000_000
        assert value["target_stamp_ns"] == stamp_ns
        if expected is None:
            assert value["target_ros_age_s"] is None
            assert value["target_received_age_s"] is None
        else:
            assert value["target_ros_age_s"] == pytest.approx(expected)
            assert value["target_received_age_s"] == pytest.approx(.05)
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_target_reader_keeps_latest_point_without_relaxing_freshness():
    from rclpy.qos import HistoryPolicy, ReliabilityPolicy
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        qos = node._target_point_subscription.qos_profile
        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.depth == 1
        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert node._detection_subscription.qos_profile.depth == 10
        assert node.get_parameter("max_target_age_s").value == .5
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        point = _fresh_point(node)
        original = node.get_clock().now()
        for offset in (-.6, .1):
            point.header.stamp = Time(
                nanoseconds=original.nanoseconds + round(offset * 1e9),
                clock_type=original.clock_type).to_msg()
            node._on_target_point(point)
            node._control_cycle()
        assert node._cycle_counts["stale_ros_stamp"] == 2
        assert point.header.stamp == Time(
            nanoseconds=original.nanoseconds + 100_000_000,
            clock_type=original.clock_type).to_msg()
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()
