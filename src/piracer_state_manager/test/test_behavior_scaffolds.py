import json
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool

from piracer_state_manager.acquire_node import AcquireNode
from piracer_state_manager.follow_node import PiracerFollowNode


class _PublisherSpy:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_acquire_still_refuses_enable_and_accepts_disable():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = AcquireNode()
    try:
        enable = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert not enable.success
        assert "not implemented" in enable.message
        disable = node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert disable.success
    finally:
        node.destroy_node()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_accepts_target_and_can_enable():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        assert node._latest_target_point is None

        target_point = PointStamped()
        target_point.header.stamp.sec = 1
        node._on_target_point(target_point)
        # A nonzero stamp is cached. The control loop separately checks ROS age.
        assert node._latest_target_point is target_point

        enable = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert enable.success
        assert "waiting for a fresh target" in enable.message
        assert node._enabled
        assert node._publish_cmd_vel is False

        disable = node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert disable.success
        assert node._latest_target_point is None
        assert not node._enabled
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_default_blocks_every_command_but_still_calculates_controller_output():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    preview_spy = _PublisherSpy()
    node._command_publisher = spy
    node._control_sample_publisher = preview_spy
    try:
        assert not node.get_parameter("publish_cmd_vel").value
        controller = node._controller
        controller_update = controller.update
        calls = []

        def update(x, y, *, measurement_stamp_ns=None):
            calls.append((x, y, measurement_stamp_ns))
            return controller_update(
                x, y, measurement_stamp_ns=measurement_stamp_ns
            )

        node._controller.update = update
        enable = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert enable.success

        point = PointStamped()
        point.header.frame_id = "base_link"
        point.header.stamp = node.get_clock().now().to_msg()
        point.point.x = 3.0
        point.point.y = 0.5
        node._on_target_point(point)
        node._control_cycle()

        expected_stamp_ns = (
            point.header.stamp.sec * 1_000_000_000
            + point.header.stamp.nanosec
        )
        assert calls == [(3.0, 0.5, expected_stamp_ns)]
        assert not spy.messages
        assert len(preview_spy.messages) == 1
        sample = json.loads(preview_spy.messages[0].data)
        assert sample["sim_time_ns"] > 0
        assert sample["publish_cmd_vel_enabled"] is False
        assert sample["distance_m"] == pytest.approx((3.0**2 + 0.5**2) ** 0.5)
        assert sample["distance_error_m"] == pytest.approx(sample["distance_m"] - 2.25)
        assert sample["command_linear_x_mps"] == pytest.approx(0.05)
        assert sample["direction_error_rad"] > 0.0
        assert sample["steering_angle_command_rad"] > 0.0
    finally:
        node.stop_and_destroy()
        assert not spy.messages
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_default_suppresses_zero_commands_on_timeout_error_and_disable():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    preview_spy = _PublisherSpy()
    node._command_publisher = spy
    node._control_sample_publisher = preview_spy
    destroyed = False
    try:
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        node._control_cycle()  # No target.

        point = PointStamped()
        point.header.frame_id = "base_link"
        point.header.stamp = node.get_clock().now().to_msg()
        point.point.x = 3.0
        node._on_target_point(point)
        node._latest_target_received_monotonic = time.monotonic() - 1.0
        node._control_cycle()  # Stale receipt timeout.

        node._on_target_point(point)
        node._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(ValueError("no TF"))
        node._control_cycle()  # TF/controller error.

        node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        node.stop_and_destroy()
        destroyed = True
        assert not spy.messages
        assert not preview_spy.messages
    finally:
        if not destroyed:
            node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_publishes_proportional_command_when_explicitly_enabled():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init(args=["--ros-args", "-p", "publish_cmd_vel:=true"])
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    preview_spy = _PublisherSpy()
    node._command_publisher = spy
    node._control_sample_publisher = preview_spy
    destroyed = False
    try:
        assert node._publish_cmd_vel
        point = PointStamped()
        point.header.frame_id = "base_link"
        point.header.stamp = node.get_clock().now().to_msg()
        point.point.x = 3.0
        point.point.y = 0.5
        node._on_target_point(point)
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())

        node._control_cycle()

        assert len(spy.messages) == 2  # enable sends zero, control sends the P command
        command = spy.messages[-1]
        assert command.linear.x > 0.0
        assert command.angular.z > 0.0
        assert command.linear.x <= 0.05
        assert len(preview_spy.messages) == 1
        node.stop_and_destroy()
        destroyed = True
        assert len(spy.messages) == 3
        assert spy.messages[-1].linear.x == 0.0
        assert spy.messages[-1].angular.z == 0.0
    finally:
        if not destroyed:
            node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_keeps_stop_commands_for_timeout_and_error_when_publishing_enabled():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init(args=["--ros-args", "-p", "publish_cmd_vel:=true"])
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    node._command_publisher = spy
    try:
        node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        node._control_cycle()  # Missing target.
        assert len(spy.messages) == 2
        assert spy.messages[-1].linear.x == 0.0

        point = PointStamped()
        point.header.frame_id = "base_link"
        point.header.stamp = node.get_clock().now().to_msg()
        point.point.x = 3.0
        node._on_target_point(point)
        node._latest_target_received_monotonic = time.monotonic() - 1.0
        node._control_cycle()  # Stale target.
        assert len(spy.messages) == 3
        assert spy.messages[-1].linear.x == 0.0

        node._on_target_point(point)
        node._tf_buffer.transform = lambda *_: (_ for _ in ()).throw(ValueError("no TF"))
        node._control_cycle()  # Transform/controller error.
        assert len(spy.messages) == 4
        assert spy.messages[-1].linear.x == 0.0

        node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert len(spy.messages) == 5
        assert spy.messages[-1].linear.x == 0.0
        assert spy.messages[-1].angular.z == 0.0
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_disabled_follow_is_silent_and_disable_publishes_one_stop_command():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init(args=["--ros-args", "-p", "publish_cmd_vel:=true"])
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    node._command_publisher = spy
    destroyed = False
    try:
        assert node._publish_cmd_vel
        node._control_cycle()
        assert spy.messages == []

        enabled = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert enabled.success
        assert len(spy.messages) == 1

        node._control_cycle()
        assert len(spy.messages) == 2
        assert spy.messages[-1].linear.x == 0.0
        assert spy.messages[-1].angular.z == 0.0

        disabled = node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert disabled.success
        assert len(spy.messages) == 3
        assert spy.messages[-1].linear.x == 0.0
        assert spy.messages[-1].angular.z == 0.0

        repeated_disable = node._set_enabled(
            SetBool.Request(data=False), SetBool.Response()
        )
        assert repeated_disable.success
        node._control_cycle()
        assert len(spy.messages) == 3
    finally:
        if not destroyed:
            node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()
