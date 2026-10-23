import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray

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


def test_follow_subscribes_to_detections_and_target_and_can_enable():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        assert node._latest_detections is None
        assert node._latest_target_point is None

        detections = Detection2DArray()
        target_point = PointStamped()
        target_point.header.stamp.sec = 1
        node._on_detections(detections)
        node._on_target_point(target_point)
        assert node._latest_detections is detections
        # A nonzero stamp is cached. The control loop separately checks ROS age.
        assert node._latest_target_point is target_point

        enable = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert enable.success
        assert "waiting for a fresh target" in enable.message
        assert node._enabled

        disable = node._set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert disable.success
        assert node._latest_target_point is None
        assert not node._enabled
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_publishes_proportional_command_for_fresh_base_link_point():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    spy = _PublisherSpy()
    node._command_publisher = spy
    try:
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
    finally:
        node.stop_and_destroy()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()
