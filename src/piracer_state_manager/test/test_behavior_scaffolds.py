import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray

from piracer_state_manager.acquire_node import AcquireNode
from piracer_state_manager.follow_node import PiracerFollowNode


def test_acquire_and_follow_scaffolds_refuse_enable_and_accept_disable():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    nodes = [AcquireNode(), PiracerFollowNode()]
    try:
        for node, label in ((nodes[0], "Acquire"), (nodes[1], "Follow")):
            enable = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
            assert not enable.success
            assert "not implemented" in enable.message
            assert node._enabled is False

            disable = node._set_enabled(SetBool.Request(data=False), SetBool.Response())
            assert disable.success
            assert disable.message == f"{label} disabled"
            assert node._enabled is False
    finally:
        for node in nodes:
            node.destroy_node()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()


def test_follow_caches_latest_detection_and_target_point_without_enabling():
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    node = PiracerFollowNode()
    try:
        assert node._latest_detections is None
        assert node._latest_target_point is None

        detections = Detection2DArray()
        detections.header.stamp.sec = 1
        target_point = PointStamped()
        target_point.header.stamp.sec = 1
        node._on_detections(detections)
        node._on_target_point(target_point)

        assert node._latest_detections is detections
        assert node._latest_target_point is target_point
        assert node._enabled is False

        response = node._set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert not response.success
        assert node._enabled is False
    finally:
        node.destroy_node()
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()
