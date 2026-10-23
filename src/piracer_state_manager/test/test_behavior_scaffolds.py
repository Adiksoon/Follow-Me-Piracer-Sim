import pytest

rclpy = pytest.importorskip("rclpy")
from std_srvs.srv import SetBool

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
