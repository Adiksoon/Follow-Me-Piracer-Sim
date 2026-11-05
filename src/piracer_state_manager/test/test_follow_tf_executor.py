"""Exercise real TF delivery while FOLLOW's original control executor is busy."""
import threading
import time

import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener

from piracer_state_manager.follow_node import PiracerFollowNode


def _transform(parent, child, stamp_ns, x=0.):
    value = TransformStamped()
    value.header.frame_id = parent
    value.child_frame_id = child
    value.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    value.transform.translation.x = x
    value.transform.rotation.w = 1.
    return value


def _spin(executor):
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


@pytest.mark.parametrize("isolated", [False, True], ids=["old_shared_executor", "isolated_tf"])
def test_tf_remains_current_during_a_blocked_control_callback(isolated):
    assert not rclpy.ok()
    rclpy.init(args=["--ros-args", "-r", "/tf:=/follow_tf_test/dynamic",
                     "-r", "/tf_static:=/follow_tf_test/static",
                     "-r", "__node:=follow_tf_test_control"])
    control = pub_node = executor = spin_thread = listener = None
    blocked, release = threading.Event(), threading.Event()
    try:
        pub_node = Node("follow_tf_test_publisher", use_global_arguments=False)
        dynamic = pub_node.create_publisher(TFMessage, "/follow_tf_test/dynamic", 100)
        static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        static = pub_node.create_publisher(TFMessage, "/follow_tf_test/static", static_qos)
        # Publish once before the listener exists, testing retained static data.
        static.publish(TFMessage(transforms=[_transform("base_link", "test_camera", 0, .2)]))
        if isolated:
            control = PiracerFollowNode()
            control._control_timer.cancel()
            buffer = control._tf_buffer
            assert control._tf_node.get_name() == "piracer_follow_tf_listener"
            assert control._tf_node.get_name() != control.get_name()
            assert control._tf_listener.tf_sub.topic_name == "/follow_tf_test/dynamic"
            assert control._tf_listener.tf_static_sub.topic_name == "/follow_tf_test/static"
        else:
            control = Node("follow_tf_test_control")
            buffer = Buffer()
            listener = TransformListener(buffer, control)

        executor = SingleThreadedExecutor()
        executor.add_node(control)
        spin_thread = threading.Thread(target=_spin, args=(executor,))
        spin_thread.start()
        epoch, origin = 10_000_000_000, time.monotonic_ns()
        now_ns = lambda: epoch + time.monotonic_ns() - origin
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            dynamic.publish(TFMessage(transforms=[
                _transform("vo_odom", "base_link", now_ns(), .5)]))
            if buffer.can_transform("vo_odom", "test_camera", Time()):
                break
            time.sleep(.02)
        assert buffer.can_transform("vo_odom", "test_camera", Time())

        def block_control():
            timer.cancel()
            blocked.set()
            release.wait(timeout=4)
        timer = control.create_timer(.001, block_control)
        assert blocked.wait(timeout=2)
        other_callbacks = []
        guard = control.create_guard_condition(lambda: other_callbacks.append("control"))
        guard.trigger()
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            dynamic.publish(TFMessage(transforms=[
                _transform("vo_odom", "base_link", now_ns(), .5)]))
            time.sleep(.02)
        assert not release.is_set() and not other_callbacks
        checked_now = now_ns()
        latest = buffer.lookup_transform("vo_odom", "base_link", Time())
        stamp_ns = latest.header.stamp.sec*1_000_000_000 + latest.header.stamp.nanosec
        age = (checked_now-stamp_ns)/1e9
        if isolated:
            assert 0 <= age < .3
            assert control.get_parameter("max_localization_age_s").value == 1.
            assert control._tf_pose("vo_odom", checked_now) == pytest.approx((.5, 0., 0.))
            camera = buffer.lookup_transform("vo_odom", "test_camera", Time())
            assert camera.transform.translation.x == pytest.approx(.7)
            assert control._target_point_subscription.callback_group is control.default_callback_group
            assert control._control_timer.callback_group is control.default_callback_group
        else:
            assert age > 1.  # The unchanged freshness gate would reject this.
        release.set()
        deadline = time.monotonic()+2
        while not other_callbacks and time.monotonic()<deadline:
            time.sleep(.01)
        assert other_callbacks == ["control"]
    finally:
        release.set()
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=3)
            assert not spin_thread.is_alive()
        if listener is not None:
            listener.unregister()
        if control is not None:
            if isolated:
                worker = control._tf_thread
                control.stop_and_destroy()
                assert not worker.is_alive() and control._tf_node is None
            else:
                control.destroy_node()
        if pub_node is not None:
            pub_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_stop_cleanup_joins_receiver_even_when_final_zero_cannot_publish():
    rclpy.init()
    node = PiracerFollowNode()
    worker = node._tf_thread
    node._enabled = True
    node._publish_cmd_vel = True
    class ClosedPublisher:
        def publish(self, _):
            raise RuntimeError("publisher unavailable")
    node._command_publisher = ClosedPublisher()
    try:
        with pytest.raises(RuntimeError, match="publisher unavailable"):
            node.stop_and_destroy()
        assert not worker.is_alive() and node._tf_node is None
    finally:
        rclpy.try_shutdown()


def test_context_shutdown_releases_tf_receiver():
    rclpy.init()
    node = PiracerFollowNode()
    worker = node._tf_thread
    node._enabled = True
    rclpy.shutdown()
    node.stop_and_destroy()
    assert not worker.is_alive() and node._tf_node is None
