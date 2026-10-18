from copy import deepcopy
import math
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from tf2_ros import TransformException

from piracer_navigation.follow_goal_controller import FollowGoalController


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _GoalFuture:
    def add_done_callback(self, callback):
        self.callback = callback


class _ActionClient:
    def __init__(self):
        self.goals = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _GoalFuture()


class _GoalHandle:
    def __init__(self):
        self.cancel_count = 0

    def cancel_goal_async(self):
        self.cancel_count += 1


class _ExactStampBuffer:
    def transform(self, point, target_frame, timeout):
        transformed = deepcopy(point)
        transformed.header.frame_id = target_frame
        if target_frame == "base_link":
            transformed.point.x = 2.50
            transformed.point.y = 0.10
        else:
            transformed.point.x = 3.0
            transformed.point.y = 4.0
        return transformed

    def lookup_transform(self, target_frame, source_frame, stamp, timeout):
        result = TransformStamped()
        result.header.frame_id = target_frame
        result.child_frame_id = source_frame
        result.header.stamp = stamp.to_msg()
        result.transform.translation.x = 1.0
        result.transform.translation.y = 2.0
        result.transform.rotation.w = 1.0
        return result


class _MissingTransformBuffer:
    def transform(self, point, target_frame, timeout):
        raise TransformException("no transform at source stamp")


def _point_at(node, stamp):
    point = PointStamped()
    point.header.frame_id = "sim_camera_optical_frame"
    point.header.stamp = stamp.to_msg()
    point.point.x = 2.50
    point.point.y = 0.0
    return point


def _node_with_collectors():
    node = FollowGoalController()
    node._tf_buffer = _ExactStampBuffer()
    node._goal_publisher = _Publisher()
    node._error_publisher = _Publisher()
    node._correction_publisher = _Publisher()
    node._stop_publisher = _Publisher()
    return node


def test_navigation_goal_selects_the_follow_specific_tree():
    node = _node_with_collectors()
    client = _ActionClient()
    node._action_client = client
    try:
        result = node.set_parameters([Parameter("enable_navigation", value=True)])
        assert result[0].successful

        node._send_initial_goal_if_enabled(PoseStamped())

        assert len(client.goals) == 1
        tree_path = Path(client.goals[0].behavior_tree)
        expected = (
            Path(get_package_share_directory("piracer_navigation"))
            / "behavior_trees"
            / "follow_visible_target.xml"
        )
        assert tree_path == expected
        assert tree_path.is_file()
    finally:
        node.destroy_node()


def test_close_target_requests_gate_stop_and_does_not_issue_a_goal():
    node = _node_with_collectors()
    client = _ActionClient()
    node._action_client = client
    try:
        result = node.set_parameters([Parameter("enable_navigation", value=True)])
        assert result[0].successful
        point = _point_at(node, node.get_clock().now())
        point.point.x = 1.99

        node._on_target_point(point)

        assert [message.data for message in node._stop_publisher.messages] == [True]
        assert node._goal_publisher.messages == []
        assert node._correction_publisher.messages == []
        assert client.goals == []
    finally:
        node.destroy_node()


def test_close_target_cancels_active_goal_and_repeats_only_one_stop():
    node = _node_with_collectors()
    handle = _GoalHandle()
    node._goal_handle = handle
    node._active_goal = True
    try:
        point = _point_at(node, node.get_clock().now())
        point.point.x = 1.99
        node._on_target_point(point)

        next_point = _point_at(node, node.get_clock().now())
        next_point.point.x = 1.98
        node._on_target_point(next_point)

        assert handle.cancel_count == 1
        assert [message.data for message in node._stop_publisher.messages] == [True]
        assert node._goal_publisher.messages == []
    finally:
        node.destroy_node()


def test_follow_releases_gate_stop_after_target_returns_to_two_meters():
    node = _node_with_collectors()
    node._target_too_close = True
    try:
        point = _point_at(node, node.get_clock().now())
        point.point.x = 2.0

        node._on_target_point(point)

        assert [message.data for message in node._stop_publisher.messages] == [False]
        assert len(node._goal_publisher.messages) == 1
        assert node._goal_publisher.messages[0].header.frame_id == "map"
    finally:
        node.destroy_node()


def test_exact_stamp_transform_produces_map_goal_and_diagnostics_without_action():
    node = _node_with_collectors()
    try:
        stamp = node.get_clock().now()
        node._on_target_point(_point_at(node, stamp))

        assert len(node._goal_publisher.messages) == 1
        goal = node._goal_publisher.messages[0]
        assert goal.header.frame_id == "map"
        assert goal.header.stamp == stamp.to_msg()
        assert goal.pose.position.x == pytest.approx(1.25)
        assert goal.pose.position.y == pytest.approx(2.10)
        expected_yaw = math.atan2(4.0 - 2.10, 3.0 - 1.25)
        assert goal.pose.orientation.z == pytest.approx(
            math.sin(expected_yaw / 2.0)
        )
        assert node._error_publisher.messages[0].vector.z == pytest.approx(
            2.50
        )
        assert node._correction_publisher.messages[0].vector.x == pytest.approx(0.25)
        assert node._correction_publisher.messages[0].vector.y == pytest.approx(0.10)
        assert not node._active_goal and not node._goal_pending
    finally:
        node.destroy_node()


def test_stale_target_is_ignored_before_tf_or_publication():
    node = _node_with_collectors()
    try:
        node._tf_buffer = _MissingTransformBuffer()
        stamp = node.get_clock().now() - Duration(seconds=2.0)
        node._on_target_point(_point_at(node, stamp))
        assert node._goal_publisher.messages == []
        assert node._error_publisher.messages == []
    finally:
        node.destroy_node()


def test_p_parameters_change_live_and_invalid_gain_is_rejected():
    node = _node_with_collectors()
    try:
        result = node.set_parameters([Parameter("range_kp", value=0.5)])
        assert result[0].successful
        assert node._controller.range_axis.kp == pytest.approx(0.5)
        assert node.get_parameter("enable_navigation").value is False

        node._on_target_point(_point_at(node, node.get_clock().now()))
        assert node._correction_publisher.messages[0].vector.x == pytest.approx(0.125)
        assert node._goal_publisher.messages[0].pose.position.x == pytest.approx(1.125)

        rejected = node.set_parameters([Parameter("range_kp", value=-0.1)])
        assert not rejected[0].successful
        assert node._controller.range_axis.kp == pytest.approx(0.5)
        assert not node._active_goal and not node._goal_pending
    finally:
        node.destroy_node()


def test_repeated_target_stamp_is_ignored():
    node = _node_with_collectors()
    try:
        stamp = node.get_clock().now()
        point = _point_at(node, stamp)
        node._on_target_point(point)
        assert len(node._goal_publisher.messages) == 1

        node._on_target_point(point)
        assert len(node._goal_publisher.messages) == 1
        assert len(node._error_publisher.messages) == 1
    finally:
        node.destroy_node()
