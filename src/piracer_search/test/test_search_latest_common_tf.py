"""Offline tests of SEARCH against the installed TF2 Buffer, without ROS nodes."""
from collections import deque
import importlib
import types
import unittest

from geometry_msgs.msg import TransformStamped
from rclpy.clock import ClockType
from rclpy.time import Time
from tf2_ros import Buffer
from piracer_search.search_node import SearchNode

NOW = 12_000_000_000


def transform(parent, child, ns, x=0.0):
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = parent, child
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1_000_000_000)
    msg.transform.translation.x = x
    msg.transform.rotation.w = 1.0
    return msg


class CountingBuffer:
    def __init__(self):
        self.inner = Buffer()
        self.calls = []

    def lookup_transform(self, target, source, query):
        self.calls.append((target, source, query.nanoseconds))
        return self.inner.lookup_transform(target, source, query)

    def clear(self):
        self.inner.clear()

class RealTfLatestCommonTest(unittest.TestCase):
    def test_mock_fixture_preserves_real_modules_and_package_aliases(self):
        import piracer_search
        import test_search_node as fixtures

        real_search = importlib.import_module("piracer_search.search_node")
        real_timing = importlib.import_module("piracer_search.tf_timing")
        real_receiver = real_search.SearchTfReceiver
        fixtures.SearchNodeCallbackTest.setUpClass()

        self.assertIsNot(real_search, fixtures.SearchNodeCallbackTest.module)
        self.assertIsNot(real_timing, fixtures.SearchNodeCallbackTest.tf_module)
        self.assertIs(real_search, importlib.import_module("piracer_search.search_node"))
        self.assertIs(real_timing, importlib.import_module("piracer_search.tf_timing"))
        self.assertIs(real_search, piracer_search.search_node)
        self.assertIs(real_timing, piracer_search.tf_timing)
        self.assertIs(SearchNode, real_search.SearchNode)
        self.assertIs(real_receiver, real_search.SearchTfReceiver)

    def make_node(self, info_stamp=NOW - 50_000_000,
                  tf_future_lead_sec=0.1):
        node = object.__new__(SearchNode)
        node._localization_available = True
        node._localization_generation = 0
        node._localization_invalidated_at_ns = None
        node._last_accepted_odom_stamp_ns = None
        node._last_accepted_full_tf_stamp_ns = None
        node._last_ros_time_ns = None
        node._tf_future_lead_ns = round(tf_future_lead_sec * 1_000_000_000)
        node._clock_epoch = 0
        node._localization_reason = None
        class Handle:
            count = 0
            def cancel_goal_async(self):
                self.count += 1
        handle = Handle()
        node._owned_goals = {1: {
            "token": 1, "handle": handle, "state": "active",
            "cancel_requested": False, "cancel_reason": None,
            "localization_invalidated": False,
        }}
        node._current_goal_token = 1
        node.get_parameter = lambda name: types.SimpleNamespace(
            value=1.0 if name == "max_localization_age_sec" else "base_link")
        node.get_clock = lambda: types.SimpleNamespace(
            clock_type=ClockType.ROS_TIME,
            now=lambda: Time(nanoseconds=NOW, clock_type=ClockType.ROS_TIME))
        node.get_logger = lambda: types.SimpleNamespace(info=lambda *a: None,
                                                       warning=lambda *a: None)
        node._odom_info = deque([types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=Time(nanoseconds=info_stamp).to_msg()),
            lost=False)], maxlen=10)
        node._tf_buffer = CountingBuffer()
        return node, handle

    def static_map(self, node):
        node._tf_buffer.inner.set_transform_static(transform("map", "vo_odom", 0), "offline")

    def test_boundaries_with_real_tf2(self):
        for offset, accepted in ((-62_000_000, True), (41_000_000, True),
                             (100_000_000, True), (100_000_001, False),
                             (-1_000_000_000, True), (-1_000_000_001, False)):
            with self.subTest(offset=offset):
                node, handle = self.make_node()
                self.static_map(node)
                sample = transform("vo_odom", "base_link", NOW + offset, 0.75)
                node._tf_buffer.inner.set_transform(sample, "offline")
                result = node._localization_pose("map")
                self.assertEqual(accepted, result is not None)
                self.assertEqual([("map", "base_link", 0)], node._tf_buffer.calls)
                self.assertEqual(0 if accepted else 1, handle.count)
                if accepted:
                    self.assertEqual(NOW + offset, node._last_accepted_full_tf_stamp_ns)
                    self.assertAlmostEqual(0.75, result[0])
                self.assertEqual(NOW + offset, node._stamp_ns(sample))

    def test_simulation_200ms_future_tf_boundaries_with_real_tf2(self):
        for offset, accepted in ((133_000_000, True), (200_000_000, True),
                                 (200_000_001, False),
                                 (-1_000_000_001, False)):
            with self.subTest(offset=offset):
                node, handle = self.make_node(tf_future_lead_sec=0.2)
                self.static_map(node)
                sample = transform("vo_odom", "base_link", NOW + offset, 0.75)
                node._tf_buffer.inner.set_transform(sample, "offline")
                result = node._localization_pose("map")
                self.assertEqual(accepted, result is not None)
                self.assertEqual(0 if accepted else 1, handle.count)
                if accepted:
                    self.assertEqual(NOW + offset,
                                     node._last_accepted_full_tf_stamp_ns)

    def test_common_time_uses_future_interpolation_endpoint(self):
        node, _ = self.make_node()
        buf = node._tf_buffer.inner
        buf.set_transform(transform("map", "vo_odom", NOW - 50_000_000, 1.0), "offline")
        buf.set_transform(transform("map", "vo_odom", NOW + 41_000_000, 2.0), "offline")
        buf.set_transform(transform("vo_odom", "base_link", NOW - 50_000_000, 0.0), "offline")
        buf.set_transform(transform("vo_odom", "base_link", NOW + 200_000_000, 2.5), "offline")
        self.assertAlmostEqual(2.91, node._localization_pose("map")[0])
        self.assertEqual(NOW + 41_000_000, node._last_accepted_full_tf_stamp_ns)

    def test_unused_future_edge_sample_does_not_reject_common_time(self):
        node, _ = self.make_node()
        buf = node._tf_buffer.inner
        for stamp in (NOW - 50_000_000, NOW + 41_000_000):
            buf.set_transform(transform("map", "vo_odom", stamp, 2.0), "offline")
            buf.set_transform(transform("vo_odom", "base_link", stamp, 1.0), "offline")
        buf.set_transform(transform("vo_odom", "base_link", NOW + 200_000_000, 50.0), "offline")
        self.assertAlmostEqual(3.0, node._localization_pose("map")[0])
        self.assertEqual(NOW + 41_000_000, node._last_accepted_full_tf_stamp_ns)

    def test_disconnected_nonoverlapping_or_stale_limiting_edge_fails(self):
        for condition in ("disconnected", "nonoverlap", "stale_map", "stale_native"):
            with self.subTest(condition=condition):
                node, handle = self.make_node()
                buf = node._tf_buffer.inner
                if condition in ("stale_map", "stale_native"):
                    old = NOW - 1_000_000_001
                    map_latest = old if condition == "stale_map" else NOW
                    native_latest = old if condition == "stale_native" else NOW
                    for parent, child, latest in (("map", "vo_odom", map_latest),
                                                   ("vo_odom", "base_link", native_latest)):
                        buf.set_transform(transform(parent, child, old - 1), "offline")
                        buf.set_transform(transform(parent, child, latest), "offline")
                    common = buf.lookup_transform("map", "base_link", Time())
                    self.assertEqual(old, node._stamp_ns(common))
                    self.assertIsNone(node._localization_pose("map"))
                    self.assertIn("older than max_localization_age_sec", node._localization_reason)
                    self.assertEqual(1, handle.count)
                    continue
                if condition != "disconnected":
                    map_ns = NOW - 100_000_000
                    buf.set_transform(transform("map", "vo_odom", map_ns), "offline")
                    if condition == "nonoverlap":
                        buf.set_transform(transform("map", "vo_odom", NOW - 90_000_000), "offline")
                native_ns = NOW - 100_000_000
                if condition == "nonoverlap":
                    native_ns = NOW - 30_000_000
                buf.set_transform(transform("vo_odom", "base_link", native_ns), "offline")
                self.assertIsNone(node._localization_pose("map"))
                self.assertEqual(1, handle.count)

    def test_stale_info_prevents_lookup(self):
        node, handle = self.make_node(NOW - 1_000_000_001)
        self.static_map(node)
        node._tf_buffer.inner.set_transform(transform("vo_odom", "base_link", NOW), "offline")
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual([], node._tf_buffer.calls)
        self.assertEqual(1, handle.count)

    def test_barrier_regression_and_lost_keep_fail_safe(self):
        node, handle = self.make_node(info_stamp=NOW - 10_000_000)
        self.static_map(node)
        node._localization_invalidated_at_ns = NOW - 50_000_000
        node._tf_buffer.inner.set_transform(
            transform("vo_odom", "base_link", NOW - 50_000_000), "offline")
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(1, handle.count)

        node, handle = self.make_node(info_stamp=NOW - 10_000_000)
        self.static_map(node)
        node._localization_invalidated_at_ns = NOW - 100_000_000
        node._tf_buffer.inner.set_transform(
            transform("vo_odom", "base_link", NOW - 50_000_000), "offline")
        self.assertIsNotNone(node._localization_pose("map"))
        self.assertIsNone(node._localization_invalidated_at_ns)
        self.assertEqual(0, handle.count)

        node, handle = self.make_node()
        self.static_map(node)
        node._tf_buffer.inner.set_transform(
            transform("vo_odom", "base_link", NOW + 41_000_000), "offline")
        self.assertIsNotNone(node._localization_pose("map"))
        node._tf_buffer.inner.clear()
        self.static_map(node)
        node._tf_buffer.inner.set_transform(
            transform("vo_odom", "base_link", NOW + 40_000_000), "offline")
        self.assertIsNone(node._localization_pose("map"))
        self.assertIn("timestamp regressed", node._localization_reason)
        self.assertEqual(1, handle.count)

        node, handle = self.make_node()
        info = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=Time(nanoseconds=NOW).to_msg()), lost=True)
        node._odom_info_callback(info)
        self.assertEqual(1, handle.count)
        self.assertEqual([], node._tf_buffer.calls)

    def test_zero_stamp_and_final_clock_changes_fail_closed(self):
        node, handle = self.make_node()
        self.static_map(node)
        node._tf_buffer.inner.set_transform_static(
            transform("vo_odom", "base_link", 0), "offline")
        self.assertIsNone(node._localization_pose("map"))
        self.assertIn("stamp is zero", node._localization_reason)
        self.assertEqual(1, handle.count)

        for final_now, expected in ((NOW + 950_000_000, "older"),
                                    (NOW - 1, "clock moved backwards")):
            with self.subTest(final_now=final_now):
                node, handle = self.make_node(info_stamp=NOW)
                self.static_map(node)
                node._tf_buffer.inner.set_transform(
                    transform("vo_odom", "base_link", NOW - 100_000_000), "offline")
                times = iter((NOW, final_now, final_now))
                node.get_clock = lambda: types.SimpleNamespace(
                    clock_type=ClockType.ROS_TIME,
                    now=lambda: Time(nanoseconds=next(times), clock_type=ClockType.ROS_TIME))
                self.assertIsNone(node._localization_pose("map"))
                self.assertIn(expected, node._localization_reason)
                self.assertEqual(1, handle.count)


if __name__ == "__main__":
    unittest.main()
