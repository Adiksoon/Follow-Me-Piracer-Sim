import importlib
import sys
import types
import unittest
from collections import deque
from unittest.mock import patch


class Future:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.callback = None

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def add_done_callback(self, callback):
        self.callback = callback


class FakeHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancel_count = 0
        self.cancel_future = Future()
        self.result_future = Future()

    def cancel_goal_async(self):
        self.cancel_count += 1
        return self.cancel_future

    def get_result_async(self):
        return self.result_future


class FakeAction:
    def __init__(self):
        self.ready = True
        self.sent = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        future = Future()
        self.sent.append((goal, future))
        return future


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class SearchNodeCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy = types.ModuleType("rclpy")
        rclpy.__path__ = []
        rclpy.init = lambda **kwargs: None
        rclpy.spin = lambda node: None
        rclpy.shutdown = lambda: None
        node_module = types.ModuleType("rclpy.node")

        class FakeNode:
            def __init__(self, name):
                self.parameters = {}
                self.subscriptions = []

            def declare_parameter(self, name, value):
                self.parameters[name] = value

            def get_parameter(self, name):
                return types.SimpleNamespace(value=self.parameters[name])

            def create_subscription(self, message_type, topic, callback, qos):
                record = types.SimpleNamespace(
                    message_type=message_type, topic=topic, callback=callback, qos=qos
                )
                self.subscriptions.append(record)
                return record

            def create_publisher(self, *args):
                return FakePublisher()

            def create_service(self, *args):
                return types.SimpleNamespace()

            def create_timer(self, *args):
                return types.SimpleNamespace()

            def get_clock(self):
                return types.SimpleNamespace(
                    clock_type="ros",
                    now=lambda: types.SimpleNamespace(
                        nanoseconds=10_000_000_000, to_msg=lambda: "sim-stamp"
                    ),
                )

            def get_logger(self):
                return types.SimpleNamespace(
                    warning=lambda *args: None,
                    info=lambda *args: None,
                    debug=lambda *args: None,
                )

        node_module.Node = FakeNode

        class FakeQoSProfile:
            def __init__(self, *, history, depth, reliability, durability):
                self.history = history
                self.depth = depth
                self.reliability = reliability
                self.durability = durability

        qos_module = types.ModuleType("rclpy.qos")
        qos_module.QoSProfile = FakeQoSProfile
        qos_module.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
        qos_module.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort")
        qos_module.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient_local", VOLATILE="volatile")
        qos_module.qos_profile_sensor_data = FakeQoSProfile(
            history="keep_last", depth=5, reliability="best_effort", durability="volatile"
        )
        action_module = types.ModuleType("rclpy.action")
        action_module.ActionClient = type("ActionClient", (), {
            "__init__": lambda self, *args: None,
        })
        time_module = types.ModuleType("rclpy.time")
        class FakeTime:
            def __init__(self, *, nanoseconds=0, clock_type="ros"):
                self.nanoseconds = nanoseconds
                self.clock_type = clock_type

        time_module.Time = FakeTime

        class FakePoseStamped:
            def __init__(self):
                self.header = types.SimpleNamespace(frame_id=None, stamp=None)
                self.pose = types.SimpleNamespace(
                    position=types.SimpleNamespace(x=None, y=None, z=None),
                    orientation=types.SimpleNamespace(z=None, w=None),
                )

        geometry_module = types.ModuleType("geometry_msgs.msg")
        geometry_module.PoseStamped = FakePoseStamped
        geometry_parent = types.ModuleType("geometry_msgs")
        geometry_parent.__path__ = []
        nav2_action = type("NavigateToPose", (), {"Goal": type("Goal", (), {})})
        nav2_action_module = types.ModuleType("nav2_msgs.action")
        nav2_action_module.NavigateToPose = nav2_action
        nav2_parent = types.ModuleType("nav2_msgs")
        nav2_parent.__path__ = []
        nav_module = types.ModuleType("nav_msgs.msg")
        nav_module.OccupancyGrid = type("OccupancyGrid", (), {})
        nav_parent = types.ModuleType("nav_msgs")
        nav_parent.__path__ = []
        rtabmap_module = types.ModuleType("rtabmap_msgs.msg")
        rtabmap_module.OdomInfo = type("OdomInfo", (), {})
        rtabmap_parent = types.ModuleType("rtabmap_msgs")
        rtabmap_parent.__path__ = []
        service_module = types.ModuleType("std_srvs.srv")
        service_module.SetBool = type("SetBool", (), {})
        service_parent = types.ModuleType("std_srvs")
        service_parent.__path__ = []
        tf_module = types.ModuleType("tf2_ros")
        class FakeTFException(Exception):
            pass

        tf_module.LookupException = type("LookupException", (FakeTFException,), {})
        tf_module.ConnectivityException = type("ConnectivityException", (FakeTFException,), {})
        tf_module.ExtrapolationException = type("ExtrapolationException", (FakeTFException,), {})
        tf_module.TimeoutException = type("TimeoutException", (FakeTFException,), {})
        tf_module.Buffer = type("Buffer", (), {})
        tf_module.TransformListener = type("TransformListener", (), {
            "__init__": lambda self, *args: None,
        })
        marker_module = types.ModuleType("visualization_msgs.msg")
        marker_module.MarkerArray = type("MarkerArray", (), {"__init__": lambda self: setattr(self, "markers", [])})
        marker_parent = types.ModuleType("visualization_msgs")
        marker_parent.__path__ = []
        action_status = types.ModuleType("action_msgs.msg")
        action_status.GoalStatus = types.SimpleNamespace(
            STATUS_SUCCEEDED=4, STATUS_ABORTED=6, STATUS_CANCELED=5
        )
        action_parent = types.ModuleType("action_msgs")
        action_parent.__path__ = []

        marker_helpers = types.ModuleType("piracer_search.search_markers")
        marker_helpers._candidate_markers = lambda *args, **kwargs: []
        marker_helpers._stale_marker_deletions = lambda *args, **kwargs: []

        modules = {
            "rclpy": rclpy,
            "rclpy.node": node_module,
            "rclpy.qos": qos_module,
            "rclpy.action": action_module,
            "rclpy.time": time_module,
            "geometry_msgs": geometry_parent,
            "geometry_msgs.msg": geometry_module,
            "nav2_msgs": nav2_parent,
            "nav2_msgs.action": nav2_action_module,
            "nav_msgs": nav_parent,
            "nav_msgs.msg": nav_module,
            "rtabmap_msgs": rtabmap_parent,
            "rtabmap_msgs.msg": rtabmap_module,
            "std_srvs": service_parent,
            "std_srvs.srv": service_module,
            "tf2_ros": tf_module,
            "visualization_msgs": marker_parent,
            "visualization_msgs.msg": marker_module,
            "action_msgs": action_parent,
            "action_msgs.msg": action_status,
            "piracer_search.search_markers": marker_helpers,
        }
        with patch.dict(sys.modules, modules):
            cls.module = importlib.import_module("piracer_search.search_node")
        cls.marker_helpers = marker_helpers

    def make_node(self):
        node = object.__new__(self.module.SearchNode)
        node._enabled = True
        node._search_epoch = 0
        node._map = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="map")
        )
        node._frontier_key = ((1, 2),)
        node._failed_attempts = {}
        node._completed_candidates = set()
        node._retry_target = None
        node._replacement_target = None
        node._goal_sequence = 0
        node._pending_goal = None
        node._active_goal = None
        node._challenger_identity = None
        node._challenger_target = None
        node._challenger_epoch = None
        node._challenger_count = 0
        node._published_marker_keys = set()
        node._odom_info = deque(maxlen=10)
        node._localization_available = False
        node._localization_invalidated_at_ns = None
        node._last_accepted_odom_stamp_ns = None
        node._last_ros_time_ns = None
        node._localization_reason = None
        node._now_ns = 10_000_000_000
        node._tf_buffer = type("TFBuffer", (), {"clear": lambda self: None})()
        node._action = FakeAction()
        node._markers = FakePublisher()
        node._logger = type(
            "Logger", (), {"warning": lambda *args: None, "info": lambda *args: None,
                           "debug": lambda *args: None}
        )()
        node.get_logger = lambda: node._logger
        params = {
            "behavior_tree": "/share/search_navigate_to_pose.xml",
            "camera.horizontal_fov_deg": 65.0,
            "camera.max_range_m": 5.0,
            "base_frame": "base_link",
            "max_localization_age_sec": 0.5,
            "odom_info_topic": "/visual_vo/odom_info",
            "goal_switch_margin": 0.10,
        }
        node.get_parameter = lambda name: types.SimpleNamespace(value=params[name])
        node.get_clock = lambda: types.SimpleNamespace(
            clock_type="ros",
            now=lambda: types.SimpleNamespace(
                nanoseconds=node._now_ns, to_msg=lambda: "sim-stamp"
            ),
        )
        node._grid = lambda: {"width": 1, "height": 1, "resolution": 1.0,
                              "origin": [0.0, 0.0, 0.0], "data": (0,)}
        node._localization_pose = lambda frame: (1.0, 2.0, 0.5)
        return node

    def make_localization_node(self):
        node = self.make_node()
        del node._localization_pose

        class TFBuffer:
            def __init__(self):
                self.calls = []
                self.missing = set()
                self.errors = {}

            def lookup_transform(self, target, source, stamp):
                self.calls.append((target, source, stamp.nanoseconds))
                error = self.errors.get((target, stamp.nanoseconds))
                if error is None:
                    error = self.errors.get(target)
                if error is not None:
                    raise error
                if target in self.missing:
                    raise self.expected_exception(target)
                return types.SimpleNamespace(transform=types.SimpleNamespace(
                    translation=types.SimpleNamespace(x=1.0, y=2.0),
                    rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ))

            def clear(self):
                self.calls.clear()

        node._tf_buffer = TFBuffer()
        node._tf_buffer.expected_exception = self.module.ExtrapolationException
        return node

    @staticmethod
    def odom_info(stamp_ns, lost=False):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=types.SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            )),
            lost=lost,
        )

    @staticmethod
    def candidate(candidate_id, score=0.0, x=1.0):
        return {"id": candidate_id, "score": score, "pose_world": [x, 2.0, 0.5]}

    @classmethod
    def result(cls, candidates):
        ranked = sorted(candidates, key=lambda item: -item["score"])
        return {
            "status": "RECOMMENDATION_AVAILABLE" if candidates else "NO_FRONTIERS",
            "frontier_cells": [[1, 2]],
            "ranking": [item["id"] for item in ranked],
            "candidates": candidates,
        }

    def test_map_subscription_uses_reliable_transient_local_depth_one_qos(self):
        qos = self.module._map_subscription_qos()
        self.assertEqual(1, qos.depth)
        self.assertEqual("keep_last", qos.history)
        self.assertEqual("reliable", qos.reliability)
        self.assertEqual("transient_local", qos.durability)

    def test_odom_info_uses_sensor_data_qos(self):
        self.assertIs(self.module._odom_info_subscription_qos(),
                      self.module.qos_profile_sensor_data)

    def test_search_node_declares_and_wires_project_odom_info_topic(self):
        node = self.module.SearchNode()
        self.assertEqual("/visual_vo/odom_info",
                         node.get_parameter("odom_info_topic").value)
        odom_subscription = next(
            subscription for subscription in node.subscriptions
            if subscription.message_type is self.module.OdomInfo
        )
        self.assertEqual("/visual_vo/odom_info", odom_subscription.topic)

    def test_future_sample_does_not_hide_latest_nonfuture_sample(self):
        node = self.make_localization_node()
        old_stamp = 9_950_000_000
        future_stamp = 10_054_000_000
        node._odom_info_callback(self.odom_info(old_stamp))
        node._odom_info_callback(self.odom_info(future_stamp))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual([("vo_odom", "base_link", old_stamp),
                          ("map", "base_link", old_stamp)], node._tf_buffer.calls)
        node._now_ns = 10_100_000_000
        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual(future_stamp, node._tf_buffer.calls[-1][2])

    def test_only_future_sample_does_not_allow_localization(self):
        node = self.make_localization_node()
        node._odom_info_callback(self.odom_info(10_054_000_000))
        self.assertIsNone(node._localization_pose("map"))
        self.assertFalse(node._localization_available)
        self.assertEqual([], node._tf_buffer.calls)

    def test_zero_stamp_is_ignored_and_exact_age_boundary_is_inclusive(self):
        node = self.make_localization_node()
        node._odom_info_callback(self.odom_info(0))
        node._odom_info_callback(self.odom_info(9_500_000_000))
        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        node = self.make_localization_node()
        node._odom_info_callback(self.odom_info(9_499_999_999))
        self.assertIsNone(node._localization_pose("map"))
        self.assertFalse(node._localization_available)

    def test_newest_candidate_with_coverage_is_selected_without_cancelling_goal(self):
        node = self.make_localization_node()
        warnings = []
        node._logger.warning = warnings.append
        handle = FakeHandle()
        node._active_goal = {
            "token": 73, "candidate": self.candidate("active"),
            "identity": node._goal_identity(self.candidate("active"), "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        older_stamp = 13_898_000_000
        newer_stamp = 13_931_000_000
        node._now_ns = 14_000_000_000
        node._odom_info_callback(self.odom_info(older_stamp))
        node._odom_info_callback(self.odom_info(newer_stamp))
        node._tf_buffer.errors[("vo_odom", newer_stamp)] = (
            self.module.ExtrapolationException("newer transform is future")
        )

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        self.assertEqual([
            ("vo_odom", "base_link", newer_stamp),
            ("vo_odom", "base_link", older_stamp),
            ("map", "base_link", older_stamp),
        ], node._tf_buffer.calls)
        self.assertEqual(older_stamp, node._last_accepted_odom_stamp_ns)
        self.assertTrue(node._localization_available)
        self.assertIsNone(node._localization_invalidated_at_ns)
        self.assertEqual(0, handle.cancel_count)
        self.assertEqual([], warnings)

    def test_newest_complete_candidate_wins_with_out_of_order_deque(self):
        node = self.make_localization_node()
        stamps = (9_900_000_000, 9_950_000_000, 9_925_000_000)
        for stamp in stamps:
            node._odom_info_callback(self.odom_info(stamp))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        self.assertEqual(9_950_000_000, node._last_accepted_odom_stamp_ns)
        self.assertEqual([
            ("vo_odom", "base_link", 9_950_000_000),
            ("map", "base_link", 9_950_000_000),
        ], node._tf_buffer.calls)

    def test_map_tf_failure_for_newer_candidate_falls_back_without_mixing_stamps(self):
        node = self.make_localization_node()
        warnings = []
        node._logger.warning = warnings.append
        older_stamp = 9_900_000_000
        newer_stamp = 9_950_000_000
        node._odom_info_callback(self.odom_info(older_stamp))
        node._odom_info_callback(self.odom_info(newer_stamp))
        node._tf_buffer.errors[("map", newer_stamp)] = (
            self.module.ExtrapolationException("map transform unavailable")
        )

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        self.assertEqual([
            ("vo_odom", "base_link", newer_stamp),
            ("map", "base_link", newer_stamp),
            ("vo_odom", "base_link", older_stamp),
            ("map", "base_link", older_stamp),
        ], node._tf_buffer.calls)
        self.assertEqual(older_stamp, node._last_accepted_odom_stamp_ns)
        self.assertEqual([], warnings)

    def test_all_fresh_tf_failures_invalidate_once_using_newest_first_error(self):
        node = self.make_localization_node()
        node._localization_available = True
        warnings = []
        node._logger.warning = warnings.append
        handle = FakeHandle()
        candidate = self.candidate("active")
        node._active_goal = {
            "token": 74, "candidate": candidate,
            "identity": node._goal_identity(candidate, "map"),
            "epoch": 0, "is_retry": True, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        older_stamp = 9_900_000_000
        newer_stamp = 9_950_000_000
        node._odom_info_callback(self.odom_info(older_stamp))
        node._odom_info_callback(self.odom_info(newer_stamp))
        node._tf_buffer.errors[("map", newer_stamp)] = (
            self.module.ExtrapolationException("newest map failure")
        )
        node._tf_buffer.errors[("vo_odom", older_stamp)] = (
            self.module.ExtrapolationException("older odom failure")
        )
        node._failed_attempts = {"previous": 3}

        self.assertIsNone(node._localization_pose("map"))

        self.assertEqual([
            ("vo_odom", "base_link", newer_stamp),
            ("map", "base_link", newer_stamp),
            ("vo_odom", "base_link", older_stamp),
        ], node._tf_buffer.calls)
        self.assertLessEqual(len(node._tf_buffer.calls), 2 * 2)
        self.assertEqual(10_000_000_000, node._localization_invalidated_at_ns)
        self.assertEqual("map to base frame TF unavailable", node._localization_reason)
        self.assertTrue(node._active_goal["localization_invalidated"])
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual({"previous": 3}, node._failed_attempts)
        self.assertEqual(1, len(warnings))
        self.assertIn("selected_measurement_stamp_ns=9950000000", warnings[0])
        self.assertIn("target_frame='map'", warnings[0])
        self.assertIn("exception_message='newest map failure'", warnings[0])

    def test_stale_candidates_cannot_rescue_localization(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._odom_info_callback(self.odom_info(9_499_999_999))
        node._odom_info_callback(self.odom_info(9_400_000_000))

        self.assertIsNone(node._localization_pose("map"))

        self.assertEqual([], node._tf_buffer.calls)
        self.assertFalse(node._localization_available)
        self.assertEqual("OdomInfo is older than max_localization_age_sec",
                         node._localization_reason)

    def test_timestamp_below_accepted_floor_cannot_rescue_localization(self):
        node = self.make_localization_node()
        node._last_accepted_odom_stamp_ns = 9_900_000_000
        node._odom_info_callback(self.odom_info(9_899_999_999))

        self.assertIsNone(node._localization_pose("map"))

        self.assertEqual([], node._tf_buffer.calls)
        self.assertFalse(node._localization_available)

    def test_timestamp_equal_to_accepted_floor_remains_eligible(self):
        node = self.make_localization_node()
        node._last_accepted_odom_stamp_ns = 9_900_000_000
        node._odom_info_callback(self.odom_info(9_900_000_000))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        self.assertEqual(9_900_000_000, node._last_accepted_odom_stamp_ns)
        self.assertEqual([
            ("vo_odom", "base_link", 9_900_000_000),
            ("map", "base_link", 9_900_000_000),
        ], node._tf_buffer.calls)

    def test_each_expected_tf2_availability_exception_allows_fallback(self):
        expected_exceptions = (
            self.module.LookupException,
            self.module.ConnectivityException,
            self.module.ExtrapolationException,
            self.module.TimeoutException,
        )
        for exception_type in expected_exceptions:
            with self.subTest(exception_type=exception_type.__name__):
                node = self.make_localization_node()
                older_stamp = 9_900_000_000
                newer_stamp = 9_950_000_000
                node._odom_info_callback(self.odom_info(older_stamp))
                node._odom_info_callback(self.odom_info(newer_stamp))
                node._tf_buffer.errors[("vo_odom", newer_stamp)] = exception_type(
                    "availability failure"
                )

                self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

                self.assertEqual(older_stamp, node._last_accepted_odom_stamp_ns)
                self.assertEqual([
                    ("vo_odom", "base_link", newer_stamp),
                    ("vo_odom", "base_link", older_stamp),
                    ("map", "base_link", older_stamp),
                ], node._tf_buffer.calls)

    def test_unexpected_tf_exception_invalidates_without_trying_older_candidate(self):
        for target_frame, reason, expected_calls in (
            ("vo_odom", "vo_odom to base frame TF unavailable", 1),
            ("map", "map to base frame TF unavailable", 2),
        ):
            with self.subTest(target_frame=target_frame):
                node = self.make_localization_node()
                warnings = []
                node._logger.warning = warnings.append
                node._localization_available = True
                older_stamp = 9_900_000_000
                newer_stamp = 9_950_000_000
                node._odom_info_callback(self.odom_info(older_stamp))
                node._odom_info_callback(self.odom_info(newer_stamp))
                node._tf_buffer.errors[(target_frame, newer_stamp)] = RuntimeError(
                    "unexpected TF implementation failure"
                )

                self.assertIsNone(node._localization_pose("map"))

                self.assertEqual(expected_calls, len(node._tf_buffer.calls))
                self.assertEqual(10_000_000_000,
                                 node._localization_invalidated_at_ns)
                self.assertEqual(reason, node._localization_reason)
                self.assertEqual(1, len(warnings))
                self.assertIn("selected_measurement_stamp_ns=9950000000", warnings[0])
                self.assertIn("exception_class=RuntimeError", warnings[0])

    def test_lost_info_creates_strict_recovery_barrier(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._last_accepted_odom_stamp_ns = 9_900_000_000
        node._odom_info_callback(self.odom_info(11_000_000_000, lost=True))
        self.assertEqual(11_000_000_000, node._localization_invalidated_at_ns)
        node._now_ns = 11_400_000_000
        node._odom_info_callback(self.odom_info(11_000_000_000))
        self.assertEqual([], list(node._odom_info))
        node._odom_info_callback(self.odom_info(11_000_000_001))
        self.assertIsNotNone(node._localization_pose("map"))
        self.assertIsNone(node._localization_invalidated_at_ns)

    def test_lost_transition_uses_ros_now_and_duplicate_lost_can_raise_by_stamp(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._now_ns = 10_200_000_000
        node._odom_info_callback(self.odom_info(10_100_000_000, lost=True))
        self.assertEqual(10_200_000_000, node._localization_invalidated_at_ns)
        node._now_ns = 10_300_000_000
        node._odom_info_callback(self.odom_info(10_250_000_000, lost=True))
        self.assertEqual(10_250_000_000, node._localization_invalidated_at_ns)

    def test_repeated_localization_failure_does_not_move_recovery_boundary(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.missing.add("map")
        self.assertIsNone(node._localization_pose("map"))
        boundary = node._localization_invalidated_at_ns
        node._now_ns += 100_000_000
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(boundary, node._localization_invalidated_at_ns)

    def test_new_info_during_repeated_tf_failure_does_not_slide_boundary(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._last_accepted_odom_stamp_ns = 9_900_000_000
        node._odom_info_callback(self.odom_info(9_950_000_000))
        node._tf_buffer.missing.add("map")
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(10_000_000_000, node._localization_invalidated_at_ns)

        node._now_ns = 10_100_000_000
        node._odom_info_callback(self.odom_info(10_050_000_000))
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(10_000_000_000, node._localization_invalidated_at_ns)

        node._tf_buffer.missing.clear()
        node._odom_info_callback(self.odom_info(10_050_000_000))
        self.assertIsNotNone(node._localization_pose("map"))
        self.assertIsNone(node._localization_invalidated_at_ns)

    def test_older_info_cannot_replace_latest_accepted_timestamp_after_deque_rollover(self):
        node = self.make_localization_node()
        node._last_accepted_odom_stamp_ns = 9_990_000_000
        for stamp_ns in range(9_800_000_000, 9_900_000_000, 10_000_000):
            node._odom_info_callback(self.odom_info(stamp_ns))
        self.assertEqual(10, len(node._odom_info))
        self.assertIsNone(node._localization_pose("map"))
        self.assertFalse(node._localization_available)
        self.assertEqual([], node._tf_buffer.calls)

    def test_clock_regression_clears_old_samples_and_requires_new_axis_data(self):
        node = self.make_localization_node()
        node._odom_info_callback(self.odom_info(9_900_000_000))
        self.assertIsNotNone(node._localization_pose("map"))
        self.assertEqual(2, len(node._tf_buffer.calls))
        node._now_ns = 1_000_000_000
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual([], list(node._odom_info))
        self.assertEqual([], node._tf_buffer.calls)
        node._odom_info_callback(self.odom_info(1_000_000_000))
        self.assertIsNotNone(node._localization_pose("map"))

    def test_clock_regression_clears_cached_future_odom_info(self):
        node = self.make_localization_node()
        future_stamp = 10_500_000_000
        node._odom_info_callback(self.odom_info(future_stamp))
        self.assertEqual([future_stamp], [
            node._stamp_ns(message) for message in node._odom_info
        ])
        node._now_ns = 1_000_000_000
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual([], list(node._odom_info))
        node._odom_info_callback(self.odom_info(1_000_000_000))
        self.assertIsNotNone(node._localization_pose("map"))

    def test_localization_failure_cancels_goal_and_aborted_race_has_no_penalty(self):
        node = self.make_localization_node()
        candidate = self.candidate("moving")
        handle = FakeHandle()
        node._active_goal = {
            "token": 30, "candidate": candidate, "identity": node._goal_identity(candidate, "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": True, "cancel_reason": "search_disabled",
        }
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.missing.add("vo_odom")
        self.assertIsNone(node._localization_pose("map"))
        self.assertTrue(node._active_goal["localization_invalidated"])
        self.assertEqual(0, handle.cancel_count)
        node._goal_result(Future(types.SimpleNamespace(status=6)), 30)
        self.assertEqual({}, node._failed_attempts)

    def test_no_odom_info_on_startup_blocks_evaluation_and_dispatch(self):
        node = self.make_localization_node()
        with patch.object(self.module, "evaluate_search") as evaluate:
            node._search_tick()
        evaluate.assert_not_called()
        self.assertEqual([], node._action.sent)

    def test_either_missing_transform_cancels_active_goal_once(self):
        for target_frame in ("vo_odom", "map"):
            with self.subTest(target_frame=target_frame):
                node = self.make_localization_node()
                candidate = self.candidate("active")
                handle = FakeHandle()
                identity = node._goal_identity(candidate, "map")
                node._active_goal = {
                    "token": 70, "candidate": candidate, "identity": identity,
                    "epoch": 0, "is_retry": False, "handle": handle,
                    "cancel_requested": False, "cancel_reason": None,
                }
                node._odom_info_callback(self.odom_info(9_900_000_000))
                node._tf_buffer.missing.add(target_frame)
                self.assertIsNone(node._localization_pose("map"))
                self.assertTrue(node._active_goal["localization_invalidated"])
                self.assertEqual(1, handle.cancel_count)
                node._odom_info_callback(self.odom_info(10_000_000_000, lost=True))
                self.assertEqual(1, handle.cancel_count)

    def test_tf_lookup_failures_log_original_context_and_preserve_invalidation(self):
        for target_frame, reason, expected_call_count in (
            ("vo_odom", "vo_odom to base frame TF unavailable", 1),
            ("map", "map to base frame TF unavailable", 2),
        ):
            with self.subTest(target_frame=target_frame):
                node = self.make_localization_node()
                warnings = []
                node._logger.warning = warnings.append
                node._localization_available = True
                node._last_accepted_odom_stamp_ns = 9_900_000_000
                candidate = self.candidate("active")
                handle = FakeHandle()
                node._active_goal = {
                    "token": 71, "candidate": candidate,
                    "identity": node._goal_identity(candidate, "map"),
                    "epoch": 0, "is_retry": True, "handle": handle,
                    "cancel_requested": False, "cancel_reason": None,
                }
                node._failed_attempts = {"prior": 2}
                node._odom_info_callback(self.odom_info(9_900_000_000))
                original_error = RuntimeError("complete TF failure detail")
                node._tf_buffer.errors[target_frame] = original_error

                self.assertIsNone(node._localization_pose("map"))

                self.assertEqual(expected_call_count, len(node._tf_buffer.calls))
                self.assertEqual(10_000_000_000,
                                 node._localization_invalidated_at_ns)
                self.assertTrue(node._active_goal["localization_invalidated"])
                self.assertEqual(1, handle.cancel_count)
                self.assertEqual({"prior": 2}, node._failed_attempts)
                self.assertEqual(1, len(warnings))
                warning = warnings[0]
                self.assertIn(f"Search localization invalidated: {reason}", warning)
                self.assertIn("selected_measurement_stamp_ns=9900000000", warning)
                self.assertIn("requested_stamp_ns=9900000000", warning)
                self.assertIn("ros_now_ns=10000000000", warning)
                self.assertIn("age_ns=100000000", warning)
                self.assertIn("max_age_ns=500000000", warning)
                self.assertIn(f"target_frame='{target_frame}'", warning)
                self.assertIn("source_frame='base_link'", warning)
                self.assertIn("exception_class=RuntimeError", warning)
                self.assertIn("exception_message='complete TF failure detail'", warning)

    def test_tf_failure_warning_is_suppressed_for_repeated_same_reason(self):
        node = self.make_localization_node()
        warnings = []
        node._logger.warning = warnings.append
        node._localization_available = True
        handle = FakeHandle()
        node._active_goal = {
            "token": 72, "candidate": self.candidate("active"),
            "identity": node._goal_identity(self.candidate("active"), "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.missing.add("vo_odom")
        self.assertIsNone(node._localization_pose("map"))
        boundary = node._localization_invalidated_at_ns
        self.assertEqual(1, len(warnings))

        node._now_ns = 10_100_000_000
        node._odom_info_callback(self.odom_info(10_050_000_000))
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(boundary, node._localization_invalidated_at_ns)
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual(1, len(warnings))

    def test_successful_tf_lookups_emit_no_failure_diagnostic(self):
        node = self.make_localization_node()
        warnings = []
        node._logger.warning = warnings.append
        node._odom_info_callback(self.odom_info(9_900_000_000))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        self.assertEqual(2, len(node._tf_buffer.calls))
        self.assertEqual([], warnings)

    def test_recovery_before_terminal_result_waits_then_fresh_data_allows_next_goal(self):
        node = self.make_localization_node()
        candidate = self.candidate("active")
        node._odom_info_callback(self.odom_info(9_950_000_000))
        self.assertTrue(node._send_goal(candidate, "/tree.xml"))
        token = node._pending_goal["token"]
        handle = FakeHandle()
        node._goal_response(Future(handle), token)

        node._odom_info_callback(self.odom_info(10_000_000_000, lost=True))
        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(node._active_goal["localization_invalidated"])
        node._now_ns = 10_100_000_000
        node._odom_info_callback(self.odom_info(10_050_000_000))
        evaluation = self.result([candidate])
        with patch.object(self.module, "evaluate_search", return_value=evaluation):
            node._search_tick()
        self.assertEqual(1, len(node._action.sent))

        node._goal_result(Future(types.SimpleNamespace(status=5)), token)
        self.assertEqual({}, node._failed_attempts)
        node._now_ns = 10_120_000_000
        node._odom_info_callback(self.odom_info(10_110_000_000))
        with patch.object(self.module, "evaluate_search", return_value=evaluation):
            node._search_tick()
        self.assertEqual(2, len(node._action.sent))

    def test_pending_accept_after_localization_loss_is_cancelled_without_rejection_penalty(self):
        node = self.make_node()
        candidate = self.candidate("pending")
        node._send_goal(candidate, "/tree.xml")
        token = node._pending_goal["token"]
        node._invalidate_localization("test loss", 10_000_000_000)
        self.assertTrue(node._pending_goal["localization_invalidated"])
        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(1, handle.cancel_count)
        node._goal_result(Future(types.SimpleNamespace(status=5)), token)
        self.assertEqual({}, node._failed_attempts)

    def test_rejection_after_localization_loss_has_no_attempt_penalty(self):
        node = self.make_node()
        node._send_goal(self.candidate("rejected"), "/tree.xml")
        token = node._pending_goal["token"]
        node._invalidate_localization("test loss", 10_000_000_000)
        node._goal_response(Future(FakeHandle(accepted=False)), token)
        self.assertEqual({}, node._failed_attempts)

    def test_second_validation_after_evaluation_can_block_goal_send(self):
        node = self.make_node()
        pose = (1.0, 2.0, 0.5)
        validations = iter((pose, None))
        node._localization_pose = lambda frame: next(validations)
        with patch.object(self.module, "evaluate_search", return_value=self.result([
            self.candidate("candidate")
        ])) as evaluate:
            node._search_tick()
        evaluate.assert_called_once()
        self.assertEqual([], node._action.sent)

    def test_send_goal_revalidates_after_marker_work_advances_clock(self):
        node = self.make_localization_node()
        node._odom_info_callback(self.odom_info(10_000_000_000))
        node._publish_markers = lambda *args: setattr(node, "_now_ns", 10_600_000_000)
        with patch.object(self.module, "evaluate_search", return_value=self.result([
            self.candidate("candidate")
        ])):
            node._search_tick()
        self.assertEqual([], node._action.sent)
        self.assertIsNone(node._pending_goal)
        self.assertEqual(10_600_000_000, node._localization_invalidated_at_ns)

    def test_search_evaluates_once_for_goal_and_markers_from_nav2_map(self):
        node = self.make_node()
        first = self.candidate("best", 0.5, 3.0)
        second = self.candidate("second", 0.2, 4.0)
        result = self.result([first, second])
        seen = []
        self.module._candidate_markers = lambda candidates, selected, *args, **kwargs: (
            seen.append((candidates, selected)) or []
        )
        with patch.object(self.module, "evaluate_search", return_value=result) as evaluate:
            node._search_tick()
        evaluate.assert_called_once()
        self.assertEqual(1, len(node._action.sent))
        goal = node._action.sent[0][0]
        self.assertEqual("map", goal.pose.header.frame_id)
        self.assertEqual("sim-stamp", goal.pose.header.stamp)
        self.assertEqual(3.0, goal.pose.pose.position.x)
        self.assertEqual("/share/search_navigate_to_pose.xml", goal.behavior_tree)
        self.assertIs(seen[0][0][0], first)
        self.assertIs(seen[0][1], first)

    def test_rejected_candidate_gets_one_retry_then_next_ranked_candidate(self):
        node = self.make_node()
        first, second = self.candidate("first", 0.5), self.candidate("second", 0.2)
        result = self.result([first, second])

        for expected in ("first", "first"):
            candidate = node._next_ranked_candidate(result)
            self.assertEqual(expected, candidate["id"])
            node._send_goal(candidate, "/tree.xml")
            token = node._pending_goal["token"]
            node._goal_response(Future(FakeHandle(accepted=False)), token)

        first_identity = node._goal_identity(first, "map")
        self.assertEqual(2, node._failed_attempts[first_identity])
        self.assertEqual("second", node._next_ranked_candidate(result)["id"])

    def test_aborted_goal_gets_one_retry_and_success_exhausts_candidate(self):
        node = self.make_node()
        first, second = self.candidate("first", 0.5), self.candidate("second", 0.2)
        result = self.result([first, second])
        for expected in ("first", "first"):
            candidate = node._next_ranked_candidate(result)
            self.assertEqual(expected, candidate["id"])
            handle = FakeHandle()
            node._active_goal = {
                "token": len(node._failed_attempts) + 1,
                "candidate": candidate, "handle": handle,
                "cancel_requested": False, "cancel_reason": None,
            }
            token = node._active_goal["token"]
            node._goal_result(Future(types.SimpleNamespace(status=6)), token)
        first_identity = node._goal_identity(first, "map")
        self.assertEqual(2, node._failed_attempts[first_identity])
        self.assertEqual("second", node._next_ranked_candidate(result)["id"])

        node._active_goal = {
            "token": 10, "candidate": second, "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        node._goal_result(Future(types.SimpleNamespace(status=4)), 10)
        self.assertIn(node._goal_identity(second, "map"), node._completed_candidates)
        self.assertIsNone(node._next_ranked_candidate(result))

    def test_intentional_cancel_does_not_consume_failure_retry(self):
        node = self.make_node()
        candidate = self.candidate("current")
        handle = FakeHandle()
        node._active_goal = {
            "token": 7, "candidate": candidate, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._cancel_active_goal("higher_scoring_candidate")
        self.assertEqual(1, handle.cancel_count)
        node._goal_result(Future(types.SimpleNamespace(status=5)), 7)
        self.assertNotIn(node._goal_identity(candidate, "map"), node._failed_attempts)

    def test_aborted_result_after_cancel_request_consumes_failure_attempt(self):
        node = self.make_node()
        candidate = self.candidate("aborted")
        handle = FakeHandle()
        node._active_goal = {
            "token": 8, "candidate": candidate, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._cancel_active_goal("search_disabled")
        node._goal_result(Future(types.SimpleNamespace(status=6)), 8)
        self.assertEqual(1, node._failed_attempts[node._goal_identity(candidate, "map")])

    def test_repeated_enable_preserves_retry_and_completion_history(self):
        node = self.make_node()
        retry = node._goal_identity(self.candidate("retry"), "map")
        done = node._goal_identity(self.candidate("done"), "map")
        node._failed_attempts[retry] = 1
        node._completed_candidates.add(done)
        response = types.SimpleNamespace(success=None, message="")

        node._set_search(types.SimpleNamespace(data=True), response)

        self.assertTrue(response.success)
        self.assertEqual(1, node._failed_attempts[retry])
        self.assertIn(done, node._completed_candidates)

    def test_false_to_true_resets_failure_and_completion_history(self):
        node = self.make_node()
        retry = node._goal_identity(self.candidate("retry"), "map")
        done = node._goal_identity(self.candidate("done"), "map")
        replacement = node._goal_identity(self.candidate("replacement"), "map")
        node._enabled = False
        node._failed_attempts[retry] = 1
        node._completed_candidates.add(done)
        node._retry_target = retry
        node._replacement_target = replacement
        node._challenger_identity = replacement
        node._challenger_target = retry
        node._challenger_epoch = node._search_epoch
        node._challenger_count = 1

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=True), response)

        self.assertTrue(response.success)
        self.assertEqual({}, node._failed_attempts)
        self.assertEqual(set(), node._completed_candidates)
        self.assertIsNone(node._retry_target)
        self.assertIsNone(node._replacement_target)
        self.assertIsNone(node._challenger_identity)
        self.assertEqual(0, node._challenger_count)

    def test_better_candidate_preempts_only_after_two_margin_wins(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        challenger = self.candidate("challenger", 0.31)
        handle = FakeHandle()
        node._active_goal = {
            "token": 5, "candidate": current, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([current, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(0, handle.cancel_count)
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("higher_scoring_candidate", node._active_goal["cancel_reason"])

    def test_better_candidate_below_margin_does_not_preempt(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        challenger = self.candidate("challenger", 0.29)
        handle = FakeHandle()
        node._active_goal = {
            "token": 6, "candidate": current, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([current, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}
        node._consider_goal_switch(evaluation, by_id)
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(0, handle.cancel_count)

    def test_one_margin_win_does_not_interrupt_retry_even_when_rank_changes(self):
        node = self.make_node()
        first = self.candidate("first", 0.20)
        challenger = self.candidate("challenger", 0.31)
        identity = node._goal_identity(first, "map")
        node._retry_target = identity
        node._failed_attempts[identity] = 1

        node._consider_goal_switch(
            self.result([first, challenger]),
            {item["id"]: item for item in (first, challenger)},
        )

        self.assertEqual(1, node._challenger_count)
        self.assertEqual("first", node._next_ranked_candidate(
            self.result([challenger, first])
        )["id"])
        self.assertIsNone(node._replacement_target)

    def test_exact_margin_qualifies_only_after_same_physical_challenger_twice(self):
        node = self.make_node()
        first = self.candidate("first", 0.20)
        challenger = self.candidate("challenger", 0.30)
        handle = FakeHandle()
        identity = node._goal_identity(first, "map")
        node._retry_target = identity
        node._failed_attempts[identity] = 1
        node._active_goal = {
            "token": 51, "candidate": first, "identity": identity,
            "epoch": node._search_epoch, "is_retry": True, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([first, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(0, handle.cancel_count)
        node._consider_goal_switch(evaluation, by_id)

        self.assertEqual(1, handle.cancel_count)
        self.assertEqual(node._goal_identity(challenger, "map"), node._replacement_target)
        self.assertIsNone(node._retry_target)
        self.assertIs(node._active_goal["handle"], handle)

        node._goal_result(Future(types.SimpleNamespace(status=5)), 51)
        self.assertIsNone(node._active_goal)
        self.assertEqual("challenger", node._next_ranked_candidate(evaluation)["id"])

    def test_changed_challenger_identity_resets_margin_streak(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        first_challenger = self.candidate("challenger", 0.31, 3.0)
        changed_challenger = self.candidate("challenger", 0.31, 4.0)
        handle = FakeHandle()
        node._active_goal = {
            "token": 52, "candidate": current, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._consider_goal_switch(
            self.result([current, first_challenger]),
            {"current": current, "challenger": first_challenger},
        )
        node._consider_goal_switch(
            self.result([current, changed_challenger]),
            {"current": current, "challenger": changed_challenger},
        )

        self.assertEqual(0, handle.cancel_count)
        self.assertEqual(1, node._challenger_count)
        self.assertIsNone(node._replacement_target)

    def test_changed_target_identity_resets_margin_streak(self):
        node = self.make_node()
        owned = self.candidate("current", 0.20, 1.0)
        refreshed = self.candidate("current", 0.20, 9.0)
        challenger = self.candidate("challenger", 0.31, 3.0)
        node._active_goal = {
            "token": 56, "candidate": owned, "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        node._consider_goal_switch(
            self.result([owned, challenger]),
            {"current": owned, "challenger": challenger},
        )
        self.assertEqual(1, node._challenger_count)

        node._consider_goal_switch(
            self.result([refreshed, challenger]),
            {"current": refreshed, "challenger": challenger},
        )

        self.assertEqual(0, node._challenger_count)
        self.assertIsNone(node._replacement_target)

    def test_pending_qualified_replacement_waits_for_acceptance_and_terminal(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        challenger = self.candidate("challenger", 0.30, 3.0)
        node._send_goal(current, "/tree.xml")
        token = node._pending_goal["token"]
        evaluation = self.result([current, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)
        node._consider_goal_switch(evaluation, by_id)

        self.assertEqual("higher_scoring_candidate", node._pending_goal["cancel_reason"])
        self.assertEqual(node._goal_identity(challenger, "map"), node._replacement_target)
        self.assertIsNotNone(node._pending_goal)
        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        node._goal_result(Future(types.SimpleNamespace(status=5)), token)

        self.assertIsNone(node._active_goal)
        self.assertEqual("challenger", node._next_ranked_candidate(evaluation)["id"])

    def test_qualified_replacement_does_not_restore_old_retry_on_abort(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        challenger = self.candidate("challenger", 0.31, 3.0)
        identity = node._goal_identity(current, "map")
        node._retry_target = identity
        node._failed_attempts[identity] = 1
        handle = FakeHandle()
        node._active_goal = {
            "token": 57, "candidate": current, "identity": identity,
            "epoch": node._search_epoch, "is_retry": True, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([current, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}
        node._consider_goal_switch(evaluation, by_id)
        node._consider_goal_switch(evaluation, by_id)
        replacement = node._replacement_target

        node._goal_result(Future(types.SimpleNamespace(status=6)), 57)

        self.assertIsNone(node._retry_target)
        self.assertEqual(replacement, node._replacement_target)
        self.assertEqual(1, node._failed_attempts[identity])
        self.assertEqual("challenger", node._next_ranked_candidate(evaluation)["id"])

    def test_challenger_streak_survives_first_failure_and_applies_to_retry(self):
        node = self.make_node()
        first = self.candidate("first", 0.20)
        challenger = self.candidate("challenger", 0.31)
        identity = node._goal_identity(first, "map")
        handle = FakeHandle()
        node._active_goal = {
            "token": 53, "candidate": first, "identity": identity,
            "epoch": node._search_epoch, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([first, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)
        node._goal_result(Future(types.SimpleNamespace(status=6)), 53)
        self.assertEqual(1, node._challenger_count)
        self.assertEqual(identity, node._retry_target)

        retry_handle = FakeHandle()
        node._active_goal = {
            "token": 54, "candidate": first, "identity": identity,
            "epoch": node._search_epoch, "is_retry": True, "handle": retry_handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._consider_goal_switch(evaluation, by_id)

        self.assertEqual(1, retry_handle.cancel_count)
        self.assertEqual(node._goal_identity(challenger, "map"), node._replacement_target)

    def test_retry_disappearing_before_dispatch_is_abandoned_for_current_candidate(self):
        node = self.make_node()
        first, second = self.candidate("first"), self.candidate("second", 0.2, 2.0)
        node._retry_target = node._goal_identity(first, "map")
        node._failed_attempts[node._retry_target] = 1

        self.assertEqual("second", node._next_ranked_candidate(self.result([second]))["id"])
        self.assertIsNone(node._retry_target)

    def test_invalid_retry_pending_is_cancelled_and_not_retried_after_rejection(self):
        node = self.make_node()
        first, second = self.candidate("first"), self.candidate("second", 0.2, 2.0)
        first_identity = node._goal_identity(first, "map")
        node._retry_target = first_identity
        node._failed_attempts[first_identity] = 1
        node._send_goal(first, "/tree.xml")
        node._pending_goal["is_retry"] = True

        node._consider_goal_switch(self.result([second]), {"second": second})
        self.assertEqual("retry_invalidated", node._pending_goal["cancel_reason"])
        token = node._pending_goal["token"]
        node._goal_response(Future(FakeHandle(accepted=False)), token)

        self.assertIsNone(node._pending_goal)
        self.assertIsNone(node._retry_target)
        self.assertEqual(1, node._failed_attempts[first_identity])
        self.assertEqual("second", node._next_ranked_candidate(self.result([second]))["id"])

    def test_invalid_retry_active_waits_for_terminal_and_does_not_recreate_retry(self):
        node = self.make_node()
        first, second = self.candidate("first"), self.candidate("second", 0.2, 2.0)
        first_identity = node._goal_identity(first, "map")
        node._retry_target = first_identity
        node._failed_attempts[first_identity] = 1
        handle = FakeHandle()
        node._active_goal = {
            "token": 55, "candidate": first, "identity": first_identity,
            "epoch": node._search_epoch, "is_retry": True, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }

        node._consider_goal_switch(self.result([second]), {"second": second})

        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        self.assertEqual("retry_invalidated", node._active_goal["cancel_reason"])
        node._goal_result(Future(types.SimpleNamespace(status=6)), 55)
        self.assertIsNone(node._active_goal)
        self.assertIsNone(node._retry_target)
        self.assertEqual(1, node._failed_attempts[first_identity])
        self.assertEqual("second", node._next_ranked_candidate(self.result([second]))["id"])

    def test_frontier_change_invalidates_missing_pending_retry_before_challenger_check(self):
        node = self.make_node()
        first, second = self.candidate("first"), self.candidate("second", 0.2, 2.0)
        identity = node._goal_identity(first, "map")
        node._retry_target = identity
        node._failed_attempts[identity] = 1
        node._send_goal(first, "/tree.xml")
        token = node._pending_goal["token"]
        result = self.result([second])
        result["frontier_cells"] = [[4, 5]]

        with patch.object(self.module, "evaluate_search", return_value=result):
            node._search_tick()

        self.assertEqual("retry_invalidated", node._pending_goal["cancel_reason"])
        self.assertIsNone(node._retry_target)
        node._consider_goal_switch(
            self.result([first, self.candidate("challenger", 0.8, 3.0)]),
            {"first": first, "challenger": self.candidate("challenger", 0.8, 3.0)},
        )
        self.assertIsNone(node._replacement_target)

        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        node._goal_result(Future(types.SimpleNamespace(status=5)), token)
        self.assertIsNone(node._active_goal)
        self.assertNotIn(identity, node._failed_attempts)

    def test_frontier_change_rejected_pending_retry_releases_without_failure(self):
        node = self.make_node()
        first, second = self.candidate("first"), self.candidate("second", 0.2, 2.0)
        identity = node._goal_identity(first, "map")
        node._retry_target = identity
        node._failed_attempts[identity] = 1
        node._send_goal(first, "/tree.xml")
        token = node._pending_goal["token"]
        result = self.result([second])
        result["frontier_cells"] = [[4, 5]]

        with patch.object(self.module, "evaluate_search", return_value=result):
            node._search_tick()
        node._goal_response(Future(FakeHandle(accepted=False)), token)

        self.assertIsNone(node._pending_goal)
        self.assertIsNone(node._retry_target)
        self.assertNotIn(identity, node._failed_attempts)

    def test_frontier_change_invalidates_missing_active_retry_but_not_first_attempt(self):
        second = self.candidate("second", 0.2, 2.0)
        result = self.result([second])
        result["frontier_cells"] = [[4, 5]]
        for is_retry in (True, False):
            with self.subTest(is_retry=is_retry):
                node = self.make_node()
                first = self.candidate("first")
                identity = node._goal_identity(first, "map")
                if is_retry:
                    node._retry_target = identity
                    node._failed_attempts[identity] = 1
                handle = FakeHandle()
                node._active_goal = {
                    "token": 70 + int(is_retry), "candidate": first,
                    "identity": identity, "epoch": node._search_epoch,
                    "is_retry": is_retry, "handle": handle,
                    "cancel_requested": False, "cancel_reason": None,
                }

                with patch.object(self.module, "evaluate_search", return_value=result):
                    node._search_tick()

                if is_retry:
                    self.assertEqual(1, handle.cancel_count)
                    self.assertEqual("retry_invalidated", node._active_goal["cancel_reason"])
                    self.assertIsNone(node._retry_target)
                    self.assertIsNotNone(node._active_goal)
                    node._goal_result(
                        Future(types.SimpleNamespace(status=5)), 71
                    )
                    self.assertIsNone(node._active_goal)
                else:
                    self.assertEqual(0, handle.cancel_count)
                    self.assertIsNotNone(node._active_goal)

    def test_frontier_change_invalidates_pending_retry_when_pose_or_frame_changes(self):
        original = self.candidate("same-id", x=1.0)
        for frame_id, candidate in (
            ("map", self.candidate("same-id", x=2.0)),
            ("odom", self.candidate("same-id", x=1.0)),
        ):
            with self.subTest(frame_id=frame_id, pose=candidate["pose_world"]):
                node = self.make_node()
                original_identity = node._goal_identity(original, "map")
                node._retry_target = original_identity
                node._failed_attempts[original_identity] = 1
                node._send_goal(original, "/tree.xml")
                if frame_id != "map":
                    node._map.header.frame_id = frame_id
                result = self.result([candidate])
                result["frontier_cells"] = [[4, 5]]

                with patch.object(self.module, "evaluate_search", return_value=result):
                    node._search_tick()

                self.assertEqual("retry_invalidated", node._pending_goal["cancel_reason"])
                self.assertIsNone(node._retry_target)

    def test_pending_cancel_latches_replacement_against_later_challengers(self):
        node = self.make_node()
        current = self.candidate("A", 0.20)
        replacement = self.candidate("B", 0.31, 3.0)
        later = self.candidate("C", 0.50, 4.0)
        node._send_goal(current, "/tree.xml")
        token = node._pending_goal["token"]
        first_evaluation = self.result([current, replacement])
        by_id = {item["id"]: item for item in first_evaluation["candidates"]}

        node._consider_goal_switch(first_evaluation, by_id)
        node._consider_goal_switch(first_evaluation, by_id)
        replacement_identity = node._goal_identity(replacement, "map")
        self.assertEqual(replacement_identity, node._replacement_target)
        self.assertEqual("higher_scoring_candidate", node._pending_goal["cancel_reason"])

        node._consider_goal_switch(
            self.result([current, later, replacement]),
            {"A": current, "B": replacement, "C": later},
        )
        self.assertEqual(replacement_identity, node._replacement_target)
        self.assertEqual(0, node._challenger_count)

        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        node._goal_result(Future(types.SimpleNamespace(status=5)), token)
        latest = self.result([later, replacement])
        with patch.object(self.module, "evaluate_search", return_value=latest):
            node._search_tick()

        self.assertEqual("B", node._pending_goal["candidate"]["id"])
        self.assertIsNone(node._replacement_target)

    def test_rejected_pending_goal_keeps_eligible_replacement_but_drops_ineligible_one(self):
        for replacement_present in (True, False):
            with self.subTest(replacement_present=replacement_present):
                node = self.make_node()
                current = self.candidate("A", 0.20)
                replacement = self.candidate("B", 0.31, 3.0)
                leader = self.candidate("C", 0.60, 4.0)
                node._send_goal(current, "/tree.xml")
                token = node._pending_goal["token"]
                evaluation = self.result([current, replacement])
                by_id = {item["id"]: item for item in evaluation["candidates"]}
                node._consider_goal_switch(evaluation, by_id)
                node._consider_goal_switch(evaluation, by_id)
                replacement_identity = node._replacement_target
                node._consider_goal_switch(
                    self.result([current, leader, replacement]),
                    {"A": current, "B": replacement, "C": leader},
                )
                self.assertEqual(replacement_identity, node._replacement_target)
                node._goal_response(Future(FakeHandle(accepted=False)), token)

                candidates = [leader, replacement] if replacement_present else [leader]
                with patch.object(self.module, "evaluate_search",
                                  return_value=self.result(candidates)):
                    node._search_tick()

                expected = "B" if replacement_present else "C"
                self.assertEqual(expected, node._pending_goal["candidate"]["id"])

    def test_changed_pose_or_frame_invalidates_deferred_retry_identity(self):
        node = self.make_node()
        first = self.candidate("same-id", x=1.0)
        node._retry_target = node._goal_identity(first, "map")
        node._failed_attempts[node._retry_target] = 1
        changed_pose = self.candidate("same-id", x=2.0)

        self.assertEqual(changed_pose, node._next_ranked_candidate(self.result([changed_pose])))
        self.assertNotEqual(node._retry_target, node._goal_identity(changed_pose, "map"))

        node._retry_target = node._goal_identity(first, "map")
        node._map.header.frame_id = "odom"
        self.assertEqual(changed_pose, node._next_ranked_candidate(self.result([changed_pose])))
        self.assertIsNone(node._retry_target)

    def test_old_epoch_terminal_outcomes_release_owner_without_changing_new_history(self):
        for status in (4, 6, 5):
            with self.subTest(status=status):
                node = self.make_node()
                candidate = self.candidate("old")
                identity = node._goal_identity(candidate, "map")
                node._active_goal = {
                    "token": 60 + status, "candidate": candidate,
                    "identity": identity, "epoch": 0, "is_retry": False,
                    "handle": FakeHandle(), "cancel_requested": False,
                    "cancel_reason": "search_disabled",
                }
                node._search_epoch = 1
                new_identity = node._goal_identity(self.candidate("new"), "map")
                node._failed_attempts[new_identity] = 1
                node._completed_candidates.add(new_identity)

                node._goal_result(
                    Future(types.SimpleNamespace(status=status)), 60 + status
                )

                self.assertIsNone(node._active_goal)
                self.assertEqual({new_identity: 1}, node._failed_attempts)
                self.assertEqual({new_identity}, node._completed_candidates)
                self.assertIsNone(node._retry_target)

    def test_old_epoch_rejection_releases_pending_without_changing_new_history(self):
        node = self.make_node()
        candidate = self.candidate("old")
        node._send_goal(candidate, "/tree.xml")
        token = node._pending_goal["token"]
        node._search_epoch = 1
        new_identity = node._goal_identity(self.candidate("new"), "map")
        node._failed_attempts[new_identity] = 1
        node._goal_response(Future(FakeHandle(accepted=False)), token)

        self.assertIsNone(node._pending_goal)
        self.assertEqual({new_identity: 1}, node._failed_attempts)
        self.assertIsNone(node._retry_target)

    def test_old_epoch_accepted_goal_terminal_outcomes_preserve_new_history(self):
        for status in (4, 6, 5):
            with self.subTest(status=status):
                node = self.make_node()
                candidate = self.candidate("old")
                node._send_goal(candidate, "/tree.xml")
                token = node._pending_goal["token"]
                node._search_epoch = 1
                new_identity = node._goal_identity(self.candidate("new"), "map")
                node._failed_attempts[new_identity] = 1
                node._completed_candidates.add(new_identity)
                if status == 5:
                    node._pending_goal["cancel_reason"] = "search_disabled"
                handle = FakeHandle()

                node._goal_response(Future(handle), token)
                node._goal_result(Future(types.SimpleNamespace(status=status)), token)

                self.assertIsNone(node._active_goal)
                self.assertEqual({new_identity: 1}, node._failed_attempts)
                self.assertEqual({new_identity}, node._completed_candidates)
                self.assertIsNone(node._retry_target)

    def test_latched_replacement_is_revalidated_before_dispatch(self):
        node = self.make_node()
        original = self.candidate("replacement", 0.5, 3.0)
        changed = self.candidate("replacement", 0.6, 4.0)
        node._replacement_target = node._goal_identity(original, "map")

        self.assertEqual(changed, node._next_ranked_candidate(self.result([changed])))
        self.assertIsNone(node._replacement_target)

    def test_marker_candidate_selection_and_server_unavailable_preserve_replacement_latch(self):
        node = self.make_node()
        replacement = self.candidate("replacement", 0.4, 4.0)
        first_leader = self.candidate("leader-a", 0.9, 1.0)
        second_leader = self.candidate("leader-b", 0.8, 2.0)
        node._replacement_target = node._goal_identity(replacement, "map")
        node._action.ready = False
        seen = []
        self.module._candidate_markers = lambda candidates, selected, *args, **kwargs: (
            seen.append(selected) or []
        )

        with patch.object(self.module, "evaluate_search", side_effect=[
            self.result([first_leader, replacement]),
            self.result([second_leader, replacement]),
        ]):
            node._search_tick()
            node._search_tick()

        self.assertEqual([replacement, replacement], seen)
        self.assertEqual(node._goal_identity(replacement, "map"), node._replacement_target)
        self.assertEqual([], node._action.sent)

        node._action.ready = True
        with patch.object(self.module, "evaluate_search", return_value=self.result([
            first_leader, second_leader, replacement
        ])):
            node._search_tick()

        self.assertEqual(1, len(node._action.sent))
        self.assertEqual("replacement", node._pending_goal["candidate"]["id"])
        self.assertIsNone(node._replacement_target)

    def test_same_tick_dispatch_uses_latched_replacement_over_rank_leader(self):
        node = self.make_node()
        replacement = self.candidate("replacement", 0.4, 4.0)
        leader = self.candidate("leader", 0.9, 1.0)
        node._replacement_target = node._goal_identity(replacement, "map")

        with patch.object(self.module, "evaluate_search", return_value=self.result([
            leader, replacement
        ])):
            node._search_tick()

        self.assertEqual(1, len(node._action.sent))
        self.assertEqual("replacement", node._pending_goal["candidate"]["id"])
        self.assertIsNone(node._replacement_target)

    def test_goal_accepted_after_disable_is_cancelled_and_waited_out(self):
        node = self.make_node()
        candidate = self.candidate("pending")
        node._send_goal(candidate, "/tree.xml")
        token = node._pending_goal["token"]
        node._enabled = False
        node._search_epoch += 1
        node._pending_goal["cancel_reason"] = "search_disabled"
        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        node._goal_result(Future(types.SimpleNamespace(status=5)), token)
        self.assertIsNone(node._active_goal)
        self.assertFalse(node._failed_attempts)

    def test_goal_response_exception_keeps_unknown_pending_owner_without_retry(self):
        node = self.make_node()
        candidate = self.candidate("uncertain")
        node._send_goal(candidate, "/tree.xml")
        token = node._pending_goal["token"]
        node._goal_response(Future(error=RuntimeError("response lost")), token)

        self.assertIsNotNone(node._pending_goal)
        self.assertTrue(node._pending_goal["outcome_unknown"])
        self.assertEqual({}, node._failed_attempts)
        with patch.object(self.module, "evaluate_search", return_value=self.result([candidate])):
            node._search_tick()
        self.assertEqual(1, len(node._action.sent))

    def test_result_exception_keeps_owner_blocks_replacement_and_cancels_once(self):
        node = self.make_node()
        candidate = self.candidate("uncertain")
        handle = FakeHandle()
        node._active_goal = {
            "token": 21, "candidate": candidate, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._goal_result(Future(error=RuntimeError("result lost")), 21)
        self.assertIs(node._active_goal["handle"], handle)
        self.assertTrue(node._active_goal["outcome_unknown"])
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual({}, node._failed_attempts)

        with patch.object(self.module, "evaluate_search", return_value=self.result([candidate])):
            node._search_tick()
        self.assertEqual([], node._action.sent)
        node._goal_result(Future(error=RuntimeError("result still lost")), 21)
        self.assertEqual(1, handle.cancel_count)

    def test_cancel_response_does_not_release_owner_and_terminal_result_does(self):
        node = self.make_node()
        handle = FakeHandle()
        node._active_goal = {
            "token": 22, "candidate": self.candidate("current"), "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._cancel_active_goal("result_unknown")
        handle.cancel_future.result()
        self.assertIsNotNone(node._active_goal)
        node._goal_result(Future(types.SimpleNamespace(status=5)), 22)
        self.assertIsNone(node._active_goal)

    def test_stale_goal_callbacks_cannot_change_a_different_owner(self):
        node = self.make_node()
        new_candidate = self.candidate("new")
        node._pending_goal = {
            "token": 31, "candidate": new_candidate, "cancel_reason": None,
        }
        node._goal_response(Future(FakeHandle(accepted=False)), 30)
        self.assertEqual(31, node._pending_goal["token"])
        self.assertEqual({}, node._failed_attempts)

        handle = FakeHandle()
        node._active_goal = {
            "token": 32, "candidate": new_candidate, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._goal_result(Future(types.SimpleNamespace(status=4)), 30)
        self.assertEqual(32, node._active_goal["token"])
        self.assertFalse(node._completed_candidates)

    def test_unknown_result_status_keeps_owner_and_requests_cancel_once(self):
        node = self.make_node()
        handle = FakeHandle()
        node._active_goal = {
            "token": 23, "candidate": self.candidate("current"), "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._goal_result(Future(types.SimpleNamespace(status=2)), 23)
        self.assertIsNotNone(node._active_goal)
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual({}, node._failed_attempts)

    def test_disable_waits_for_terminal_result_and_clears_selected_marker_afterward(self):
        node = self.make_node()
        handle = FakeHandle()
        node._active_goal = {
            "token": 11, "candidate": self.candidate("active"), "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        marker_inputs = []
        self.module._candidate_markers = lambda candidates, selected, *args, **kwargs: (
            marker_inputs.append((candidates, selected)) or []
        )
        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)

        self.assertFalse(response.success)
        self.assertIn("waiting", response.message.lower())
        self.assertFalse(node._enabled)
        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        self.assertIs(node._active_goal["candidate"], marker_inputs[-1][1])

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertFalse(response.success)
        self.assertEqual(1, handle.cancel_count)

        node._goal_result(Future(types.SimpleNamespace(status=5)), 11)
        self.assertIsNone(node._active_goal)
        self.assertEqual(0, len(marker_inputs[-1][0]))
        self.assertIsNone(marker_inputs[-1][1])

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertTrue(response.success)

    def test_explicit_rejection_while_disabled_clears_selected_marker(self):
        node = self.make_node()
        candidate = self.candidate("rejected")
        node._send_goal(candidate, "/tree.xml")
        token = node._pending_goal["token"]
        node._enabled = False
        node._pending_goal["cancel_reason"] = "search_disabled"
        marker_inputs = []
        self.module._candidate_markers = lambda candidates, selected, *args, **kwargs: (
            marker_inputs.append((candidates, selected)) or []
        )

        node._goal_response(Future(FakeHandle(accepted=False)), token)

        self.assertIsNone(node._pending_goal)
        self.assertEqual(0, len(marker_inputs[-1][0]))
        self.assertIsNone(marker_inputs[-1][1])

    def test_missing_frontier_cells_preserves_retry_and_completion_history(self):
        node = self.make_node()
        retry = node._goal_identity(self.candidate("retry"), "map")
        done = node._goal_identity(self.candidate("done"), "map")
        node._failed_attempts[retry] = 1
        node._completed_candidates.add(done)
        node._active_goal = {
            "token": 24, "candidate": self.candidate("active"), "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        result = self.result([self.candidate("active")])
        result.pop("frontier_cells")
        with patch.object(self.module, "evaluate_search", return_value=result):
            node._search_tick()
        self.assertEqual(1, node._failed_attempts[retry])
        self.assertIn(done, node._completed_candidates)

    def test_explicit_empty_frontier_cells_resets_retry_and_completion_history(self):
        node = self.make_node()
        node._failed_attempts["retry"] = 1
        node._completed_candidates.add("done")
        node._active_goal = {
            "token": 25, "candidate": self.candidate("active"), "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        result = self.result([self.candidate("active")])
        result["frontier_cells"] = []
        with patch.object(self.module, "evaluate_search", return_value=result):
            node._search_tick()
        self.assertEqual({}, node._failed_attempts)
        self.assertEqual(set(), node._completed_candidates)

    def test_failure_budget_survives_frontier_changes_until_second_failure(self):
        node = self.make_node()
        first = self.candidate("first", 0.8)
        second = self.candidate("second", 0.2, 2.0)
        identity = node._goal_identity(first, "map")
        node._failed_attempts[identity] = 1
        node._retry_target = identity
        first_result = self.result([first, second])
        first_result["frontier_cells"] = [[3, 4]]

        with patch.object(self.module, "evaluate_search", return_value=first_result):
            node._search_tick()

        self.assertEqual("first", node._pending_goal["candidate"]["id"])
        self.assertEqual(1, node._failed_attempts[identity])
        token = node._pending_goal["token"]
        node._goal_response(Future(FakeHandle(accepted=False)), token)
        self.assertEqual(2, node._failed_attempts[identity])

        second_result = self.result([first, second])
        second_result["frontier_cells"] = [[5, 6]]
        node._action.ready = False
        with patch.object(self.module, "evaluate_search", return_value=second_result):
            node._search_tick()

        self.assertEqual(2, node._failed_attempts[identity])
        self.assertEqual("second", node._next_ranked_candidate(second_result)["id"])

    def test_frontier_change_prunes_lost_failure_and_preserves_eligible_replacement(self):
        node = self.make_node()
        lost = self.candidate("lost", 0.7)
        replacement = self.candidate("replacement", 0.2, 2.0)
        lost_identity = node._goal_identity(lost, "map")
        replacement_identity = node._goal_identity(replacement, "map")
        node._failed_attempts[lost_identity] = 1
        node._retry_target = lost_identity
        node._replacement_target = replacement_identity
        node._action.ready = False
        result = self.result([replacement])
        result["frontier_cells"] = [[3, 4]]

        with patch.object(self.module, "evaluate_search", return_value=result):
            node._search_tick()

        self.assertNotIn(lost_identity, node._failed_attempts)
        self.assertIsNone(node._retry_target)
        self.assertEqual(replacement_identity, node._replacement_target)

    def test_retry_identity_loss_without_frontier_change_cancels_owned_retry(self):
        node = self.make_node()
        original = self.candidate("same-id", 0.5, 1.0)
        changed = self.candidate("same-id", 0.6, 2.0)
        identity = node._goal_identity(original, "map")
        handle = FakeHandle()
        node._retry_target = identity
        node._failed_attempts[identity] = 1
        node._active_goal = {
            "token": 88, "candidate": original, "identity": identity,
            "epoch": node._search_epoch, "is_retry": True, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        result = self.result([changed])

        with patch.object(self.module, "evaluate_search", return_value=result):
            node._search_tick()

        self.assertIsNone(node._retry_target)
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("retry_invalidated", node._active_goal["cancel_reason"])
        self.assertIs(node._active_goal["candidate"], original)

    def test_disabled_search_does_not_evaluate_or_send_a_goal(self):
        node = self.make_node()
        node._enabled = False
        with patch.object(self.module, "evaluate_search") as evaluate:
            node._search_tick()
        evaluate.assert_not_called()
        self.assertEqual([], node._action.sent)

    def test_selected_goal_ring_uses_stored_pose_without_adding_to_current_candidates(self):
        node = self.make_node()
        selected = self.candidate("in-flight", -1.0)
        node._active_goal = {
            "token": 9, "candidate": selected, "handle": FakeHandle(),
            "cancel_requested": True, "cancel_reason": "higher_scoring_candidate",
        }
        captured = []
        self.module._candidate_markers = lambda candidates, chosen, *args, **kwargs: (
            captured.append((candidates, chosen)) or []
        )
        node._publish_markers(self.result([self.candidate("new", 0.5)]), selected)
        self.assertNotIn(selected, captured[0][0])
        self.assertIs(selected, captured[0][1])

    def test_selected_marker_uses_stored_pose_when_evaluation_refreshes_same_id(self):
        node = self.make_node()
        stored = self.candidate("same-id", -1.0, 1.0)
        refreshed = self.candidate("same-id", 0.9, 9.0)
        node._active_goal = {
            "token": 10, "candidate": stored, "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        captured = []
        self.module._candidate_markers = lambda candidates, selected, *args, **kwargs: (
            captured.append((candidates, selected)) or []
        )

        with patch.object(self.module, "evaluate_search", return_value=self.result([refreshed])):
            node._search_tick()

        self.assertIs(refreshed, captured[0][0][0])
        self.assertIs(stored, captured[0][1])
        self.assertEqual(1.0, captured[0][1]["pose_world"][0])

    def test_empty_evaluation_without_selection_deletes_previous_candidate_markers(self):
        node = self.make_node()
        node._published_marker_keys = {("frontier_candidates", 2)}
        deleted = types.SimpleNamespace(ns="frontier_candidates", id=2, action=2)
        self.module._candidate_markers = lambda *args, **kwargs: []
        self.module._stale_marker_deletions = lambda previous, current, *args: (
            [deleted] if previous == {("frontier_candidates", 2)} and not current else []
        )

        node._publish_markers({"candidates": []}, None)

        self.assertEqual(set(), node._published_marker_keys)
        self.assertEqual([deleted], node._markers.messages[-1].markers)


if __name__ == "__main__":
    unittest.main()
