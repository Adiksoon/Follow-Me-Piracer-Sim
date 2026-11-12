import importlib
import ast
import json
import math
import sys
import types
import unittest
from collections import deque
from concurrent.futures import Future as WorkerFuture
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


def own_active_goal(node, candidate, token, handle):
    goal = {
        "token": token, "epoch": node._search_epoch, "candidate": candidate,
        "frame_id": "map", "handle": handle, "state": "active",
        "cancel_requested": False, "cancel_reason": None,
        "localization_invalidated": False,
    }
    node._owned_goals[token] = goal
    node._current_goal_token = token
    node._active_goal_token = token
    node._goal_active = True
    return goal


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class CompletedEvaluationExecutor:
    """Keep existing decision tests deterministic; real concurrency has its own tests."""

    def submit(self, callback, *args):
        future = WorkerFuture()
        try:
            future.set_result(callback(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


class SearchNodeCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy = types.ModuleType("rclpy")
        rclpy.__path__ = []
        rclpy.init = lambda **kwargs: None
        rclpy.spin = lambda node: lifecycle_events.append("rclpy_spin")
        rclpy.shutdown = lambda: None
        node_module = types.ModuleType("rclpy.node")
        executor_module = types.ModuleType("rclpy.executors")
        callback_groups_module = types.ModuleType("rclpy.callback_groups")
        lifecycle_events = []

        class FakeMutuallyExclusiveCallbackGroup:
            pass

        class FakeGuardCondition:
            def __init__(self, callback, callback_group):
                self.callback = callback
                self.callback_group = callback_group
                self.trigger_count = 0

            def trigger(self):
                self.trigger_count += 1

        callback_groups_module.MutuallyExclusiveCallbackGroup = FakeMutuallyExclusiveCallbackGroup
        node_module.FakeGuardCondition = FakeGuardCondition

        class FakeMultiThreadedExecutor:
            instances = []
            spin_error = None

            def __init__(self, *, num_threads):
                self.num_threads = num_threads
                self.nodes = []
                self.add_count = 0
                self.__class__.instances.append(self)
                lifecycle_events.append("executor_create")

            def add_node(self, node):
                self.nodes.append(node)
                self.add_count += 1
                lifecycle_events.append("executor_add_node")

            def spin(self):
                lifecycle_events.append("executor_spin")
                if self.__class__.spin_error is not None:
                    raise self.__class__.spin_error

            def shutdown(self):
                lifecycle_events.append("executor_shutdown")
                return True

        executor_module.MultiThreadedExecutor = FakeMultiThreadedExecutor
        executor_module.SingleThreadedExecutor = object
        executor_module.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
        rclpy.executors = executor_module
        rclpy.callback_groups = callback_groups_module
        rclpy.lifecycle_events = lifecycle_events

        def fake_init(**kwargs):
            lifecycle_events.append("rclpy_init")

        def fake_shutdown():
            lifecycle_events.append("rclpy_shutdown")

        rclpy.init = fake_init
        rclpy.shutdown = fake_shutdown
        utilities_module = types.ModuleType("rclpy.utilities")
        utilities_module.try_shutdown = fake_shutdown
        parameter_module = types.ModuleType("rclpy.parameter")
        parameter_module.Parameter = object

        class FakeNode:
            def __init__(self, name):
                self.context = object()
                self.default_callback_group = object()
                self.parameters = {"use_sim_time": True}
                self.subscriptions = []
                self.services = []
                self.timers = []
                self.guards = []
                lifecycle_events.append("node_create")

            def get_namespace(self):
                return "/"

            def destroy_node(self):
                lifecycle_events.append("node_destroy")

            def declare_parameter(self, name, value):
                self.parameters[name] = value

            def get_parameter(self, name):
                return types.SimpleNamespace(value=self.parameters[name])

            def create_subscription(self, message_type, topic, callback, qos,
                                   **kwargs):
                record = types.SimpleNamespace(
                    message_type=message_type, topic=topic, callback=callback,
                    qos=qos,
                    callback_group=kwargs.get(
                        "callback_group", self.default_callback_group
                    ),
                )
                self.subscriptions.append(record)
                return record

            def create_publisher(self, *args):
                return FakePublisher()

            def create_guard_condition(self, callback, *, callback_group):
                guard = node_module.FakeGuardCondition(callback, callback_group)
                self.guards.append(guard)
                return guard

            def create_service(self, *args, **kwargs):
                record = types.SimpleNamespace(
                    callback_group=kwargs.get(
                        "callback_group", self.default_callback_group
                    )
                )
                self.services.append(record)
                return record

            def create_timer(self, *args, **kwargs):
                record = types.SimpleNamespace(
                    callback_group=kwargs.get(
                        "callback_group", self.default_callback_group
                    )
                )
                self.timers.append(record)
                return record

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
        class FakeActionClient:
            def __init__(self, *args, **kwargs):
                self.callback_group = kwargs.get(
                    "callback_group", args[0].default_callback_group
                )

        action_module.ActionClient = FakeActionClient
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
        class FakeTFBuffer:
            def __init__(self, *args, **kwargs):
                self.set_calls = []
                self.set_error = None

            def set_transform(self, transform, authority):
                self.set_calls.append((transform, authority))
                if self.set_error is not None:
                    raise self.set_error

        tf_module.Buffer = FakeTFBuffer
        class FakeTransformListener:
            def callback(self, data):
                self.callback_calls += 1
                for transform in data.transforms:
                    self.buffer.set_transform(transform, "default_authority")

            def __init__(self, buffer, *args, **kwargs):
                self.buffer = buffer
                self.spin_thread = kwargs.get("spin_thread", False)
                self.group = "reentrant"
                self.callback_calls = 0

        tf_module.TransformListener = FakeTransformListener
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
            "rclpy.utilities": utilities_module,
            "rclpy.parameter": parameter_module,
            "rclpy.executors": executor_module,
            "rclpy.callback_groups": callback_groups_module,
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
        cls.tf_exceptions = tf_module
        package = importlib.import_module("piracer_search")
        with (
            patch.dict(sys.modules, modules),
            patch.object(package, "search_node", create=True),
            patch.object(package, "tf_timing", create=True),
        ):
            # Real-TF tests may have imported these during pytest collection.
            # Reload under the fakes, then restore both caches and package aliases.
            sys.modules.pop("piracer_search.search_node", None)
            sys.modules.pop("piracer_search.tf_timing", None)
            cls.module = importlib.import_module("piracer_search.search_node")
            cls.tf_module = importlib.import_module("piracer_search.tf_timing")
        class FakeTfReceiver:
            def __init__(self, buffer, error_reporter, **kwargs):
                self.node = object()
                self.listener = FakeTransformListener(buffer, self.node)
                self.context = kwargs["context"]
                self.namespace = kwargs["namespace"]
                self.use_sim_time = kwargs["use_sim_time"]
                self.received_odom_info = []
                self.odom_info_wakeup = kwargs["odom_info_wakeup"]

            def receive_odom_info(self, message):
                self.received_odom_info.append(message)

            def start(self):
                lifecycle_events.append("tf_receiver_start")

            def close(self):
                lifecycle_events.append("tf_receiver_close")

        cls.module.SearchTfReceiver = FakeTfReceiver
        cls.marker_helpers = marker_helpers

    def test_main_uses_two_worker_executor_and_cleans_up_after_normal_spin(self):
        # The executor class is imported into the production module; its event
        # record is retained here even after the fixture restores sys.modules.
        events = self.module.rclpy.lifecycle_events
        events.clear()
        self.module.MultiThreadedExecutor.instances.clear()
        self.module.MultiThreadedExecutor.spin_error = None

        self.module.main()

        self.assertEqual(
            ["rclpy_init", "node_create", "executor_create",
             "executor_add_node", "tf_receiver_start", "executor_spin", "executor_shutdown",
             "tf_receiver_close", "node_destroy", "rclpy_shutdown"],
            events,
        )
        self.assertEqual(1, len(self.module.MultiThreadedExecutor.instances))
        executor = self.module.MultiThreadedExecutor.instances[0]
        self.assertEqual(2, executor.num_threads)
        self.assertEqual(1, executor.add_count)
        self.assertEqual(1, len(executor.nodes))

    def test_main_cleans_up_after_exception_from_spin(self):
        events = self.module.rclpy.lifecycle_events
        events.clear()
        self.module.MultiThreadedExecutor.instances.clear()
        error = RuntimeError("spin failed")
        self.module.MultiThreadedExecutor.spin_error = error

        with self.assertRaisesRegex(RuntimeError, "spin failed"):
            self.module.main()

        self.assertEqual(
            ["rclpy_init", "node_create", "executor_create",
             "executor_add_node", "tf_receiver_start", "executor_spin", "executor_shutdown",
             "tf_receiver_close", "node_destroy", "rclpy_shutdown"],
            events,
        )
        self.assertEqual(1, len(self.module.MultiThreadedExecutor.instances))
        executor = self.module.MultiThreadedExecutor.instances[0]
        self.assertEqual(2, executor.num_threads)
        self.assertEqual(1, executor.add_count)
        self.assertEqual(1, len(executor.nodes))
        self.module.MultiThreadedExecutor.spin_error = None

    def test_search_entities_share_explicit_group_separate_from_clock_default(self):
        node = self.module.SearchNode()
        group = node._search_callback_group
        self.assertIsInstance(group, self.module.MutuallyExclusiveCallbackGroup)
        self.assertIsNot(group, node.default_callback_group)
        self.assertTrue(all(
            subscription.callback_group is group
            for subscription in node.subscriptions
            if subscription is not node._odom_info_subscription
        ))
        self.assertTrue(all(
            service.callback_group is group
            for service in node.services
        ))
        self.assertIs(group, node.timers[0].callback_group)
        self.assertIs(group, node._action.callback_group)
        self.assertEqual("reentrant", node._tf_listener.group)
        self.assertFalse(node._tf_listener.spin_thread)
        self.assertIs(node.context, node._tf_receiver.context)
        self.assertTrue(node._tf_receiver.use_sim_time)
        self.assertEqual("/", node._tf_receiver.namespace)
        self.assertIsInstance(node._odom_info_callback_group,
                              self.module.MutuallyExclusiveCallbackGroup)
        self.assertIsNot(group, node._odom_info_callback_group)
        self.assertIsNot(node.default_callback_group, node._odom_info_callback_group)
        self.assertIs(node._odom_info_callback_group,
                      node._odom_info_subscription.callback_group)
        self.assertIs(group, node._odom_info_wakeup.callback_group)
        self.assertIs(node._odom_info_wakeup, node._tf_receiver.odom_info_wakeup)

    def test_receiver_errors_are_logged_without_diagnostic_mode(self):
        node = self.make_node()
        warnings = []
        node._logger.warning = warnings.append

        node._receiver_error("odom_info_wakeup", RuntimeError("trigger failed"))

        self.assertEqual(1, len(warnings))
        self.assertIn("odom_info_wakeup", warnings[0])
        self.assertIn("RuntimeError: trigger failed", warnings[0])

    def make_node(self):
        node = object.__new__(self.module.SearchNode)
        node._enabled = True
        node._search_epoch = 0
        node._localization_generation = 0
        node._evaluation_executor = CompletedEvaluationExecutor()
        node._evaluation_future = None
        node._evaluation_context = None
        node._map = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="map"),
            info=types.SimpleNamespace(resolution=0.1),
        )
        node._goal_sequence = 0
        node._owned_goals = {}
        node._current_goal_token = None
        node._active_goal_token = None
        node._goal_active = False
        node._blacklisted_frontiers = []
        node._progress_timeout_ns = 30_000_000_000
        node._progress_goal_token = None
        node._progress_min_distance = None
        node._progress_started_ns = None
        node._published_marker_keys = set()
        node._odom_info = deque(maxlen=10)
        node._last_processed_odom_info_sequence = 0
        node._localization_available = False
        node._localization_invalidated_at_ns = None
        node._last_accepted_odom_stamp_ns = None
        node._last_accepted_full_tf_stamp_ns = None
        node._tf_future_lead_ns = 100_000_000
        node._last_ros_time_ns = None
        node._clock_epoch = 0
        node._localization_reason = None
        node._now_ns = 10_000_000_000
        node._tf_buffer = type("TFBuffer", (), {"clear": lambda self: None})()
        node._action = FakeAction()
        node._markers = FakePublisher()
        node._frontier_pose = FakePublisher()
        node._logger = type(
            "Logger", (), {"warning": lambda *args: None, "info": lambda *args: None,
                           "debug": lambda *args: None}
        )()
        node.get_logger = lambda: node._logger
        params = {
            "behavior_tree": "/share/search_navigate_to_pose.xml",
            "base_frame": "base_link",
            "max_localization_age_sec": 0.5,
            "odom_info_topic": "/visual_vo/odom_info_lite",
            "tf_future_lead_sec": 0.1,
            "progress_timeout": 30.0,
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

    def make_localization_node(self, max_age_sec=0.5):
        node = self.make_node()
        get_parameter = node.get_parameter
        node.get_parameter = lambda name: types.SimpleNamespace(
            value=max_age_sec
        ) if name == "max_localization_age_sec" else get_parameter(name)
        del node._localization_pose

        class TFBuffer:
            def __init__(self):
                self.calls = []
                self.missing = set()
                self.errors = {}
                self.stamps = {}
                self.timed_stamps = {}
                self.stamp_sequences = {}
                self.sequence_indexes = {}

            def lookup_transform(self, target, source, stamp):
                self.calls.append((target, source, stamp.nanoseconds))
                error = self.errors.get((target, stamp.nanoseconds))
                if error is None:
                    error = self.errors.get(target)
                if error is not None:
                    raise error
                if target == "map" and "vo_odom" in self.errors:
                    raise self.errors["vo_odom"]
                if target in self.missing or (target == "map" and "vo_odom" in self.missing):
                    raise self.expected_exception("vo_odom" if "vo_odom" in self.missing else target)
                index = self.sequence_indexes.get(target, 0)
                sequence = self.stamp_sequences.get(target)
                if sequence is not None:
                    stamp_ns = sequence[min(index, len(sequence) - 1)]
                    self.sequence_indexes[target] = index + 1
                else:
                    stamp_ns = self.stamps.get(target, node._now_ns - 100_000_000)
                stamp_ns = self.timed_stamps.get((target, stamp.nanoseconds), stamp_ns)
                if target == "map" and "vo_odom" in self.stamps:
                    stamp_ns = min(stamp_ns, self.stamps["vo_odom"])
                return types.SimpleNamespace(
                    header=types.SimpleNamespace(stamp=types.SimpleNamespace(
                        sec=stamp_ns // 1_000_000_000,
                        nanosec=stamp_ns % 1_000_000_000,
                    )),
                    transform=types.SimpleNamespace(
                        translation=types.SimpleNamespace(x=1.0, y=2.0),
                        rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )

            def clear(self):
                self.calls.clear()
                self.stamps.clear()
                self.timed_stamps.clear()
                self.stamp_sequences.clear()
                self.sequence_indexes.clear()


        node._tf_buffer = TFBuffer()
        node._tf_buffer.expected_exception = self.tf_exceptions.ExtrapolationException
        return node



    @staticmethod
    def odom_info(stamp_ns, lost=False):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(
                    sec=stamp_ns // 1_000_000_000,
                    nanosec=stamp_ns % 1_000_000_000,
                ),
                frame_id="",
            ),
            lost=lost,
        )

    @staticmethod
    def candidate(candidate_id, score=0.0, x=1.0):
        frontier = [x + 2.0 * (sum(map(ord, candidate_id)) % 31), 2.0]
        return {"id": candidate_id, "score": score, "pose_world": [x, 2.0, 0.5],
                "cluster_id": candidate_id,
                "min_distance": 2.0,
                "source_frontier_world": frontier,
                "source_frontiers_world": [frontier],
                "frontiers_world": [frontier]}

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
        self.assertEqual("/visual_vo/odom_info_lite",
                         node.get_parameter("odom_info_topic").value)
        self.assertEqual("/visual_vo/odom_info_lite",
                         node._odom_info_subscription.topic)
        self.assertIs(self.module.OdomInfo,
                      node._odom_info_subscription.message_type)
        self.assertIs(self.module.qos_profile_sensor_data,
                      node._odom_info_subscription.qos)
        self.assertIn(node._odom_info_subscription, node.subscriptions)
        self.assertFalse(hasattr(node._tf_receiver, "odom_info_subscription"))
        self.assertEqual(100_000_000, node._tf_future_lead_ns)

    def test_tf_future_lead_rejects_nonfinite_or_negative_configuration(self):
        base_node = self.module.SearchNode.__mro__[1]
        original_declare = base_node.declare_parameter
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=invalid):
                def declare_with_invalid_value(fake_node, name, default):
                    original_declare(fake_node, name, default)
                    if name == "tf_future_lead_sec":
                        fake_node.parameters[name] = invalid

                with patch.object(base_node, "declare_parameter",
                                  declare_with_invalid_value):
                    with self.assertRaisesRegex(
                            ValueError, "finite and non-negative"):
                        self.module.SearchNode()

    def test_odom_info_reception_only_delegates_without_validation(self):
        node = self.module.SearchNode()
        message = self.odom_info(9_950_000_000, lost=True)
        with patch.object(node, "_odom_info_callback") as validate:
            node._odom_info_subscription.callback(message)
        self.assertEqual([message], node._tf_receiver.received_odom_info)
        validate.assert_not_called()
        self.assertEqual([], list(node._odom_info))

    def test_odom_subscription_failure_closes_already_created_tf_helper(self):
        events = self.module.rclpy.lifecycle_events
        events.clear()
        original = self.module.Node.create_subscription

        def create(node, message_type, topic, callback, qos, **kwargs):
            if topic == "/visual_vo/odom_info_lite":
                raise RuntimeError("subscription failed")
            return original(node, message_type, topic, callback, qos, **kwargs)

        with patch.object(self.module.Node, "create_subscription", create):
            with self.assertRaisesRegex(RuntimeError, "subscription failed"):
                self.module.SearchNode()
        self.assertEqual(1, events.count("tf_receiver_close"))

    def test_future_sample_does_not_hide_latest_nonfuture_sample(self):
        node = self.make_localization_node()
        old_stamp = 9_950_000_000
        future_stamp = 10_054_000_000
        node._odom_info_callback(self.odom_info(old_stamp))
        node._odom_info_callback(self.odom_info(future_stamp))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual([("map", "base_link", 0)], node._tf_buffer.calls)
        node._now_ns = 10_100_000_000
        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual(0, node._tf_buffer.calls[-1][2])

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

    def test_latest_common_time_future_and_age_boundaries(self):
        for offset, accepted in ((41_000_000, True), (100_000_000, True),
                             (100_000_001, False), (-1_000_000_000, True),
                             (-1_000_000_001, False)):
            with self.subTest(offset=offset):
                node = self.make_localization_node(max_age_sec=1.0)
                node._odom_info_callback(self.odom_info(9_900_000_000))
                node._tf_buffer.stamps['map'] = node._now_ns + offset
                handle = FakeHandle()
                own_active_goal(node, self.candidate("active"), 1, handle)
                result = node._localization_pose('map')
                self.assertEqual(accepted, result is not None)
                self.assertEqual(0 if accepted else 1, handle.cancel_count)
                self.assertEqual([('map', 'base_link', 0)], node._tf_buffer.calls)

    def test_run17_shaped_fresh_info_and_older_fresh_tf_are_accepted(self):
        node = self.make_localization_node()
        node._now_ns = 7_000_000_000
        node._tf_buffer.stamps.update({
            "vo_odom": 6_521_000_000,
            "map": 6_600_000_000,
        })
        node._odom_info_callback(self.odom_info(6_888_000_000))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual([("map", "base_link", 0)], node._tf_buffer.calls)
        self.assertEqual(6_888_000_000, node._last_accepted_odom_stamp_ns)
        self.assertEqual(6_521_000_000, node._last_accepted_full_tf_stamp_ns)

    def test_full_tf_exact_age_boundary_is_inclusive(self):
        node = self.make_localization_node()
        node._tf_buffer.stamps['map'] = 9_500_000_000
        node._odom_info_callback(self.odom_info(9_900_000_000))
        self.assertIsNotNone(node._localization_pose('map'))
        self.assertEqual(9_500_000_000, node._last_accepted_full_tf_stamp_ns)

    def test_full_tf_over_age_limit_rejects_without_floor_updates(self):
        node = self.make_localization_node()
        node._tf_buffer.stamps['map'] = 9_499_999_999
        node._odom_info_callback(self.odom_info(9_900_000_000))
        self.assertIsNone(node._localization_pose('map'))
        self.assertIsNone(node._last_accepted_odom_stamp_ns)
        self.assertIsNone(node._last_accepted_full_tf_stamp_ns)

    def test_one_second_limit_accepts_run18_shaped_730ms_tf_and_exact_boundary(self):
        node = self.make_localization_node(max_age_sec=1.0)
        node._tf_buffer.stamps.update({"vo_odom": 9_270_000_000,
                                       "map": 9_270_000_000})
        node._odom_info_callback(self.odom_info(9_837_000_000))
        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        node = self.make_localization_node()
        node._tf_buffer.stamps.update({"vo_odom": 9_270_000_000,
                                       "map": 9_270_000_000})
        node._odom_info_callback(self.odom_info(9_837_000_000))
        self.assertIsNone(node._localization_pose("map"))
        self.assertIn("older than max_localization_age_sec",
                      node._localization_reason)

        node = self.make_localization_node(max_age_sec=1.0)
        node._tf_buffer.stamps.update({"vo_odom": 9_000_000_000,
                                       "map": 9_000_000_000})
        node._odom_info_callback(self.odom_info(9_900_000_000))
        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

    def test_one_second_limit_rejects_tf_1001ms_old_in_both_roles(self):
        for target in ("vo_odom", "map"):
            with self.subTest(target=target):
                node = self.make_localization_node(max_age_sec=1.0)
                node._tf_buffer.stamps.update({"vo_odom": 9_500_000_000,
                                               "map": 9_500_000_000})
                node._tf_buffer.stamps[target] = 8_999_000_000
                node._odom_info_callback(self.odom_info(9_900_000_000))
                self.assertIsNone(node._localization_pose("map"))
                self.assertIn("older than max_localization_age_sec",
                              node._localization_reason)

    def test_one_second_limit_does_not_delay_lost_info_cancellation(self):
        node = self.make_localization_node(max_age_sec=1.0)
        node._localization_available = True
        node._tf_buffer.stamps.update({"vo_odom": 9_950_000_000,
                                       "map": 9_950_000_000})
        handle = FakeHandle()
        candidate = self.candidate("active")
        goal = own_active_goal(node, candidate, 91, handle)
        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))
        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(goal["localization_invalidated"])
        self.assertEqual([], node._tf_buffer.calls)

    def test_full_tf_zero_future_and_regressing_stamps_reject(self):
        for stamp_ns, floor, reason in ((0, None, 'stamp is zero'),
                                       (10_100_000_001, None, 'stamp is in the future'),
                                       (9_899_999_999, 9_900_000_000, 'timestamp regressed')):
            with self.subTest(stamp_ns=stamp_ns):
                node = self.make_localization_node()
                node._tf_buffer.stamps['map'] = stamp_ns
                node._last_accepted_full_tf_stamp_ns = floor
                node._odom_info_callback(self.odom_info(9_900_000_000))
                self.assertIsNone(node._localization_pose('map'))
                self.assertIn(reason, node._localization_reason)
                self.assertIsNone(node._last_accepted_odom_stamp_ns)

    def test_tf_availability_failure_does_not_try_older_info(self):
        node = self.make_localization_node()
        for stamp in (9_900_000_000, 9_950_000_000):
            node._odom_info_callback(self.odom_info(stamp))
        node._tf_buffer.errors["vo_odom"] = self.tf_exceptions.ExtrapolationException(
            "latest native transform unavailable"
        )

        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual([("map", "base_link", 0)], node._tf_buffer.calls)
        self.assertIsNone(node._last_accepted_odom_stamp_ns)
        self.assertIsNone(node._last_accepted_full_tf_stamp_ns)


    def test_tf_stamps_at_barrier_reject_independently_then_newer_data_recovers(self):
        for stream in ("info", "native", "full"):
            with self.subTest(stream=stream):
                node = self.make_localization_node()
                node._now_ns = 10_500_000_000
                node._localization_invalidated_at_ns = 10_000_000_000
                node._tf_buffer.stamps.update({
                    "vo_odom": 10_100_000_000,
                    "map": 10_100_000_000,
                })
                info_stamp = 10_100_000_000
                if stream == "info":
                    info_stamp = 10_000_000_000
                else:
                    node._tf_buffer.stamps[
                        "vo_odom" if stream == "native" else "map"
                    ] = 10_000_000_000
                node._odom_info_callback(self.odom_info(info_stamp))

                self.assertIsNone(node._localization_pose("map"))
                self.assertFalse(node._localization_available)
                self.assertIsNone(node._last_accepted_odom_stamp_ns)
                self.assertIsNone(node._last_accepted_full_tf_stamp_ns)

                node._odom_info_callback(self.odom_info(10_200_000_000))
                node._tf_buffer.stamps.update({
                    "vo_odom": 10_200_000_000,
                    "map": 10_200_000_000,
                })
                self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
                self.assertIsNone(node._localization_invalidated_at_ns)

    def test_successful_validation_commits_all_floors_and_later_failure_commits_none(self):
        node = self.make_localization_node()
        node._last_accepted_odom_stamp_ns = 9_800_000_000
        node._last_accepted_full_tf_stamp_ns = 9_800_000_000
        node._tf_buffer.stamps.update({
            "vo_odom": 9_950_000_000,
            "map": 9_950_000_000,
        })
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.stamps.update({
            "vo_odom": 9_950_000_000,
            "map": 9_940_000_000,
        })

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual((9_900_000_000, 9_940_000_000), (
            node._last_accepted_odom_stamp_ns,
            node._last_accepted_full_tf_stamp_ns,
        ))

        node._now_ns = 10_100_000_000
        node._odom_info_callback(self.odom_info(10_000_000_000))
        node._tf_buffer.stamps.update({
            "vo_odom": 10_050_000_000,
            "map": 10_050_000_000,
        })
        node._tf_buffer.errors["map"] = self.tf_exceptions.ExtrapolationException(
            "full path unavailable"
        )
        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual((9_900_000_000, 9_940_000_000), (
            node._last_accepted_odom_stamp_ns,
            node._last_accepted_full_tf_stamp_ns,
        ))

    def test_clock_regression_during_tf_lookups_rejects_captured_values(self):
        node = self.make_localization_node()
        node._last_accepted_full_tf_stamp_ns = 9_800_000_000
        node._odom_info_callback(self.odom_info(9_900_000_000))
        clock_values = iter((10_000_000_000, 9_900_000_000))
        node.get_clock = lambda: types.SimpleNamespace(
            clock_type="ros",
            now=lambda: types.SimpleNamespace(
                nanoseconds=next(clock_values), to_msg=lambda: "sim-stamp"
            ),
        )

        self.assertIsNone(node._localization_pose("map"))
        self.assertIsNone(node._last_accepted_odom_stamp_ns)
        self.assertIsNone(node._last_accepted_full_tf_stamp_ns)
        self.assertEqual([], list(node._odom_info))
        self.assertEqual({}, node._tf_buffer.stamps)
        self.assertFalse(node._localization_available)

    def test_info_expiring_after_lookup_invalidates_localization(self):
        node = self.make_localization_node()
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._localization_available = True
        clock_values = iter((10_000_000_000, 10_500_000_001, 10_500_000_001))
        node.get_clock = lambda: types.SimpleNamespace(
            clock_type="ros",
            now=lambda: types.SimpleNamespace(
                nanoseconds=next(clock_values), to_msg=lambda: "sim-stamp"
            ),
        )

        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(1, len(node._tf_buffer.calls))
        self.assertTrue(node._localization_reason.startswith("OdomInfo"))

    def test_positive_info_still_rejects_stale_future_and_regressing_samples(self):
        for stamp_ns, accepted_floor in (
            (10_054_000_000, None),
            (9_499_999_999, None),
            (9_900_000_000, 9_950_000_000),
        ):
            with self.subTest(stamp_ns=stamp_ns, accepted_floor=accepted_floor):
                node = self.make_localization_node()
                if accepted_floor is not None:
                    node._last_accepted_odom_stamp_ns = accepted_floor
                node._odom_info_callback(self.odom_info(stamp_ns))
                self.assertIsNone(node._localization_pose("map"))
                self.assertEqual([], node._tf_buffer.calls)

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
        node._last_accepted_full_tf_stamp_ns = 9_900_000_000
        node._tf_buffer.stamps.update({"vo_odom": 9_900_000_000,
                                       "map": 9_900_000_000})
        node._odom_info_callback(self.odom_info(9_900_000_000))

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))

        self.assertEqual(9_900_000_000, node._last_accepted_odom_stamp_ns)
        self.assertEqual([("map", "base_link", 0)], node._tf_buffer.calls)

    def test_each_expected_tf2_availability_exception_invalidates(self):
        expected_exceptions = (
            self.tf_exceptions.LookupException,
            self.tf_exceptions.ConnectivityException,
            self.tf_exceptions.ExtrapolationException,
            self.tf_exceptions.TimeoutException,
        )
        for target, expected_calls in (
            ("vo_odom", [("map", "base_link", 0)]),
            ("map", [("map", "base_link", 0)]),
        ):
            for exception_type in expected_exceptions:
                with self.subTest(target=target,
                                  exception_type=exception_type.__name__):
                    node = self.make_localization_node()
                    node._odom_info_callback(self.odom_info(9_950_000_000))
                    node._tf_buffer.errors[target] = exception_type(
                        "availability failure"
                    )

                    self.assertIsNone(node._localization_pose("map"))
                    self.assertEqual(expected_calls, node._tf_buffer.calls)
                    self.assertIsNone(node._last_accepted_odom_stamp_ns)

    def test_unexpected_tf_exception_invalidates_without_trying_older_candidate(self):
        for target_frame, reason, expected_calls in (
            ("vo_odom", "full TF unavailable", 1),
            ("map", "full TF unavailable", 1),
        ):
            with self.subTest(target_frame=target_frame):
                node = self.make_localization_node()
                warnings = []
                node._logger.warning = warnings.append
                node._localization_available = True
                node._odom_info_callback(self.odom_info(9_950_000_000))
                node._tf_buffer.errors[target_frame] = RuntimeError(
                    "unexpected TF implementation failure"
                )

                self.assertIsNone(node._localization_pose("map"))

                self.assertEqual(expected_calls, len(node._tf_buffer.calls))
                self.assertEqual(10_000_000_000,
                                 node._localization_invalidated_at_ns)
                self.assertEqual(reason, node._localization_reason)
                self.assertEqual(1, len(warnings))
                self.assertIn("selected_info_stamp_ns=9950000000", warnings[0])
                self.assertIn("requested_stamp_ns=0", warnings[0])
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

    def test_lost_info_cancels_goal_immediately_with_fresh_cached_tf(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._tf_buffer.stamps.update({
            "vo_odom": 9_950_000_000,
            "map": 9_950_000_000,
        })
        handle = FakeHandle()
        candidate = self.candidate("active")
        goal = own_active_goal(node, candidate, 91, handle)

        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))

        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(goal["localization_invalidated"])
        self.assertEqual(9_950_000_000, node._tf_buffer.stamps["vo_odom"])
        self.assertEqual(9_950_000_000, node._tf_buffer.stamps["map"])
        self.assertEqual([], node._tf_buffer.calls)

    def test_tf_executor_loss_handoff_cannot_be_overwritten_by_healthy_sample(self):
        node = self.make_localization_node()
        handoff = self.tf_module.OdomInfoHandoff()

        class Receiver:
            def odom_info_snapshot(self):
                return handoff.snapshot()

            def acknowledge_odom_info_loss(self, sequence):
                handoff.acknowledge_loss(sequence)

            def invalidate_odom_info(self, barrier_stamp_ns, *, after_sequence):
                handoff.invalidate(barrier_stamp_ns, after_sequence=after_sequence)

        node._tf_receiver = Receiver()
        node._localization_available = True
        handle = FakeHandle()
        candidate = self.candidate("active")
        goal = own_active_goal(node, candidate, 91, handle)
        handoff.add(9_900_000_000, "vo_odom", False)
        lost, _ = handoff.add(9_950_000_000, "vo_odom", True)
        healthy, _ = handoff.add(10_000_000_001, "vo_odom", False)

        node._drain_odom_info_snapshot()

        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(goal["localization_invalidated"])
        self.assertEqual(10_000_000_000, node._localization_invalidated_at_ns)
        self.assertEqual([healthy.sequence], [sample.sequence for sample in node._odom_info])
        self.assertEqual(healthy.stamp_ns, node._stamp_ns(node._odom_info[-1]))
        self.assertEqual(lost.sequence, node._last_processed_odom_info_sequence - 1)
        self.assertIsNone(handoff.snapshot()[1])

    def test_handoff_preserves_latest_temporally_eligible_odom_info(self):
        node = self.make_localization_node()
        handoff = self.tf_module.OdomInfoHandoff()

        class Receiver:
            def odom_info_snapshot(self):
                return handoff.snapshot()

        node._tf_receiver = Receiver()
        handoff.add(9_800_000_000, "vo_odom", False)
        handoff.add(10_100_000_000, "vo_odom", False)
        handoff.add(9_900_000_000, "vo_odom", False)

        node._drain_odom_info_snapshot()

        self.assertEqual((1.0, 2.0, 0.0), node._localization_pose("map"))
        self.assertEqual(9_900_000_000, node._last_accepted_odom_stamp_ns)

    def test_clock_regression_clears_pending_handoff_epoch(self):
        node = self.make_localization_node()
        handoff = self.tf_module.OdomInfoHandoff()

        class Receiver:
            def odom_info_snapshot(self):
                return handoff.snapshot()

            def reset_odom_info(self):
                return handoff.reset()

        node._tf_receiver = Receiver()
        old, _ = handoff.add(10_500_000_000, "vo_odom", False)
        node._last_ros_time_ns = 10_000_000_000
        node._now_ns = 1_000_000_000

        node._drain_odom_info_snapshot()

        self.assertEqual(((), None, old.sequence), handoff.snapshot())
        self.assertEqual(old.sequence, node._last_processed_odom_info_sequence)
        self.assertEqual([], list(node._odom_info))

    def test_clock_regression_during_snapshot_drain_discards_old_epoch_batch(self):
        node = self.make_localization_node()
        handoff = self.tf_module.OdomInfoHandoff()
        old, _ = handoff.add(9_850_000_000, "vo_odom", False)
        node._last_ros_time_ns = 10_000_000_000
        node._now_ns = 10_000_000_000

        class Receiver:
            def odom_info_snapshot(self):
                snapshot = handoff.snapshot()
                node._now_ns = 9_900_000_000
                return snapshot

            def reset_odom_info(self):
                return handoff.reset()

        node._tf_receiver = Receiver()
        node._drain_odom_info_snapshot()

        self.assertEqual(1, node._clock_epoch)
        self.assertEqual([], list(node._odom_info))
        self.assertEqual(old.sequence, node._last_processed_odom_info_sequence)
        self.assertEqual(((), None, old.sequence), handoff.snapshot())
        self.assertIsNone(node._localization_pose("map"))

    def test_delayed_old_loss_after_clock_reset_does_not_block_current_epoch(self):
        for newer_loss in (False, True):
            with self.subTest(newer_loss=newer_loss):
                node = self.make_localization_node()
                handoff = self.tf_module.OdomInfoHandoff()
                node._tf_receiver = types.SimpleNamespace(
                    odom_info_snapshot=handoff.snapshot,
                    reset_odom_info=handoff.reset,
                    acknowledge_odom_info_loss=handoff.acknowledge_loss,
                    invalidate_odom_info=handoff.invalidate,
                    set_odom_info_barrier=handoff.set_barrier,
                )
                node._last_ros_time_ns = 10_000_000_000
                node._now_ns = 1_000_000_000
                node._drain_odom_info_snapshot()
                self.assertEqual(1, node._clock_epoch)

                handoff.add(9_950_000_000, "vo_odom", True)
                if newer_loss:
                    # The sticky loss floor must not carry the old epoch
                    # into a newer loss received before Search drains it.
                    handoff.add(1_000_000_000, "vo_odom", True)
                node._drain_odom_info_snapshot()
                expected_barrier = 1_000_000_000 if newer_loss else None
                self.assertEqual(expected_barrier, node._localization_invalidated_at_ns)
                self.assertFalse(node._localization_available)
                self.assertIsNone(handoff.snapshot()[1])

                node._now_ns = 1_100_000_000
                _, disposition = handoff.add(node._now_ns, "vo_odom", False)
                self.assertEqual("queued", disposition)
                node._drain_odom_info_snapshot()
                node._tf_buffer.stamps.update({
                    "vo_odom": node._now_ns, "map": node._now_ns,
                })
                self.assertIsNotNone(node._localization_pose("map"))
                self.assertEqual(1_100_000_000, node._last_accepted_odom_stamp_ns)
                self.assertIsNone(node._localization_invalidated_at_ns)

    def test_delayed_old_loss_after_recovery_still_cancels_goal(self):
        node = self.make_localization_node()
        node._last_ros_time_ns = 10_000_000_000
        node._now_ns = 1_000_000_000
        node._current_ros_time_ns()
        node._now_ns = 1_100_000_000
        node._odom_info_callback(self.odom_info(node._now_ns))
        self.assertIsNotNone(node._localization_pose("map"))
        handle = FakeHandle()
        goal = own_active_goal(node, self.candidate("active"), 1, handle)

        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))

        self.assertEqual(1, handle.cancel_count)
        self.assertFalse(node._localization_available)
        self.assertTrue(goal["localization_invalidated"])
        self.assertEqual(1_100_000_000, node._localization_invalidated_at_ns)

    def test_clock_regression_during_lost_invalidation_keeps_reset_clear(self):
        node = self.make_localization_node()
        node._last_ros_time_ns = 10_000_000_000
        node._localization_available = True
        times = iter((10_000_000_000, 9_900_000_000))
        node.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=next(times))
        )

        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))

        self.assertEqual(1, node._clock_epoch)
        self.assertFalse(node._localization_available)
        self.assertIsNone(node._localization_invalidated_at_ns)
        self.assertEqual([], list(node._odom_info))

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
        node._tf_buffer.stamps.update({
            "vo_odom": 10_050_000_000,
            "map": 10_050_000_000,
        })
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
        self.assertEqual(1, len(node._tf_buffer.calls))
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
        goal = own_active_goal(node, candidate, 30, handle)
        goal["cancel_requested"] = True
        goal["cancel_reason"] = "search_disabled"
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.missing.add("vo_odom")
        self.assertIsNone(node._localization_pose("map"))
        self.assertTrue(goal["localization_invalidated"])
        self.assertEqual(0, handle.cancel_count)
        node._goal_result(Future(types.SimpleNamespace(status=6)), 30)
        self.assertEqual([], node._blacklisted_frontiers)


    def test_either_missing_transform_cancels_active_goal_once(self):
        for target_frame in ("vo_odom", "map"):
            with self.subTest(target_frame=target_frame):
                node = self.make_localization_node()
                candidate = self.candidate("active")
                handle = FakeHandle()
                goal = own_active_goal(node, candidate, 70, handle)
                node._odom_info_callback(self.odom_info(9_900_000_000))
                node._tf_buffer.missing.add(target_frame)
                self.assertIsNone(node._localization_pose("map"))
                self.assertTrue(goal["localization_invalidated"])
                self.assertEqual(1, handle.cancel_count)
                node._odom_info_callback(self.odom_info(10_000_000_000, lost=True))
                self.assertEqual(1, handle.cancel_count)

    def test_tf_lookup_failures_log_original_context_and_preserve_invalidation(self):
        for target_frame, reason, expected_call_count in (
            ("vo_odom", "full TF unavailable", 1),
            ("map", "full TF unavailable", 1),
        ):
            with self.subTest(target_frame=target_frame):
                node = self.make_localization_node()
                warnings = []
                node._logger.warning = warnings.append
                node._localization_available = True
                node._last_accepted_odom_stamp_ns = 9_900_000_000
                candidate = self.candidate("active")
                handle = FakeHandle()
                goal = own_active_goal(node, candidate, 71, handle)
                node._odom_info_callback(self.odom_info(9_900_000_000))
                original_error = RuntimeError("complete TF failure detail")
                node._tf_buffer.errors[target_frame] = original_error

                self.assertIsNone(node._localization_pose("map"))

                self.assertEqual(expected_call_count, len(node._tf_buffer.calls))
                self.assertEqual(10_000_000_000,
                                 node._localization_invalidated_at_ns)
                self.assertTrue(goal["localization_invalidated"])
                self.assertEqual(1, handle.cancel_count)
                self.assertEqual([], node._blacklisted_frontiers)
                self.assertEqual(1, len(warnings))
                warning = warnings[0]
                self.assertIn(f"Search localization invalidated: {reason}", warning)
                self.assertIn("selected_info_stamp_ns=9900000000", warning)
                self.assertIn("requested_stamp_ns=0", warning)
                self.assertIn("validation_ros_now_ns=10000000000", warning)
                self.assertIn("info_age_ns=100000000", warning)
                self.assertIn("actual_tf_stamp_ns=None", warning)
                self.assertIn("actual_tf_age_ns=None", warning)
                self.assertIn("max_age_ns=500000000", warning)
                self.assertIn("target_frame='map'", warning)
                self.assertIn("source_frame='base_link'", warning)
                self.assertIn("exception_class=RuntimeError", warning)
                self.assertIn("exception_message='complete TF failure detail'", warning)

    def test_tf_failure_invalidates_and_cancels_goal_with_warning(self):
        node = self.make_localization_node()
        warnings = []
        events = []
        node._logger.warning = warnings.append
        node._localization_available = True
        node._last_accepted_odom_stamp_ns = 9_900_000_000
        handle = FakeHandle()
        candidate = self.candidate("active")
        goal = own_active_goal(node, candidate, 73, handle)
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.missing.add("vo_odom")
        cancel_goal = node._cancel_goal
        node._cancel_goal = lambda owned, reason: (
            events.append("cancel_started") or cancel_goal(owned, reason)
        )

        self.assertIsNone(node._localization_pose("map"))

        self.assertEqual(["cancel_started"], events)
        self.assertFalse(node._localization_available)
        self.assertTrue(goal["localization_invalidated"])
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual(1, len(warnings))
        self.assertTrue(warnings[0].startswith("Search localization invalidated:"))
        self.assertIn("exception_message='vo_odom'", warnings[0])

    def test_latest_tf_failure_does_not_fall_back_to_older_info(self):
        node = self.make_localization_node()
        node._localization_available = True
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._odom_info_callback(self.odom_info(9_950_000_000))
        node._tf_buffer.errors["vo_odom"] = (
            self.tf_exceptions.ExtrapolationException("newer unavailable")
        )

        self.assertIsNone(node._localization_pose("map"))
        self.assertEqual(1, len(node._tf_buffer.calls))
        self.assertFalse(node._localization_available)


    def test_tf_failure_warning_is_suppressed_for_repeated_same_reason(self):
        node = self.make_localization_node()
        warnings = []
        node._logger.warning = warnings.append
        node._localization_available = True
        handle = FakeHandle()
        own_active_goal(node, self.candidate("active"), 72, handle)
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

        self.assertEqual(1, len(node._tf_buffer.calls))
        self.assertEqual([], warnings)



    def test_changed_top_goal_is_sent_while_previous_goal_is_active(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 1.05, 4.0)
        node._apply_search_result(self.result([first]))
        first_token = node._current_goal_token
        first_handle = FakeHandle()
        node._goal_response(Future(first_handle), first_token)

        node._apply_search_result(self.result([first, second]))

        self.assertEqual(2, len(node._action.sent))
        self.assertEqual(0, first_handle.cancel_count)
        second_token = node._current_goal_token
        self.assertNotEqual(first_token, second_token)
        self.assertEqual([4.0, 2.0, 0.5], node._owned_goals[second_token]["candidate"]["pose_world"])
        self.assertLess(second["score"] - first["score"], 0.10)

        same_centroid = self.candidate("second-update", 1.06, 4.005)
        node._apply_search_result(self.result([same_centroid]))
        self.assertEqual(2, len(node._action.sent))
        self.assertEqual(0, first_handle.cancel_count)

    def test_disable_tracks_three_pending_requests_and_waits_for_terminal_results(self):
        node = self.make_node()
        candidates = [self.candidate(name, score, x) for name, score, x in (
            ("first", 1.0, 1.0), ("second", 1.1, 4.0), ("third", 1.2, 7.0),
        )]
        for candidate in candidates:
            node._apply_search_result(self.result([candidate]))
        tokens = (1, 2, 3)
        unrelated = FakeHandle()

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertFalse(response.success)
        self.assertEqual(set(tokens), set(node._owned_goals))

        second_handle = FakeHandle()
        node._goal_response(Future(second_handle), tokens[1])
        node._goal_response(Future(FakeHandle(accepted=False)), tokens[0])
        third_handle = FakeHandle()
        node._goal_response(Future(third_handle), tokens[2])
        self.assertEqual(1, second_handle.cancel_count)
        self.assertEqual(1, third_handle.cancel_count)
        self.assertEqual(0, unrelated.cancel_count)
        self.assertEqual({tokens[1], tokens[2]}, set(node._owned_goals))

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertFalse(response.success)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_CANCELED)), tokens[1])
        self.assertEqual({tokens[2]}, set(node._owned_goals))
        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertFalse(response.success)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_CANCELED)), tokens[2])
        self.assertFalse(node._owned_goals)
        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertTrue(response.success)

    def test_goal_response_order_is_scoped_to_each_token(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 2.0, 4.0)
        node._apply_search_result(self.result([first]))
        first_token = node._current_goal_token
        node._apply_search_result(self.result([second]))
        second_token = node._current_goal_token

        node._goal_response(Future(FakeHandle(accepted=False)), second_token)
        self.assertEqual({first_token}, set(node._owned_goals))
        self.assertIsNone(node._current_goal_token)
        handle = FakeHandle()
        node._goal_response(Future(handle), first_token)
        self.assertIs(handle, node._owned_goals[first_token]["handle"])
        self.assertIsNone(node._current_goal_token)
        self.assertEqual(first_token, node._active_goal_token)

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=False), response)
        self.assertFalse(response.success)
        self.assertEqual(1, handle.cancel_count)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_CANCELED)), first_token)
        self.assertFalse(node._owned_goals)

    def test_old_goal_result_only_settles_its_own_record(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 2.0, 4.0)
        node._apply_search_result(self.result([first]))
        first_token = node._current_goal_token
        first_handle = FakeHandle()
        node._goal_response(Future(first_handle), first_token)
        node._apply_search_result(self.result([second]))
        second_token = node._current_goal_token
        second_handle = FakeHandle()
        node._goal_response(Future(second_handle), second_token)

        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_SUCCEEDED)), first_token)

        self.assertEqual({second_token}, set(node._owned_goals))
        self.assertEqual(second_token, node._current_goal_token)
        self.assertEqual(second_token, node._progress_goal_token)
        self.assertEqual([], node._blacklisted_frontiers)

    def test_previous_result_remains_current_until_replacement_is_accepted(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 2.0, 4.0)
        node._apply_search_result(self.result([first]))
        first_token = node._current_goal_token
        node._goal_response(Future(FakeHandle()), first_token)
        node._apply_search_result(self.result([second]))
        second_token = node._current_goal_token

        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_ABORTED)), first_token)

        self.assertTrue(node._is_blacklisted(first))
        self.assertIn(second_token, node._owned_goals)
        self.assertEqual(second_token, node._current_goal_token)
        second_handle = FakeHandle()
        node._goal_response(Future(second_handle), second_token)
        self.assertEqual(second_token, node._active_goal_token)

    def test_current_result_before_replacement_acceptance_uses_upstream_active_flag(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 2.0, 4.0)
        node._apply_search_result(self.result([first]))
        first_token = node._current_goal_token
        node._goal_response(Future(FakeHandle()), first_token)
        node._apply_search_result(self.result([second]))
        pending_replacement = node._current_goal_token

        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_SUCCEEDED)), first_token)

        self.assertIsNone(node._active_goal_token)
        self.assertFalse(node._goal_active)
        self.assertEqual(pending_replacement, node._current_goal_token)
        node._apply_search_result(self.result([second]))
        self.assertEqual(3, len(node._action.sent))
        self.assertNotEqual(pending_replacement, node._current_goal_token)

    def test_progress_uses_upstream_min_distance_and_timeout_blacklists_frontier(self):
        node = self.make_node()
        candidate = self.candidate("current", 1.0, 3.0)
        candidate["min_distance"] = 8.0
        node._apply_search_result(self.result([candidate]))
        token = node._current_goal_token
        node._goal_response(Future(FakeHandle()), token)
        started = node._progress_started_ns

        closer = dict(candidate, min_distance=7.5)
        node._now_ns = started + 10_000_000_000
        node._apply_search_result(self.result([closer]))
        self.assertEqual(7.5, node._progress_min_distance)
        self.assertEqual(node._now_ns, node._progress_started_ns)
        node._now_ns += 29_000_000_000
        node._check_progress_timeout()
        self.assertFalse(node._is_blacklisted(candidate))
        node._now_ns += 1_000_000_000
        node._check_progress_timeout()
        self.assertFalse(node._is_blacklisted(candidate))
        node._now_ns += 1
        node._check_progress_timeout()
        self.assertTrue(node._is_blacklisted(candidate))
        self.assertIn(token, node._owned_goals)

        next_frontier = self.candidate("next", 0.5, 8.0)
        node._apply_search_result(self.result([closer, next_frontier]))
        self.assertEqual(2, len(node._action.sent))
        self.assertEqual(next_frontier["pose_world"][:2],
                         node._owned_goals[node._current_goal_token]["candidate"]["pose_world"][:2])

    def test_blacklist_uses_upstream_five_cell_centroid_tolerance(self):
        node = self.make_node()
        node._blacklisted_frontiers.append((1.0, 2.0))
        inside = self.candidate("inside", x=1.49)
        on_boundary = self.candidate("boundary", x=1.5)

        self.assertTrue(node._is_blacklisted(inside))
        self.assertFalse(node._is_blacklisted(on_boundary))

    def test_terminal_results_keep_upstream_success_cancel_and_abort_behavior(self):
        node = self.make_node()
        succeeded = self.candidate("success", 1.0, 1.0)
        node._apply_search_result(self.result([succeeded]))
        success_token = node._current_goal_token
        node._goal_response(Future(FakeHandle()), success_token)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_SUCCEEDED)), success_token)
        self.assertFalse(node._is_blacklisted(succeeded))
        self.assertIsNone(node._current_goal_token)
        self.assertEqual(1, len(node._frontier_pose.messages))

        canceled = self.candidate("canceled", 2.0, 4.0)
        node._apply_search_result(self.result([canceled]))
        canceled_token = node._current_goal_token
        node._goal_response(Future(FakeHandle()), canceled_token)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_CANCELED)), canceled_token)
        self.assertFalse(node._is_blacklisted(canceled))

        aborted = self.candidate("aborted", 3.0, 7.0)
        next_frontier = self.candidate("next", 2.0, 9.0)
        node._apply_search_result(self.result([aborted]))
        aborted_token = node._current_goal_token
        node._goal_response(Future(FakeHandle()), aborted_token)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_ABORTED)), aborted_token)
        self.assertTrue(node._is_blacklisted(aborted))
        node._apply_search_result(self.result([aborted, next_frontier]))
        self.assertEqual(next_frontier["pose_world"][:2],
                         node._owned_goals[node._current_goal_token]["candidate"]["pose_world"][:2])

    def test_localization_loss_blocks_new_goals_and_cancels_late_acceptance(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 2.0, 4.0)
        node._apply_search_result(self.result([first]))
        first_token = node._current_goal_token
        first_handle = FakeHandle()
        node._goal_response(Future(first_handle), first_token)
        node._apply_search_result(self.result([second]))
        second_token = node._current_goal_token

        node._invalidate_localization("test loss", 10_000_000_000)
        sent_count = len(node._action.sent)
        node._apply_search_result(self.result([self.candidate("third", 3.0, 7.0)]))
        self.assertEqual(sent_count, len(node._action.sent))
        self.assertEqual(1, first_handle.cancel_count)
        second_handle = FakeHandle()
        node._goal_response(Future(second_handle), second_token)
        self.assertEqual(1, second_handle.cancel_count)

        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_CANCELED)), first_token)
        node._goal_result(Future(types.SimpleNamespace(
            status=self.module.GoalStatus.STATUS_CANCELED)), second_token)
        self.assertFalse(node._owned_goals)

    def test_selected_goal_marker_uses_latest_owned_pose(self):
        node = self.make_node()
        first = self.candidate("first", 1.0, 1.0)
        second = self.candidate("second", 2.0, 4.0)
        own_active_goal(node, first, 1, FakeHandle())
        own_active_goal(node, second, 2, FakeHandle())
        self.assertIs(second, node._selected_for_markers({"candidates": []}))


if __name__ == "__main__":
    unittest.main()
