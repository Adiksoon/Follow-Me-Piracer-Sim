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
            header=types.SimpleNamespace(frame_id="map")
        )
        node._last_cluster_ranking = None
        node._cluster_ranking_context = None
        node._next_goal_requested = False
        node._failed_attempts = {}
        node._exhausted_candidates = set()
        node._completed_candidates = set()
        node._retry_target = None
        node._replacement_target = None
        node._replacement_proposal = None
        node._goal_switch_deadline_ns = None
        node._next_goal_dynamic = False
        node._goal_sequence = 0
        node._pending_goal = None
        node._active_goal = None
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
        node._goal_switch_deadline_ns = node._now_ns
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
            "base_frame": "base_link",
            "max_localization_age_sec": 0.5,
            "odom_info_topic": "/visual_vo/odom_info_lite",
            "tf_future_lead_sec": 0.1,
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
            "cluster_candidates": [
                {"cluster_id": item["cluster_id"], "score": item["score"],
                 "frontiers_world": item["source_frontiers_world"], "id": item["id"]}
                for item in candidates
            ],
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
                node._active_goal = {'handle': handle, 'cancel_requested': False,
                                     'cancel_reason': None}
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
        node._active_goal = {
            "token": 91, "candidate": candidate,
            "identity": node._goal_identity(candidate, "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))
        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(node._active_goal["localization_invalidated"])
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
        node._active_goal = {
            "token": 91, "candidate": candidate,
            "identity": node._goal_identity(candidate, "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }

        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))

        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(node._active_goal["localization_invalidated"])
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
        node._active_goal = {
            "token": 91, "candidate": candidate,
            "identity": node._goal_identity(candidate, "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        handoff.add(9_900_000_000, "vo_odom", False)
        lost, _ = handoff.add(9_950_000_000, "vo_odom", True)
        healthy, _ = handoff.add(10_000_000_001, "vo_odom", False)

        node._drain_odom_info_snapshot()

        self.assertEqual(1, handle.cancel_count)
        self.assertTrue(node._active_goal["localization_invalidated"])
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
        node._active_goal = {
            "handle": handle, "cancel_requested": False, "cancel_reason": None,
        }

        node._odom_info_callback(self.odom_info(9_950_000_000, lost=True))

        self.assertEqual(1, handle.cancel_count)
        self.assertFalse(node._localization_available)
        self.assertTrue(node._active_goal["localization_invalidated"])
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
        node._active_goal = {
            "token": 73, "candidate": candidate,
            "identity": node._goal_identity(candidate, "map"),
            "epoch": 0, "is_retry": False, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._odom_info_callback(self.odom_info(9_900_000_000))
        node._tf_buffer.missing.add("vo_odom")
        node._reset_challenger = lambda: events.append("challenger_reset")
        cancel_active_goal = node._cancel_active_goal
        node._cancel_active_goal = lambda reason: (
            events.append("cancel_started") or cancel_active_goal(reason)
        )

        self.assertIsNone(node._localization_pose("map"))

        self.assertEqual(["challenger_reset", "cancel_started"], events)
        self.assertFalse(node._localization_available)
        self.assertTrue(node._active_goal["localization_invalidated"])
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

        self.assertEqual(1, len(node._tf_buffer.calls))
        self.assertEqual([], warnings)



    def test_cluster_backlog_is_never_sent_from_result_application(self):
        node = self.make_node()
        detailed = [self.candidate(f"safe-{i}", 1.0 - i / 10, 1.0 + i)
                    for i in range(3)]
        for candidate in detailed:
            candidate["cluster_size"] = 2
            node._completed_candidates.add(node._goal_identity(candidate, "map"))
        unsafe = self.candidate("coarse-unsafe", 0.1, 8.0)
        unsafe.update({"cluster_size": 2, "source_frontier_world": [8.0, 8.0],
                       "source_seed_world": [8.0, 7.0]})
        result = {"status": "RECOMMENDATION_AVAILABLE", "clustered": True,
                  "candidates": detailed, "ranking": [c["id"] for c in detailed],
                  "cluster_candidates": [*detailed, unsafe]}

        node._apply_search_result(result)

        self.assertEqual([], node._action.sent)
        self.assertTrue(node._next_goal_requested)






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

    def test_aborted_goal_exhausts_candidate_after_one_failure(self):
        node = self.make_node()
        first, second = self.candidate("first", 0.5), self.candidate("second", 0.2)
        result = self.result([first, second])
        candidate = node._next_ranked_candidate(result)
        self.assertEqual("first", candidate["id"])
        handle = FakeHandle()
        node._active_goal = {
            "token": 1, "candidate": candidate, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._goal_result(Future(types.SimpleNamespace(status=6)), 1)
        first_identity = node._goal_identity(first, "map")
        self.assertEqual(1, node._failed_attempts[first_identity])
        self.assertIn(first_identity, node._exhausted_candidates)
        self.assertEqual("second", node._next_ranked_candidate(result)["id"])

        node._active_goal = {
            "token": 2, "candidate": second, "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        node._goal_result(Future(types.SimpleNamespace(status=4)), 2)
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
        self.assertNotIn(node._goal_identity(candidate, "map"), node._exhausted_candidates)

    def test_localization_invalidated_abort_does_not_exhaust_candidate(self):
        node = self.make_node()
        candidate = self.candidate("invalidated")
        node._active_goal = {
            "token": 9, "candidate": candidate, "handle": FakeHandle(),
            "cancel_requested": True, "cancel_reason": "localization_invalidated",
            "localization_invalidated": True,
        }

        node._goal_result(Future(types.SimpleNamespace(status=6)), 9)

        identity = node._goal_identity(candidate, "map")
        self.assertNotIn(identity, node._failed_attempts)
        self.assertNotIn(identity, node._exhausted_candidates)

    def test_localization_invalidated_success_does_not_complete_candidate(self):
        node = self.make_node()
        candidate = self.candidate("invalidated")
        node._active_goal = {
            "token": 10, "candidate": candidate, "handle": FakeHandle(),
            "epoch": node._search_epoch, "cancel_requested": True,
            "cancel_reason": "localization_invalidated",
            "localization_invalidated": True,
            "identity": node._goal_identity(candidate, "map"),
        }

        node._goal_result(Future(types.SimpleNamespace(status=4)), 10)

        self.assertNotIn(node._goal_identity(candidate, "map"), node._completed_candidates)
        self.assertIsNone(node._active_goal)

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
        replacement_candidate = self.candidate("replacement")
        replacement = node._goal_identity(replacement_candidate, "map")
        node._enabled = False
        node._failed_attempts[retry] = 1
        node._exhausted_candidates.add(replacement)
        node._completed_candidates.add(done)
        node._retry_target = retry
        node._replacement_target = replacement
        node._replacement_proposal = {"candidate": replacement_candidate, "context": {}}
        node._goal_switch_deadline_ns = 55

        response = types.SimpleNamespace(success=None, message="")
        node._set_search(types.SimpleNamespace(data=True), response)

        self.assertTrue(response.success)
        self.assertEqual({}, node._failed_attempts)
        self.assertEqual(set(), node._exhausted_candidates)
        self.assertEqual(set(), node._completed_candidates)
        self.assertIsNone(node._retry_target)
        self.assertIsNone(node._replacement_target)
        self.assertIsNone(node._replacement_proposal)
        self.assertIsNone(node._goal_switch_deadline_ns)
        self.assertEqual(
            "replacement",
            node._next_ranked_candidate(self.result([replacement_candidate]))["id"],
        )

    def test_proposal_is_cleared_on_disable_and_localization_loss(self):
        for reset in ("disable", "localization_loss"):
            with self.subTest(reset=reset):
                node = self.make_node()
                node._replacement_proposal = {"candidate": self.candidate("proposal"),
                                              "context": {}}
                node._replacement_target = node._goal_identity(
                    node._replacement_proposal["candidate"], "map"
                )
                node._goal_switch_deadline_ns = 123
                if reset == "disable":
                    response = types.SimpleNamespace(success=None, message="")
                    node._set_search(types.SimpleNamespace(data=False), response)
                else:
                    node._invalidate_localization("test lost sample")
                self.assertIsNone(node._replacement_proposal)
                self.assertIsNone(node._goal_switch_deadline_ns)
                self.assertFalse(node._next_goal_dynamic)

    def test_switch_uses_current_coarse_score_after_deadline_once(self):
        node = self.make_node()
        current = self.candidate("current", 0.99, 3.0)
        challenger = self.candidate("challenger", 0.30, 6.0)
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        node._send_goal(current, "/tree.xml")
        token = node._pending_goal["token"]
        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(20_000_000_000, node._goal_switch_deadline_ns)
        evaluation = self.result([current, challenger])
        for cluster in evaluation["cluster_candidates"]:
            cluster["score"] = 0.20 if cluster["cluster_id"] == "current" else 0.30
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._now_ns = 19_999_000_000
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(0, handle.cancel_count)
        node._now_ns = 20_000_000_000
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("higher_scoring_candidate", node._active_goal["cancel_reason"])
        self.assertIsNotNone(node._replacement_proposal)


    def test_goal_switch_diagnostic_records_keep_and_switch_decisions(self):
        for challenger_score, expected_decision, expected_reason in (
                (0.29, "KEEP", "score_delta_below_margin"),
                (0.30, "SWITCH", "challenger_score_delta_met_margin")):
            with self.subTest(decision=expected_decision):
                node = self.make_node()
                records = []
                node._logger.info = records.append
                current = self.candidate("current", 0.2, 1.0)
                challenger = self.candidate("challenger", challenger_score, 3.0)
                result = self.result([current, challenger])
                node._localization_pose = lambda frame: (0.0, 2.0, 0.0)
                handle = FakeHandle()
                node._active_goal = {
                    "token": 99, "candidate": current,
                    "identity": node._goal_identity(current, "map"),
                    "epoch": node._search_epoch, "handle": handle,
                    "cancel_requested": False, "cancel_reason": None,
                }
                node._now_ns = 20_000_000_000
                node._goal_switch_deadline_ns = node._now_ns

                node._consider_goal_switch(result, {item["id"]: item for item in result["candidates"]})

                self.assertEqual(1, len(records))
                record = json.loads(records[0])
                self.assertEqual({
                    "event", "ros_time_ns", "active_score", "challenger_score",
                    "delta", "distance_m", "matched_clusters", "decision", "reason",
                }, set(record))
                self.assertEqual("goal_switch_evaluation", record["event"])
                self.assertEqual(20_000_000_000, record["ros_time_ns"])
                self.assertEqual(0.2, record["active_score"])
                self.assertEqual(challenger_score, record["challenger_score"])
                self.assertAlmostEqual(challenger_score - 0.2, record["delta"])
                self.assertAlmostEqual(1.0, record["distance_m"])
                self.assertEqual(1, record["matched_clusters"])
                self.assertEqual(expected_decision, record["decision"])
                self.assertEqual(expected_reason, record["reason"])
                self.assertEqual(expected_decision == "SWITCH", handle.cancel_count == 1)
                self.assertEqual(expected_decision == "SWITCH", node._replacement_proposal is not None)

    def test_goal_switch_diagnostic_records_due_skips_and_null_unavailable_values(self):
        cases = (
            ("localization_unavailable", "localization_unavailable"),
            ("no_eligible_challenger", "no_eligible_challenger"),
            ("active_frontier_unmatched", "active_frontier_unmatched"),
            ("invalid_challenger_score", "challenger_score_unavailable"),
        )
        for scenario, expected_reason in cases:
            with self.subTest(scenario=scenario):
                node = self.make_node()
                records = []
                node._logger.info = records.append
                current = self.candidate("current", 0.2, 1.0)
                challenger = self.candidate("challenger", 0.5, 3.0)
                result = self.result([current, challenger])
                node._localization_pose = lambda frame: (0.0, 2.0, 0.0)
                node._active_goal = {
                    "candidate": current, "identity": node._goal_identity(current, "map"),
                    "epoch": node._search_epoch, "handle": FakeHandle(),
                    "cancel_requested": False, "cancel_reason": None,
                }
                node._now_ns = 20_000_000_000
                node._goal_switch_deadline_ns = node._now_ns
                if scenario == "localization_unavailable":
                    node._localization_pose = lambda frame: None
                elif scenario == "no_eligible_challenger":
                    result = self.result([current])
                elif scenario == "invalid_challenger_score":
                    next(cluster for cluster in result["cluster_candidates"]
                         if cluster["cluster_id"] == "challenger")["score"] = float("nan")
                else:
                    with patch.object(self.module, "match_active_frontiers", return_value=None):
                        node._consider_goal_switch(result, {item["id"]: item for item in result["candidates"]})
                    result = None
                if result is not None:
                    node._consider_goal_switch(result, {item["id"]: item for item in result["candidates"]})

                self.assertEqual(1, len(records))
                record = json.loads(records[0])
                self.assertEqual("SKIP", record["decision"])
                self.assertEqual(expected_reason, record["reason"])
                self.assertEqual(20_000_000_000, record["ros_time_ns"])
                if scenario == "active_frontier_unmatched":
                    self.assertIsNone(record["active_score"])
                    self.assertIsNone(record["matched_clusters"])
                elif scenario == "localization_unavailable":
                    self.assertEqual(0.2, record["active_score"])
                    self.assertEqual(1, record["matched_clusters"])
                    self.assertIsNone(record["distance_m"])
                elif scenario == "invalid_challenger_score":
                    self.assertIsNone(record["challenger_score"])
                    self.assertIsNone(record["delta"])
                else:
                    self.assertIsNone(record["challenger_score"])
                    self.assertIsNone(record["delta"])

    def test_goal_switch_diagnostic_not_due_emits_nothing(self):
        node = self.make_node()
        records = []
        node._logger.info = records.append
        current = self.candidate("current", 0.2, 1.0)
        node._active_goal = {
            "candidate": current, "identity": node._goal_identity(current, "map"),
            "epoch": node._search_epoch, "handle": FakeHandle(),
            "cancel_requested": False, "cancel_reason": None,
        }
        node._now_ns = 20_000_000_000
        node._goal_switch_deadline_ns = node._now_ns + 1
        node._consider_goal_switch(self.result([current]), {current["id"]: current})
        self.assertEqual([], records)
        self.assertFalse(node._active_goal["cancel_requested"])

    def test_goal_switch_diagnostic_logger_failure_does_not_change_switch(self):
        node = self.make_node()
        def fail_logging(message):
            raise RuntimeError("logger unavailable")
        node._logger.info = fail_logging
        current = self.candidate("current", 0.2, 1.0)
        challenger = self.candidate("challenger", 0.31, 3.0)
        result = self.result([current, challenger])
        node._localization_pose = lambda frame: (0.0, 2.0, 0.0)
        handle = FakeHandle()
        node._active_goal = {
            "candidate": current, "identity": node._goal_identity(current, "map"),
            "epoch": node._search_epoch, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._now_ns = 20_000_000_000
        node._goal_switch_deadline_ns = node._now_ns

        node._consider_goal_switch(result, {item["id"]: item for item in result["candidates"]})

        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("higher_scoring_candidate", node._active_goal["cancel_reason"])
        self.assertEqual("challenger", node._replacement_proposal["candidate"]["id"])

    def test_goal_switch_diagnostic_normalizes_oversized_ros_time_without_blocking_switch(self):
        node = self.make_node()
        records = []
        node._logger.info = records.append
        current = self.candidate("current", 0.2, 1.0)
        challenger = self.candidate("challenger", 0.31, 3.0)
        result = self.result([current, challenger])
        node._localization_pose = lambda frame: (0.0, 2.0, 0.0)
        handle = FakeHandle()
        node._active_goal = {
            "candidate": current, "identity": node._goal_identity(current, "map"),
            "epoch": node._search_epoch, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._now_ns = 10 ** 1000
        node._goal_switch_deadline_ns = node._now_ns

        node._consider_goal_switch(result, {item["id"]: item for item in result["candidates"]})

        self.assertEqual(1, len(records))
        record = json.loads(records[0])
        self.assertIsNone(record["ros_time_ns"])
        self.assertEqual("SWITCH", record["decision"])
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("challenger", node._replacement_proposal["candidate"]["id"])

    def test_switch_deadline_starts_when_goal_is_accepted(self):
        initial = self.make_node()
        initial._send_goal(self.candidate("initial"), "/tree.xml")
        token = initial._pending_goal["token"]
        initial._goal_response(Future(FakeHandle()), token)
        self.assertEqual(20_000_000_000, initial._goal_switch_deadline_ns)

        replacement = self.make_node()
        replacement._next_goal_dynamic = True
        replacement._send_goal(self.candidate("replacement"), "/tree.xml")
        token = replacement._pending_goal["token"]
        replacement._goal_response(Future(FakeHandle()), token)
        self.assertEqual(25_000_000_000, replacement._goal_switch_deadline_ns)

    def test_goal_within_thirty_centimeters_is_not_preempted(self):
        node = self.make_node()
        current = self.candidate("current", 0.1)
        challenger = self.candidate("challenger", 0.9, 4.0)
        node._localization_pose = lambda frame: (0.7, 2.0, 0.0)
        handle = FakeHandle()
        node._active_goal = {"token": 88, "candidate": current,
                             "identity": node._goal_identity(current, "map"),
                             "epoch": node._search_epoch, "handle": handle,
                             "cancel_requested": False, "cancel_reason": None}
        node._consider_goal_switch(self.result([current, challenger]),
                                   {"current": current, "challenger": challenger})
        self.assertEqual(0, handle.cancel_count)
        self.assertIsNone(node._replacement_proposal)


    def test_active_cluster_outside_ready_top_three_matches_current_score(self):
        node = self.make_node()
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        active = self.candidate("active", 0.99, 20.0)
        active["source_frontiers_world"] = [[150.0, 150.0]]
        active["source_frontier_world"] = [150.0, 150.0]
        ready = [self.candidate(f"ready-{index}", score, 30.0 + index * 3)
                 for index, score in enumerate((0.30, 0.25, 0.20))]
        result = self.result(ready)
        result["cluster_candidates"].append({
            "cluster_id": "active", "score": 0.20,
            "frontiers_world": active["source_frontiers_world"],
        })
        handle = FakeHandle()
        node._active_goal = {"token": 89, "candidate": active,
                             "identity": node._goal_identity(active, "map"),
                             "epoch": node._search_epoch, "handle": handle,
                             "cancel_requested": False, "cancel_reason": None}
        node._goal_switch_deadline_ns = node._now_ns
        node._consider_goal_switch(result, {item["id"]: item for item in ready})
        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("ready-0", node._replacement_proposal["candidate"]["id"])

    def test_ambiguous_active_matches_use_highest_score(self):
        node = self.make_node()
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        active = self.candidate("active", 0.99, 20.0)
        active["source_frontiers_world"] = [[150.0, 150.0]]
        active["source_frontier_world"] = [150.0, 150.0]
        challenger = self.candidate("challenger", 0.86, 30.0)
        result = self.result([challenger])
        result["cluster_candidates"] = [
            {"cluster_id": "active-low", "score": 0.75,
             "frontiers_world": [[150.0, 150.0]]},
            {"cluster_id": "active-high", "score": 0.80,
             "frontiers_world": [[150.2, 150.2]]},
            {"cluster_id": "challenger", "score": 0.86,
             "frontiers_world": challenger["source_frontiers_world"]},
        ]
        handle = FakeHandle()
        node._active_goal = {"token": 90, "candidate": active,
                             "identity": node._goal_identity(active, "map"),
                             "epoch": node._search_epoch, "handle": handle,
                             "cancel_requested": False, "cancel_reason": None}
        node._goal_switch_deadline_ns = node._now_ns

        node._consider_goal_switch(result, {challenger["id"]: challenger})

        self.assertEqual(0, handle.cancel_count)
        self.assertIsNone(node._replacement_proposal)

    def test_ambiguous_active_matches_are_all_excluded_as_challengers(self):
        node = self.make_node()
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        active = self.candidate("active", 0.99, 20.0)
        active["source_frontiers_world"] = [[150.0, 150.0]]
        active["source_frontier_world"] = [150.0, 150.0]
        low_match = self.candidate("active-low", 0.75, 30.0)
        challenger = self.candidate("challenger", 0.91, 35.0)
        high_match = self.candidate("active-high", 0.80, 40.0)
        # Keep the lower-scoring active fragment first to verify that every
        # matched cluster is removed before challenger selection.
        result = self.result([low_match, challenger, high_match])
        result["cluster_candidates"] = [
            {"cluster_id": "active-low", "score": 0.75,
             "frontiers_world": [[150.0, 150.0]]},
            {"cluster_id": "active-high", "score": 0.80,
             "frontiers_world": [[150.2, 150.2]]},
            {"cluster_id": "challenger", "score": 0.91,
             "frontiers_world": challenger["source_frontiers_world"]},
        ]
        handle = FakeHandle()
        node._active_goal = {"token": 91, "candidate": active,
                             "identity": node._goal_identity(active, "map"),
                             "epoch": node._search_epoch, "handle": handle,
                             "cancel_requested": False, "cancel_reason": None}
        node._goal_switch_deadline_ns = node._now_ns

        node._consider_goal_switch(result, {
            item["id"]: item for item in (low_match, challenger, high_match)
        })

        self.assertEqual(1, handle.cancel_count)
        self.assertEqual("challenger", node._replacement_proposal["candidate"]["id"])


    def test_pending_retry_safety_cancel_reason_is_preserved(self):
        node = self.make_node()
        current, remaining = self.candidate("pending"), self.candidate("remaining", 0.2, 3.0)
        node._send_goal(current, "/tree.xml")
        node._pending_goal["is_retry"] = True
        node._pending_goal["cancel_reason"] = "localization_invalidated"
        node._consider_goal_switch(self.result([remaining]), {"remaining": remaining})
        self.assertEqual("localization_invalidated", node._pending_goal["cancel_reason"])

    def test_better_candidate_below_margin_does_not_preempt(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        challenger = self.candidate("challenger", 0.299)
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        handle = FakeHandle()
        node._active_goal = {
            "token": 6, "candidate": current, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([current, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}
        node._goal_switch_deadline_ns = node._now_ns
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(25_000_000_000, node._goal_switch_deadline_ns)
        challenger["score"] = 0.30
        for cluster in evaluation["cluster_candidates"]:
            cluster["score"] = 0.20 if cluster["cluster_id"] == "current" else 0.30
        node._now_ns = 24_999_000_000
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(0, handle.cancel_count)
        node._now_ns = 25_000_000_000
        node._consider_goal_switch(evaluation, by_id)
        self.assertEqual(1, handle.cancel_count)

    def test_retry_without_owned_goal_is_not_preempted(self):
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

        self.assertIsNone(node._replacement_proposal)
        self.assertEqual("first", node._next_ranked_candidate(
            self.result([challenger, first])
        )["id"])
        self.assertIsNone(node._replacement_target)

    def test_exact_margin_qualifies_on_first_eligible_comparison(self):
        node = self.make_node()
        first = self.candidate("first", 0.20)
        challenger = self.candidate("challenger", 0.30)
        handle = FakeHandle()
        identity = node._goal_identity(first, "map")
        node._retry_target = identity
        node._failed_attempts[identity] = 1
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        node._active_goal = {
            "token": 51, "candidate": first, "identity": identity,
            "epoch": node._search_epoch, "is_retry": True, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        evaluation = self.result([first, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)

        self.assertEqual(1, handle.cancel_count)
        self.assertEqual(node._goal_identity(challenger, "map"), node._replacement_target)
        self.assertIsNone(node._retry_target)
        self.assertIs(node._active_goal["handle"], handle)

        node._goal_result(Future(types.SimpleNamespace(status=5)), 51)
        self.assertIsNone(node._active_goal)
        self.assertEqual("challenger", node._next_ranked_candidate(evaluation)["id"])

    def test_one_eligible_comparison_cancels_without_confirmation_streak(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        first_challenger = self.candidate("challenger", 0.31, 3.0)
        changed_challenger = self.candidate("challenger", 0.31, 4.0)
        handle = FakeHandle()
        node._active_goal = {
            "token": 52, "candidate": current, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        node._consider_goal_switch(
            self.result([current, first_challenger]),
            {"current": current, "challenger": first_challenger},
        )
        self.assertEqual(1, handle.cancel_count)
        self.assertIsNotNone(node._replacement_proposal)

    def test_changed_target_identity_resets_margin_streak(self):
        node = self.make_node()
        owned = self.candidate("current", 0.20, 1.0)
        refreshed = self.candidate("current", 0.20, 9.0)
        challenger = self.candidate("challenger", 0.31, 3.0)
        handle = FakeHandle()
        node._active_goal = {
            "token": 56, "candidate": owned, "handle": handle,
            "cancel_requested": False, "cancel_reason": None,
        }
        node._consider_goal_switch(
            self.result([refreshed, challenger]),
            {"current": refreshed, "challenger": challenger},
        )

        self.assertEqual(0, handle.cancel_count)
        self.assertIsNone(node._replacement_proposal)
        self.assertIsNone(node._replacement_target)

    def test_unaccepted_goal_is_not_dynamically_replaced(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        challenger = self.candidate("challenger", 0.30, 3.0)
        node._send_goal(current, "/tree.xml")
        token = node._pending_goal["token"]
        evaluation = self.result([current, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)

        self.assertIsNone(node._pending_goal["cancel_reason"])
        self.assertIsNone(node._replacement_proposal)
        self.assertIsNotNone(node._pending_goal)
        handle = FakeHandle()
        node._goal_response(Future(handle), token)
        self.assertEqual(0, handle.cancel_count)
        self.assertIsNotNone(node._active_goal)
        node._goal_result(Future(types.SimpleNamespace(status=5)), token)

        self.assertIsNone(node._active_goal)
        self.assertEqual("current", node._next_ranked_candidate(evaluation)["id"])

    def test_aborted_retry_exhaustion_is_preserved(self):
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
        self.assertEqual(2, node._failed_attempts[identity])
        self.assertEqual("challenger", node._next_ranked_candidate(evaluation)["id"])


    def test_exhausted_identity_is_skipped_for_retry_replacement_and_ranking(self):
        node = self.make_node()
        exhausted = self.candidate("exhausted", 0.8)
        available = self.candidate("available", 0.2)
        identity = node._goal_identity(exhausted, "map")
        node._exhausted_candidates.add(identity)
        result = self.result([exhausted, available])

        node._replacement_target = identity
        self.assertEqual("available", node._next_ranked_candidate(result)["id"])
        self.assertIsNone(node._replacement_target)
        node._retry_target = identity
        self.assertEqual("available", node._next_ranked_candidate(result)["id"])
        self.assertIsNone(node._retry_target)

    def test_exhausted_identity_is_skipped_as_challenger(self):
        node = self.make_node()
        current = self.candidate("current", 0.20)
        exhausted = self.candidate("exhausted", 0.50)
        challenger = self.candidate("challenger", 0.35)
        exhausted_identity = node._goal_identity(exhausted, "map")
        node._exhausted_candidates.add(exhausted_identity)
        node._active_goal = {
            "token": 55, "candidate": current,
            "identity": node._goal_identity(current, "map"),
            "epoch": node._search_epoch, "is_retry": False,
            "handle": FakeHandle(), "cancel_requested": False,
            "cancel_reason": None,
        }
        node._localization_pose = lambda frame: (0.0, 0.0, 0.0)
        evaluation = self.result([current, exhausted, challenger])
        by_id = {item["id"]: item for item in evaluation["candidates"]}

        node._consider_goal_switch(evaluation, by_id)
        node._consider_goal_switch(evaluation, by_id)

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
