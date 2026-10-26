"""Offline TF receiver lifetime checks: fake ROS entities, real Python thread."""

import threading
import types
import unittest
from unittest.mock import patch

import test_search_node as fixtures


class TfReceiverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SearchNodeCallbackTest.setUpClass()
        cls.module = fixtures.SearchNodeCallbackTest.tf_module

    @classmethod
    def tearDownClass(cls):
        fixtures.SearchNodeCallbackTest.tearDownClass()

    def setUp(self):
        self.events = []
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.stop_requested = threading.Event()
        self.callback_release = threading.Event()
        self.context_shutdown = False
        self.block_callback = False
        self.node_options = None
        self.listener_args = None
        test = self

        class FakeNode:
            def __init__(self, name, **options):
                self.name = name
                self.context = options["context"]
                self.executor = None
                test.node_options = options
                test.node = self

            def destroy_node(self):
                test.events.append("node_destroy")

        class FakeExecutor:
            def __init__(self, *, context):
                self.context = context
                self.nodes = []
                test.executor = self

            def add_node(self, node):
                self.nodes.append(node)
                node.executor = self

            def spin(self):
                test.events.append("spin_start")
                test.started.set()
                try:
                    if test.context_shutdown:
                        raise test.module.ExternalShutdownException()
                    if test.block_callback:
                        if not test.callback_release.wait(timeout=2.0):
                            raise RuntimeError("offline callback not released")
                    test.stop_requested.wait(timeout=2.0)
                finally:
                    test.events.append("spin_exit")
                    test.stopped.set()

            def shutdown(self):
                test.events.append("executor_shutdown")
                test.stop_requested.set()
                self.nodes.clear()
                return True

        class FakeListener:
            def __init__(self, *args, **kwargs):
                test.listener_args = (args, kwargs)

            def unregister(self):
                test.events.append("listener_unregister")

        class FakeParameter:
            def __init__(self, name, *, value):
                self.name, self.value = name, value

        for name, value in (("Node", FakeNode),
                            ("SingleThreadedExecutor", FakeExecutor),
                            ("SearchTransformListener", FakeListener),
                            ("Parameter", FakeParameter)):
            patcher = patch.object(self.module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def receiver(self, sim_time=True):
        self.context = object()
        self.buffer, self.trace, self.reporter = object(), object(), object()
        receiver = self.module.SearchTfReceiver(
            self.buffer, self.trace, self.reporter, context=self.context,
            namespace="/test", use_sim_time=sim_time,
        )
        self.addCleanup(receiver.close)
        return receiver

    def test_only_helper_node_in_own_executor_with_startup_clock_and_same_buffer(self):
        for sim_time in (True, False):
            with self.subTest(use_sim_time=sim_time):
                receiver = self.receiver(sim_time)
                self.assertEqual([receiver.node], receiver.executor.nodes)
                self.assertIs(self.context, receiver.executor.context)
                self.assertIs(self.context, receiver.node.context)
                self.assertEqual("piracer_search_tf_listener", receiver.node.name)
                self.assertFalse(self.node_options["use_global_arguments"])
                self.assertFalse(self.node_options["enable_rosout"])
                self.assertFalse(self.node_options["start_parameter_services"])
                self.assertEqual("/test", self.node_options["namespace"])
                parameter, = self.node_options["parameter_overrides"]
                self.assertEqual(("use_sim_time", sim_time), (parameter.name, parameter.value))
                args, kwargs = self.listener_args
                self.assertEqual((self.buffer, receiver.node, self.trace, self.reporter), args)
                self.assertEqual({}, kwargs)  # Preserve native listener QoS/topics/spin_thread defaults.
                self.assertIsNone(receiver.thread.ident)
                self.assertFalse(receiver.thread.daemon)
                receiver.close()

    def test_real_thread_has_separate_identity_and_is_joined_before_destroy(self):
        receiver = self.receiver()
        receiver.start()
        self.assertTrue(self.started.wait(timeout=1.0))
        self.assertNotEqual(threading.get_ident(), receiver.thread.ident)
        receiver.close()
        receiver.close()
        self.assertFalse(receiver.thread.is_alive())
        self.assertIsNone(receiver.node.executor)
        self.assertEqual(["spin_start", "executor_shutdown", "spin_exit",
                          "listener_unregister", "node_destroy"], self.events)

    def test_context_shutdown_is_expected_and_cleanup_still_runs(self):
        self.context_shutdown = True
        receiver = self.receiver()
        receiver.start()
        self.assertTrue(self.stopped.wait(timeout=1.0))
        receiver.close()
        self.assertFalse(receiver.thread.is_alive())
        self.assertEqual("node_destroy", self.events[-1])
        self.assertEqual(1, self.events.count("listener_unregister"))

    def test_thread_start_failure_can_be_cleaned_without_joining_unstarted_thread(self):
        receiver = self.receiver()
        with patch.object(receiver.thread, "start", side_effect=RuntimeError("cannot start")):
            with self.assertRaisesRegex(RuntimeError, "cannot start"):
                receiver.start()
        receiver.close()
        self.assertEqual(["executor_shutdown", "listener_unregister", "node_destroy"], self.events)
        self.assertIsNone(receiver.thread.ident)

    def test_partial_initialization_releases_created_node_and_executor(self):
        for failing_part in ("SingleThreadedExecutor", "SearchTransformListener"):
            with self.subTest(failing_part=failing_part):
                self.events.clear()
                with patch.object(self.module, failing_part, side_effect=ValueError("init failed")):
                    with self.assertRaisesRegex(ValueError, "init failed"):
                        self.receiver()
                self.assertEqual("node_destroy", self.events[-1])
                self.assertEqual(1, self.events.count("node_destroy"))
                self.assertEqual(failing_part == "SearchTransformListener",
                                 "executor_shutdown" in self.events)

    def test_close_waits_for_active_callback_before_unregister_and_destroy(self):
        self.block_callback = True
        receiver = self.receiver()
        receiver.start()
        self.assertTrue(self.started.wait(timeout=1.0))
        closer = threading.Thread(target=receiver.close)
        closer.start()
        try:
            self.assertTrue(self.stop_requested.wait(timeout=1.0))
            self.assertTrue(receiver.thread.is_alive())
            self.assertNotIn("node_destroy", self.events)
            self.assertNotIn("listener_unregister", self.events)
        finally:
            self.callback_release.set()
            closer.join(timeout=2.0)
        self.assertFalse(closer.is_alive())
        self.assertFalse(receiver.thread.is_alive())
        self.assertLess(self.events.index("spin_exit"), self.events.index("node_destroy"))


if __name__ == "__main__":
    unittest.main()
