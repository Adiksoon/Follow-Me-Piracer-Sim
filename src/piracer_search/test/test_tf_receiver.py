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
        self.subscription_record = None
        self.wakeup = types.SimpleNamespace(trigger_count=0)
        self.wakeup.trigger = lambda: setattr(
            self.wakeup, "trigger_count", self.wakeup.trigger_count + 1
        )
        test = self

        class FakeNode:
            def __init__(self, name, **options):
                self.name = name
                self.context = options["context"]
                self.executor = None
                self.subscriptions = []
                test.node_options = options
                test.node = self

            def create_subscription(self, message_type, topic, callback, qos):
                record = types.SimpleNamespace(
                    message_type=message_type, topic=topic,
                    callback=callback, qos=qos,
                )
                self.subscriptions.append(record)
                test.subscription_record = record
                return record

            def get_clock(self):
                return types.SimpleNamespace(
                    now=lambda: types.SimpleNamespace(nanoseconds=1234)
                )

            def destroy_subscription(self, subscription):
                test.events.append("subscription_destroy")
                self.subscriptions.remove(subscription)

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
            odom_info_wakeup=self.wakeup,
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
                self.assertEqual([], receiver.node.subscriptions)
                self.assertFalse(hasattr(receiver, "odom_info_subscription"))
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
        self.assertEqual(["executor_shutdown", "listener_unregister",
                          "node_destroy"], self.events)
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

    def test_odom_info_callback_hands_off_only_metadata_and_wakes_search(self):
        receiver = self.receiver()
        receiver._trace = None
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=7, nanosec=25),
                frame_id="vo_odom",
            ),
            lost=False,
            word_matches=bytearray(500_000),
        )
        receiver.receive_odom_info(message)

        samples, pending_loss, latest_sequence = receiver.odom_info_snapshot()
        self.assertEqual(1, latest_sequence)
        self.assertIsNone(pending_loss)
        self.assertEqual(1, len(samples))
        sample = samples[0]
        self.assertEqual((7_000_000_025, "vo_odom", False),
                         (sample.stamp_ns, sample.frame_id, sample.lost))
        self.assertFalse(hasattr(sample, "word_matches"))
        self.assertEqual(1, self.wakeup.trigger_count)

    def test_metadata_reception_after_close_does_not_wake_or_change_snapshot(self):
        receiver = self.receiver()
        receiver._trace = None
        receiver.close()
        receiver.receive_odom_info(types.SimpleNamespace())
        self.assertEqual(((), None, 0), receiver.odom_info_snapshot())
        self.assertEqual(0, self.wakeup.trigger_count)

    def test_concurrent_old_loss_acknowledgement_keeps_newer_loss(self):
        handoff = self.module.OdomInfoHandoff()
        old, _ = handoff.add(500, "vo_odom", True)
        published = threading.Event()
        acknowledged = threading.Event()

        def writer():
            handoff.add(450, "vo_odom", True)
            published.set()
            acknowledged.wait(timeout=1.0)
            handoff.add(600, "vo_odom", False)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(published.wait(timeout=1.0))
            handoff.acknowledge_loss(old.sequence)
            pending = handoff.snapshot()[1]
            self.assertIsNotNone(pending)
            self.assertGreater(pending.sequence, old.sequence)
            self.assertEqual(500, pending.lost_stamp_floor_ns)
        finally:
            acknowledged.set()
            thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(handoff.snapshot()[1])

    def test_handoff_keeps_latest_ordered_samples_and_sticky_cumulative_loss(self):
        handoff = self.module.OdomInfoHandoff(capacity=3)
        first, _ = handoff.add(100, "vo_odom", False)
        lost_one, _ = handoff.add(500, "vo_odom", True)
        healthy_after_loss, _ = handoff.add(600, "vo_odom", False)
        lost_two, _ = handoff.add(450, "vo_odom", True)
        for stamp in (700, 800, 900, 1000):
            handoff.add(stamp, "vo_odom", False)

        samples, pending_loss, latest_sequence = handoff.snapshot()
        self.assertEqual(8, latest_sequence)
        self.assertEqual([6, 7, 8], [sample.sequence for sample in samples])
        self.assertEqual((lost_two.sequence, lost_two.stamp_ns),
                         (pending_loss.sequence, pending_loss.stamp_ns))
        self.assertEqual(500, pending_loss.lost_stamp_floor_ns)
        self.assertGreater(healthy_after_loss.sequence, lost_one.sequence)

        handoff.acknowledge_loss(lost_one.sequence)
        self.assertEqual(lost_two.sequence, handoff.snapshot()[1].sequence)
        handoff.acknowledge_loss(lost_two.sequence)
        self.assertIsNone(handoff.snapshot()[1])
        self.assertGreater(first.sequence, 0)

    def test_future_samples_barrier_filter_reset_and_overflow_are_bounded(self):
        handoff = self.module.OdomInfoHandoff(capacity=2)
        handoff.add(100, "vo_odom", False)
        second, _ = handoff.add(200, "vo_odom", False)
        handoff.invalidate(150, after_sequence=1)
        samples, _, _ = handoff.snapshot()
        self.assertEqual([second.sequence], [sample.sequence for sample in samples])
        self.assertEqual("dropped_invalidation_barrier",
                         handoff.add(150, "vo_odom", False)[1])
        handoff.add(300, "vo_odom", False)
        handoff.add(400, "vo_odom", False)
        samples, _, sequence = handoff.snapshot()
        self.assertEqual(5, sequence)
        self.assertEqual([4, 5], [sample.sequence for sample in samples])
        reset_sequence = handoff.reset()
        self.assertEqual(sequence, reset_sequence)
        self.assertEqual(((), None, sequence), handoff.snapshot())
        sample, _ = handoff.add(50, "vo_odom", False)
        self.assertEqual(sequence + 1, sample.sequence)

    def test_snapshot_is_consistent_during_concurrent_updates(self):
        handoff = self.module.OdomInfoHandoff(capacity=10)
        done = threading.Event()
        errors = []

        def writer():
            for sequence in range(1, 1001):
                stamp = sequence * 1_000_000
                handoff.add(stamp, f"frame-{sequence}", False)
            done.set()

        thread = threading.Thread(target=writer)
        thread.start()
        while not done.is_set():
            samples, _, latest = handoff.snapshot()
            if len(samples) > 10:
                errors.append("snapshot exceeded bounded capacity")
            for sample in samples:
                if (sample.stamp_ns != sample.sequence * 1_000_000
                        or sample.frame_id != f"frame-{sample.sequence}"):
                    errors.append("torn sample")
            if samples and samples[-1].sequence > latest:
                errors.append("sample newer than snapshot sequence")
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)


if __name__ == "__main__":
    unittest.main()
