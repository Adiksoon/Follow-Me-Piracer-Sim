"""Offline tests for bounded OdomInfo receive-path timing traces."""

import asyncio
import time
import types
import unittest
from collections import deque
from unittest.mock import patch

import test_search_node as fixtures


class OdomInfoTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SearchNodeCallbackTest.setUpClass()
        cls.search_module = fixtures.SearchNodeCallbackTest.module
        cls.tf_module = fixtures.SearchNodeCallbackTest.tf_module
        cls.executor_base = cls.search_module.MultiThreadedExecutor

    def message(self, stamp_ns=9_500_000_000, *, lost=False):
        sec, nanosec = divmod(stamp_ns, 1_000_000_000)
        return types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=sec, nanosec=nanosec),
                frame_id="vo_odom",
            ),
            lost=lost,
        )

    def subscription(self, message=None, error=None, callback=None):
        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def take_message(self, _msg_type, _raw):
                if error is not None:
                    raise error
                return (message, None) if message is not None else None

        return types.SimpleNamespace(
            handle=Handle(), msg_type=object, raw=False,
            callback=callback or (lambda _message: None),
        )

    def executor(self, subscription, trace=None):
        trace = trace or self.tf_module.InternalTfTrace()
        executor = self.tf_module.OdomInfoTraceExecutor(
            subscription, trace, num_threads=2
        )
        return executor, trace

    def install_executor_hooks(self):
        def take_subscription(_base, sub):
            with sub.handle:
                result = sub.handle.take_message(sub.msg_type, sub.raw)
            return result[0] if result is not None else None

        async def execute_subscription(_base, sub, message):
            if message:
                return sub.callback(message)
            return None

        self.take_patch = patch.object(
            self.executor_base, "_take_subscription", take_subscription,
            create=True,
        )
        self.execute_patch = patch.object(
            self.executor_base, "_execute_subscription", execute_subscription,
            create=True,
        )
        self.take_patch.start()
        self.execute_patch.start()
        self.addCleanup(self.take_patch.stop)
        self.addCleanup(self.execute_patch.stop)

    def test_take_and_full_callback_are_timed_and_correlated(self):
        self.install_executor_hooks()
        trace = self.tf_module.InternalTfTrace()
        message = self.message(lost=True)
        callback_marks = {}
        context = {"active": None}

        def callback(received):
            callback_marks["python_entry_ns"] = time.monotonic_ns()
            context["active"].annotate(
                callback_python_entry_monotonic_ns=callback_marks["python_entry_ns"],
                handoff_header_stamp_ns=9_500_000_000,
                handoff_frame_id=received.header.frame_id,
                handoff_lost=received.lost,
                handoff_sequence=3,
            )

        sub = self.subscription(message, callback=callback)
        executor, _ = self.executor(sub, trace)
        context["active"] = executor.odom_info_trace_context
        taken = executor._take_subscription(sub)
        self.assertIs(message, taken)
        asyncio.run(executor._execute_subscription(sub, taken))

        events = trace.capture()["events"]
        self.assertEqual(1, len(events))
        event, = events
        self.assertEqual("odom_info_callback", event["kind"])
        self.assertEqual("callback_return", event["phase"])
        self.assertTrue(event["message_taken"])
        self.assertEqual(9_500_000_000, event["header_stamp_ns"])
        self.assertEqual("vo_odom", event["frame_id"])
        self.assertTrue(event["lost"])
        self.assertEqual(9_500_000_000, event["handoff_header_stamp_ns"])
        self.assertEqual(3, event["handoff_sequence"])
        self.assertLessEqual(
            event["executor_take_hook_entry_monotonic_ns"],
            event["take_started_monotonic_ns"],
        )
        self.assertLessEqual(
            event["take_started_monotonic_ns"],
            event["take_finished_monotonic_ns"],
        )
        self.assertLessEqual(
            event["take_finished_monotonic_ns"],
            event["callback_executor_entry_monotonic_ns"],
        )
        self.assertLessEqual(
            event["callback_executor_entry_monotonic_ns"],
            event["callback_python_entry_monotonic_ns"],
        )
        self.assertLessEqual(
            event["callback_python_entry_monotonic_ns"],
            event["callback_executor_exit_monotonic_ns"],
        )
        self.assertGreaterEqual(event["take_duration_ns"], 0)
        self.assertGreaterEqual(event["callback_duration_ns"], 0)
        self.assertIn("not DDS arrival", event["take_interval_semantics"])
        self.assertFalse(hasattr(context["active"]._local, "span_id"))
        self.assertEqual(
            {"pending_span_count": 0, "correlation_loss_count": 0},
            executor.odom_info_trace_stats(),
        )

    def test_empty_take_and_take_exception_are_closed(self):
        self.install_executor_hooks()
        empty_sub = self.subscription()
        executor, trace = self.executor(empty_sub)
        self.assertIsNone(executor._take_subscription(empty_sub))
        empty_event, = trace.capture()["events"]
        self.assertEqual("take_empty", empty_event["phase"])
        self.assertFalse(empty_event["message_taken"])
        self.assertEqual([], trace.capture()["open_spans"])

        error_sub = self.subscription(error=RuntimeError("take failed"))
        error_executor, error_trace = self.executor(error_sub)
        with self.assertRaisesRegex(RuntimeError, "take failed"):
            error_executor._take_subscription(error_sub)
        error_event, = error_trace.capture()["events"]
        self.assertEqual("exception", error_event["phase"])
        self.assertEqual("RuntimeError", error_event["exception_class"])
        self.assertFalse(error_event["message_taken"])
        self.assertEqual([], error_trace.capture()["open_spans"])

    def test_callback_exception_closes_span_and_restores_trace_context(self):
        self.install_executor_hooks()
        trace = self.tf_module.InternalTfTrace()
        message = self.message()

        def callback(_message):
            raise ValueError("callback failed")

        sub = self.subscription(message, callback=callback)
        executor, _ = self.executor(sub, trace)
        taken = executor._take_subscription(sub)
        with self.assertRaisesRegex(ValueError, "callback failed"):
            asyncio.run(executor._execute_subscription(sub, taken))
        event, = trace.capture()["events"]
        self.assertEqual("exception", event["phase"])
        self.assertEqual("ValueError", event["exception_class"])
        self.assertEqual([], trace.capture()["open_spans"])
        self.assertFalse(hasattr(executor.odom_info_trace_context._local, "span_id"))

    def test_pending_correlation_is_bounded_and_closed_at_shutdown(self):
        self.install_executor_hooks()
        trace = self.tf_module.InternalTfTrace()
        messages = [self.message(9_000_000_000 + index) for index in range(9)]
        subs = [self.subscription(message) for message in messages]
        executor, _ = self.executor(subs[0], trace)
        for sub in subs:
            executor._odom_info_subscription = sub
            executor._take_subscription(sub)

        self.assertEqual(
            {"pending_span_count": 8, "correlation_loss_count": 1},
            executor.odom_info_trace_stats(),
        )
        executor.finish_pending_odom_info_spans()
        self.assertEqual(
            {"pending_span_count": 0, "correlation_loss_count": 9},
            executor.odom_info_trace_stats(),
        )
        events = trace.capture()["events"]
        self.assertEqual(9, len([e for e in events if e["kind"] == "odom_info_callback"]))
        self.assertEqual(
            9,
            len([e for e in events if e["kind"] == "odom_info_trace_correlation_loss"]),
        )

    def test_handoff_preserves_two_tuple_default_and_lost_event(self):
        handoff = self.tf_module.OdomInfoHandoff(capacity=4)
        self.assertEqual(2, len(handoff.add(10, "vo_odom", False)))
        before = time.monotonic_ns()
        loss, disposition, commit_ns = handoff.add(
            11, "vo_odom", True, include_commit_time=True
        )
        after = time.monotonic_ns()
        self.assertEqual("lost", disposition)
        self.assertGreaterEqual(commit_ns, before)
        self.assertLessEqual(commit_ns, after)
        handoff.add(12, "vo_odom", False)
        samples, pending_loss, latest_sequence = handoff.snapshot()
        self.assertEqual(3, latest_sequence)
        self.assertEqual([3], [sample.sequence for sample in samples])
        self.assertEqual(loss.sequence, pending_loss.sequence)
        self.assertTrue(pending_loss.lost)
        handoff.acknowledge_loss(loss.sequence)
        self.assertIsNone(handoff.snapshot()[1])

    def test_validator_trace_captures_handoff_and_search_queue_before_invalidation(self):
        node = fixtures.SearchNodeCallbackTest().make_localization_node()
        node._localization_trace_enabled = True
        node._internal_tf_trace = self.tf_module.InternalTfTrace()
        node._active_localization_trace = {"lookup_results": []}
        node._localization_available = True
        node._last_processed_odom_info_sequence = 4
        node._tf_snapshot_attempted = False
        node._odom_info = deque([self.message(9_000_000_000)])
        node._odom_info[0].sequence = 5

        handoff = self.tf_module.OdomInfoHandoff()
        handoff.add(9_100_000_000, "vo_odom", False)
        node._tf_receiver = types.SimpleNamespace(
            odom_info_snapshot=handoff.snapshot,
            invalidate_odom_info=handoff.invalidate,
        )

        result = node._localization_pose_impl("map")
        self.assertIsNone(result)
        event = next(
            item for item in node._internal_tf_trace.capture()["events"]
            if item["kind"] == "odom_info_pre_invalidation_state"
        )
        self.assertEqual(5, event["selected_odom_sequence"])
        self.assertEqual(9_000_000_000, event["selected_odom_stamp_ns"])
        self.assertEqual(1, len(event["handoff_samples"]))
        self.assertEqual(1, len(event["search_odom_info_queue"]))
        self.assertEqual(4, event["search_last_processed_odom_info_sequence_watermark"])
        self.assertGreaterEqual(
            event["handoff_snapshot_end_monotonic_ns"],
            event["handoff_snapshot_start_monotonic_ns"],
        )
        self.assertEqual("stale OdomInfo", node._active_localization_trace["reason"])


if __name__ == "__main__":
    unittest.main()
