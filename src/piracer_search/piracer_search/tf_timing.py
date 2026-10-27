"""Bounded timing trace around the existing native TF listener path."""

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock, Thread, get_ident, local
import time

from rclpy.executors import (
    ExternalShutdownException, MultiThreadedExecutor, SingleThreadedExecutor,
)
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener


class InternalTfTrace:
    """Keep a small timing history until the first exhausted TF failure."""

    def __init__(self, capacity=2048):
        self._capacity = capacity
        self._events = deque()
        self._evicted_count = 0
        self._next_span_id = 0
        self._sequences = {}
        self._open_spans = {}
        self._lock = Lock()

    def begin(self, kind, event_monotonic_ns=None, *, retain_entry=True, **details):
        event_time_ns = (
            time.monotonic_ns()
            if event_monotonic_ns is None else event_monotonic_ns
        )
        with self._lock:
            sequence = self._sequences.get(kind, 0) + 1
            self._sequences[kind] = sequence
            self._next_span_id += 1
            span_id = self._next_span_id
            event = {
                "kind": kind,
                "phase": "entry",
                "sequence": sequence,
                "span_id": span_id,
                "monotonic_ns": event_time_ns,
                **details,
            }
            self._open_spans[span_id] = dict(event)
            if retain_entry:
                self._append_locked(event)
        return span_id, sequence

    def update_span(self, span_id, **details):
        """Attach small metadata to an open span without retaining messages."""
        with self._lock:
            opened = self._open_spans.get(span_id)
            if opened is None:
                return False
            opened.update(details)
            return True

    def finish(self, span_id, error=None, success_phase="exit",
               event_monotonic_ns=None, **details):
        event_time_ns = (
            time.monotonic_ns()
            if event_monotonic_ns is None else event_monotonic_ns
        )
        with self._lock:
            opened = self._open_spans.pop(span_id, None)
            if opened is None:
                return
            event = {
                key: value for key, value in opened.items()
                if key not in ("phase", "monotonic_ns")
            }
            event.update(details)
            event["phase"] = "exception" if error is not None else success_phase
            event["monotonic_ns"] = event_time_ns
            if error is not None:
                event["exception_class"] = type(error).__name__
                event["exception_message"] = str(error)
            self._append_locked(event)

    def record(self, kind, **details):
        with self._lock:
            sequence = self._sequences.get(kind, 0) + 1
            self._sequences[kind] = sequence
            event = {
                "kind": kind,
                "phase": "event",
                "sequence": sequence,
                "monotonic_ns": time.monotonic_ns(),
                **details,
            }
            self._append_locked(event)

    def counts(self):
        with self._lock:
            counts = {
                "odom_info_callback": 0,
                "native_tf_callback": 0,
                "localization_validation": 0,
            }
            counts.update(self._sequences)
            return counts, self._evicted_count

    def _append_locked(self, event):
        if len(self._events) >= self._capacity:
            self._events.popleft()
            self._evicted_count += 1
        self._events.append(event)

    def capture(self):
        started_ns = time.monotonic_ns()
        with self._lock:
            events = deepcopy(list(self._events))
            open_spans = deepcopy(list(self._open_spans.values()))
            evicted_count = self._evicted_count
            oldest_ns = min(
                (event["monotonic_ns"] for event in events), default=None
            )
        ended_ns = time.monotonic_ns()
        return {
            "capture_start_monotonic_ns": started_ns,
            "capture_end_monotonic_ns": ended_ns,
            "evicted_count": evicted_count,
            "oldest_retained_monotonic_ns": oldest_ns,
            "open_spans": open_spans,
            "events": events,
        }


class OdomInfoTraceContext:
    """Thread-local handle for annotating the active OdomInfo span."""

    def __init__(self, trace):
        self._trace = trace
        self._local = local()

    def activate(self, span_id):
        previous = getattr(self._local, "span_id", None)
        self._local.span_id = span_id
        return previous

    def restore(self, previous):
        if previous is None:
            self.clear()
        else:
            self._local.span_id = previous

    def clear(self):
        if hasattr(self._local, "span_id"):
            del self._local.span_id

    def annotate(self, **details):
        span_id = getattr(self._local, "span_id", None)
        if span_id is None:
            return False
        try:
            return self._trace.update_span(span_id, **details)
        except Exception:
            return False


class OdomInfoTraceExecutor(MultiThreadedExecutor):
    """Instrument one subscription while preserving Humble executor behavior.

    This is selected only when localization tracing is enabled. The measured
    take interval surrounds the inherited executor hook, not DDS arrival and
    not message deserialization in isolation.
    """

    _MAX_PENDING_SPANS = 8

    def __init__(self, subscription, trace, *, num_threads=2):
        super().__init__(num_threads=num_threads)
        self._odom_info_subscription = subscription
        self._trace = trace
        self.odom_info_trace_context = OdomInfoTraceContext(trace)
        self._pending_spans = OrderedDict()
        self._pending_spans_lock = Lock()
        self._correlation_loss_count = 0

    @staticmethod
    def _message_metadata(message):
        stamp = message.header.stamp
        return {
            "header_stamp_ns": int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec),
            "frame_id": str(message.header.frame_id),
            "lost": bool(message.lost),
        }

    def _finish_span_safely(self, span_id, *, phase, **details):
        if span_id is None:
            return
        try:
            self._trace.finish(span_id, success_phase=phase, **details)
        except Exception:
            pass

    def _record_correlation_loss(self, reason, span_id=None):
        with self._pending_spans_lock:
            self._correlation_loss_count += 1
            loss_count = self._correlation_loss_count
        if span_id is not None:
            self._finish_span_safely(
                span_id, phase="correlation_lost",
                correlation_lost=True, correlation_loss_reason=reason,
            )
        try:
            self._trace.record(
                "odom_info_trace_correlation_loss",
                reason=reason,
                span_id=span_id,
                correlation_loss_count=loss_count,
            )
        except Exception:
            pass

    def _remember_span(self, message, span_id):
        identity = id(message)
        evicted = None
        with self._pending_spans_lock:
            previous = self._pending_spans.pop(identity, None)
            self._pending_spans[identity] = span_id
            if len(self._pending_spans) > self._MAX_PENDING_SPANS:
                evicted = self._pending_spans.popitem(last=False)[1]
        if previous is not None:
            self._record_correlation_loss("message_identity_reused", previous)
        if evicted is not None:
            self._record_correlation_loss("pending_span_capacity", evicted)

    def _take_subscription(self, sub):
        if sub is not self._odom_info_subscription:
            return super()._take_subscription(sub)

        hook_entry_ns = time.monotonic_ns()
        span_id = None
        try:
            span_id, _ = self._trace.begin(
                "odom_info_callback",
                event_monotonic_ns=hook_entry_ns,
                retain_entry=False,
                executor_take_hook_entry_monotonic_ns=hook_entry_ns,
                take_interval_semantics=(
                    "inherited executor take hook; includes handle acquisition "
                    "and conversion; not DDS arrival or conversion-only time"
                ),
            )
        except Exception:
            pass

        take_start_ns = time.monotonic_ns()
        try:
            message = super()._take_subscription(sub)
        except BaseException as exc:
            take_end_ns = time.monotonic_ns()
            self._finish_span_safely(
                span_id, phase="take_exception",
                error=exc,
                take_started_monotonic_ns=take_start_ns,
                take_finished_monotonic_ns=take_end_ns,
                take_duration_ns=take_end_ns - take_start_ns,
                message_taken=False,
            )
            raise
        take_end_ns = time.monotonic_ns()

        if message is None:
            self._finish_span_safely(
                span_id, phase="take_empty",
                take_started_monotonic_ns=take_start_ns,
                take_finished_monotonic_ns=take_end_ns,
                take_duration_ns=take_end_ns - take_start_ns,
                message_taken=False,
            )
            return None

        if span_id is not None:
            details = {
                "take_started_monotonic_ns": take_start_ns,
                "take_finished_monotonic_ns": take_end_ns,
                "take_duration_ns": take_end_ns - take_start_ns,
                "message_taken": True,
            }
            try:
                details.update(self._message_metadata(message))
            except Exception as exc:
                details["message_metadata_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            try:
                self._trace.update_span(span_id, **details)
            except Exception:
                pass
            self._remember_span(message, span_id)
        else:
            self._record_correlation_loss("take_span_unavailable")
        return message

    def _take_span_for(self, message):
        with self._pending_spans_lock:
            return self._pending_spans.pop(id(message), None)

    async def _execute_subscription(self, sub, message):
        if sub is not self._odom_info_subscription or message is None:
            return await super()._execute_subscription(sub, message)

        span_id = self._take_span_for(message)
        callback_start_ns = time.monotonic_ns()
        previous_context = None
        if span_id is not None:
            try:
                self._trace.update_span(
                    span_id,
                    callback_executor_entry_monotonic_ns=callback_start_ns,
                )
            except Exception:
                pass
            previous_context = self.odom_info_trace_context.activate(span_id)
        else:
            self._record_correlation_loss("callback_without_take_span")

        callback_error = None
        try:
            return await super()._execute_subscription(sub, message)
        except BaseException as exc:
            callback_error = exc
            raise
        finally:
            callback_end_ns = time.monotonic_ns()
            if span_id is not None:
                self.odom_info_trace_context.restore(previous_context)
                self._finish_span_safely(
                    span_id,
                    phase="callback_return",
                    error=callback_error,
                    event_monotonic_ns=callback_end_ns,
                    callback_executor_exit_monotonic_ns=callback_end_ns,
                    callback_duration_ns=callback_end_ns - callback_start_ns,
                )

    def finish_pending_odom_info_spans(self):
        """Close correlations left after the executor has fully shut down."""
        with self._pending_spans_lock:
            pending = list(self._pending_spans.values())
            self._pending_spans.clear()
        for span_id in pending:
            self._record_correlation_loss("shutdown_before_callback", span_id)

    def odom_info_trace_stats(self):
        with self._pending_spans_lock:
            pending = len(self._pending_spans)
            correlation_losses = self._correlation_loss_count
        return {
            "pending_span_count": pending,
            "correlation_loss_count": correlation_losses,
        }


@dataclass(frozen=True)
class OdomInfoSample:
    """Small immutable handoff record; never retain the large ROS message."""

    sequence: int
    stamp_ns: int
    frame_id: str
    lost: bool
    lost_stamp_floor_ns: int = 0


class OdomInfoHandoff:
    """Bounded, thread-safe handoff from OdomInfo reception to Search decisions."""

    def __init__(self, capacity=10):
        self._lock = Lock()
        self._samples = deque(maxlen=capacity)
        self._sequence = 0
        self._pending_loss = None
        self._barrier_stamp_ns = None

    def add(self, stamp_ns, frame_id, lost, *, include_commit_time=False):
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            stamp_ns = int(stamp_ns)
            sample = OdomInfoSample(
                sequence=sequence,
                stamp_ns=stamp_ns,
                frame_id=str(frame_id),
                lost=bool(lost),
            )
            if sample.lost:
                previous = self._pending_loss
                self._pending_loss = OdomInfoSample(
                    sequence=sequence,
                    stamp_ns=stamp_ns,
                    frame_id=sample.frame_id,
                    lost=True,
                    lost_stamp_floor_ns=max(
                        stamp_ns if stamp_ns > 0 else 0,
                        previous.lost_stamp_floor_ns if previous else 0,
                    ),
                )
                # A loss clears earlier good samples, while later good samples
                # remain available for recovery after Search sees this event.
                self._samples.clear()
                disposition = "lost"
            elif stamp_ns == 0:
                disposition = "dropped_zero_stamp"
            elif (self._barrier_stamp_ns is not None
                  and stamp_ns <= self._barrier_stamp_ns):
                disposition = "dropped_invalidation_barrier"
            else:
                self._samples.append(sample)
                disposition = "queued"
            if include_commit_time:
                return sample, disposition, time.monotonic_ns()
            return sample, disposition

    def snapshot(self):
        with self._lock:
            return tuple(self._samples), self._pending_loss, self._sequence

    def acknowledge_loss(self, sequence):
        with self._lock:
            if (self._pending_loss is not None
                    and self._pending_loss.sequence == sequence):
                self._pending_loss = None

    def invalidate(self, barrier_stamp_ns, *, after_sequence):
        with self._lock:
            self._barrier_stamp_ns = barrier_stamp_ns
            self._samples = deque(
                (sample for sample in self._samples
                 if sample.sequence > after_sequence
                 and (barrier_stamp_ns is None
                      or sample.stamp_ns > barrier_stamp_ns)),
                maxlen=self._samples.maxlen,
            )

    def reset(self):
        with self._lock:
            self._samples.clear()
            self._pending_loss = None
            self._barrier_stamp_ns = None
            return self._sequence

    def set_barrier(self, barrier_stamp_ns):
        with self._lock:
            self._barrier_stamp_ns = barrier_stamp_ns


def _is_native_transform(transform):
    return (
        transform.header.frame_id.strip("/") == "vo_odom"
        and transform.child_frame_id.strip("/") == "base_link"
    )


def _transform_stamp_ns(transform):
    stamp = transform.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class SearchTfBuffer(Buffer):
    def __init__(self, trace, diagnostics_enabled=False, error_reporter=None):
        super().__init__()
        self._trace = trace
        self._diagnostics_enabled = diagnostics_enabled
        self._error_reporter = error_reporter

    def set_transform(self, transform, authority):
        if not _is_native_transform(transform):
            return super().set_transform(transform, authority)
        if self._trace is None:
            return super().set_transform(transform, authority)
        span_id = None
        try:
            span_id, _ = self._trace.begin(
                "native_tf_set_transform",
                thread_id=get_ident(),
                stamp_ns=_transform_stamp_ns(transform),
                target_frame=transform.header.frame_id,
                source_frame=transform.child_frame_id,
                return_semantics=(
                    "delegated operation finished; acceptance is not implied"
                ),
            )
        except Exception as exc:
            self._report_trace_error("set_transform_begin", exc)
        error = None
        try:
            return super().set_transform(transform, authority)
        except BaseException as exc:
            error = exc
            raise
        finally:
            if span_id is not None:
                try:
                    self._trace.finish(span_id, error, success_phase="return")
                except Exception as exc:
                    self._report_trace_error("set_transform_finish", exc)
            if error is None and self._diagnostics_enabled:
                self._post_return_snapshot(transform, span_id)

    def _report_trace_error(self, operation, exc):
        if self._error_reporter is not None:
            try:
                self._error_reporter(operation, exc)
            except Exception:
                pass

    def _post_return_snapshot(self, transform, span_id):
        try:
            started_ns = time.monotonic_ns()
        except Exception as exc:
            self._report_trace_error("post_return_snapshot_start", exc)
            return
        try:
            latest = self.lookup_transform(
                transform.header.frame_id, transform.child_frame_id, Time()
            )
        except Exception as exc:
            details = {
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
            }
        else:
            details = {
                "target_frame": latest.header.frame_id,
                "source_frame": latest.child_frame_id,
                "stamp_ns": _transform_stamp_ns(latest),
            }
        try:
            self._trace.record(
                "native_tf_set_transform_snapshot",
                snapshot_kind="post_return_snapshot",
                snapshot_started_monotonic_ns=started_ns,
                snapshot_ended_monotonic_ns=time.monotonic_ns(),
                incoming_stamp_ns=_transform_stamp_ns(transform),
                setter_span_id=span_id,
                target_frame=details.pop("target_frame", transform.header.frame_id),
                source_frame=details.pop("source_frame", transform.child_frame_id),
                meaning="read-only latest lookup after setter return; not acceptance",
                **details,
            )
        except Exception as exc:
            self._report_trace_error("post_return_snapshot_record", exc)


class SearchTransformListener(TransformListener):
    def __init__(self, buffer, node, trace, error_reporter=None):
        self._trace = trace
        self._diagnostics_enabled = trace is not None
        self._error_reporter = error_reporter
        super().__init__(buffer, node)

    def callback(self, data):
        if not self._diagnostics_enabled:
            return super().callback(data)
        try:
            callback_entry_ns = time.monotonic_ns()
            transforms = [
                item for item in data.transforms if _is_native_transform(item)
            ]
        except Exception as exc:
            self._report_trace_error("listener_callback_prepare", exc)
            return super().callback(data)
        if not transforms:
            return super().callback(data)
        span_id = None
        try:
            span_id, _ = self._trace.begin(
                "native_tf_callback",
                event_monotonic_ns=callback_entry_ns,
                thread_id=get_ident(),
                stamp_ns_list=[_transform_stamp_ns(item) for item in transforms],
                frame_pairs=[
                    {"target_frame": item.header.frame_id,
                     "source_frame": item.child_frame_id}
                    for item in transforms
                ],
            )
        except Exception as exc:
            self._report_trace_error("listener_callback_begin", exc)
        error = None
        try:
            return super().callback(data)
        except BaseException as exc:
            error = exc
            raise
        finally:
            if span_id is not None:
                try:
                    self._trace.finish(span_id, error)
                except Exception as exc:
                    self._report_trace_error("listener_callback_finish", exc)

    def _report_trace_error(self, operation, exc):
        reporter = getattr(self, "_error_reporter", None)
        if reporter is not None:
            try:
                reporter(operation, exc)
            except Exception:
                pass


class SearchTfReceiver:
    """Own a TF-only node/executor; share only the existing thread-safe buffer."""

    def __init__(self, buffer, trace, error_reporter, *, context, namespace,
                 use_sim_time, odom_info_wakeup):
        self.executor = self.listener = self.thread = None
        self._closed = False
        self._trace = trace
        self._error_reporter = error_reporter
        self._odom_info_wakeup = odom_info_wakeup
        self._odom_info_handoff = OdomInfoHandoff()
        # Do not inherit the launch's __node:=piracer_search remapping.
        self.node = Node(
            "piracer_search_tf_listener", context=context, namespace=namespace,
            use_global_arguments=False, enable_rosout=False,
            start_parameter_services=False,
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        try:
            self.executor = SingleThreadedExecutor(context=context)
            self.executor.add_node(self.node)
            self.listener = SearchTransformListener(
                buffer, self.node, trace, error_reporter
            )
            self.thread = Thread(
                target=self._spin, name="piracer-search-tf", daemon=False
            )
        except BaseException:
            self.close()
            raise

    def start(self):
        self.thread.start()

    def _spin(self):
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass  # The shared ROS context may already be shut down by SIGINT.

    @staticmethod
    def _stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def receive_odom_info(self, message):
        """Metadata utility called by SearchNode, outside the TF executor."""
        if self._closed:
            return None
        stamp_ns = self._stamp_ns(message)
        frame_id = message.header.frame_id
        lost = bool(message.lost)
        if self._trace is not None:
            sample, disposition, commit_monotonic_ns = (
                self._odom_info_handoff.add(
                    stamp_ns, frame_id, lost, include_commit_time=True
                )
            )
        else:
            sample, disposition = self._odom_info_handoff.add(
                stamp_ns, frame_id, lost
            )
            commit_monotonic_ns = None
        if disposition in ("queued", "lost") and not self._closed:
            try:
                self._odom_info_wakeup.trigger()
            except Exception as exc:
                self._report_trace_error("odom_info_wakeup", exc)
        return sample, disposition, commit_monotonic_ns

    def odom_info_snapshot(self):
        return self._odom_info_handoff.snapshot()

    def acknowledge_odom_info_loss(self, sequence):
        self._odom_info_handoff.acknowledge_loss(sequence)

    def invalidate_odom_info(self, barrier_stamp_ns, *, after_sequence):
        self._odom_info_handoff.invalidate(
            barrier_stamp_ns, after_sequence=after_sequence
        )

    def reset_odom_info(self):
        return self._odom_info_handoff.reset()

    def set_odom_info_barrier(self, barrier_stamp_ns):
        self._odom_info_handoff.set_barrier(barrier_stamp_ns)

    def _report_trace_error(self, operation, exc):
        if self._error_reporter is not None:
            try:
                self._error_reporter(operation, exc)
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            self.executor.shutdown()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join()
        if self.listener is not None:
            self.listener.unregister()
        self.node.executor = None
        self.node.destroy_node()
