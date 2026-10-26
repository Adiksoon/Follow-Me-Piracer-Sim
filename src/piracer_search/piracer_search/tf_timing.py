"""Bounded timing trace around the existing native TF listener path."""

from collections import deque
from copy import deepcopy
from threading import Lock, get_ident
import time

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

    def begin(self, kind, event_monotonic_ns=None, **details):
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
            self._append_locked(event)
        return span_id, sequence

    def finish(self, span_id, error=None, success_phase="exit"):
        event_time_ns = time.monotonic_ns()
        with self._lock:
            opened = self._open_spans.pop(span_id, None)
            if opened is None:
                return
            event = {
                key: value for key, value in opened.items()
                if key not in ("phase", "monotonic_ns")
            }
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
