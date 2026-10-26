"""Bounded timing trace around the existing native TF listener path."""

from collections import deque
from copy import deepcopy
from threading import Lock, get_ident
import time

from tf2_ros import Buffer, TransformListener


class InternalTfTrace:
    """Keep a small timing history until the first exhausted TF failure."""

    def __init__(self, capacity=512):
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
    def __init__(self, trace):
        super().__init__()
        self._trace = trace

    def set_transform(self, transform, authority):
        if not _is_native_transform(transform):
            return super().set_transform(transform, authority)
        span_id, _ = self._trace.begin(
            "native_tf_set_transform",
            thread_id=get_ident(),
            stamp_ns=_transform_stamp_ns(transform),
        )
        error = None
        try:
            return super().set_transform(transform, authority)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._trace.finish(span_id, error, success_phase="return")


class SearchTransformListener(TransformListener):
    def __init__(self, buffer, node, trace):
        self._trace = trace
        super().__init__(buffer, node)

    def callback(self, data):
        callback_entry_ns = time.monotonic_ns()
        transforms = [item for item in data.transforms if _is_native_transform(item)]
        if not transforms:
            return super().callback(data)
        span_id, _ = self._trace.begin(
            "native_tf_callback",
            event_monotonic_ns=callback_entry_ns,
            thread_id=get_ident(),
            stamp_ns_list=[_transform_stamp_ns(item) for item in transforms],
        )
        error = None
        try:
            return super().callback(data)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._trace.finish(span_id, error)
