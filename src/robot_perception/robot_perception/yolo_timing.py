"""Serialize timing events from one YOLO image callback."""

from __future__ import annotations

import json


_ROS_EVENT_FIELDS = (
    "source_stamp_ros_ns",
    "callback_entry_ros_ns",
    "infer_start_ros_ns",
    "infer_end_ros_ns",
    "detection_publish_ros_ns",
)
_MONOTONIC_EVENT_FIELDS = (
    "callback_entry_monotonic_ns",
    "infer_start_monotonic_ns",
    "infer_end_monotonic_ns",
    "detection_publish_monotonic_ns",
)


def build_timing_diagnostic(**events: int) -> str:
    """Return JSON with source/ROS stamps and monotonic durations in seconds."""
    required = set(_ROS_EVENT_FIELDS + _MONOTONIC_EVENT_FIELDS)
    if set(events) != required:
        missing = sorted(required - set(events))
        extra = sorted(set(events) - required)
        raise ValueError(f"invalid timing fields: missing={missing}, extra={extra}")
    if any(type(value) is not int or value < 0 for value in events.values()):
        raise ValueError("timing events must be non-negative integer nanoseconds")

    monotonic_values = [events[name] for name in _MONOTONIC_EVENT_FIELDS]
    if monotonic_values != sorted(monotonic_values):
        raise ValueError("monotonic event timestamps must be nondecreasing")

    payload = dict(events)
    payload["source_age_at_callback_entry_ros_s"] = (
        events["callback_entry_ros_ns"] - events["source_stamp_ros_ns"]
    ) / 1_000_000_000.0
    payload["source_age_at_infer_start_ros_s"] = (
        events["infer_start_ros_ns"] - events["source_stamp_ros_ns"]
    ) / 1_000_000_000.0
    payload["callback_to_infer_start_monotonic_s"] = (
        events["infer_start_monotonic_ns"]
        - events["callback_entry_monotonic_ns"]
    ) / 1_000_000_000.0
    payload["inference_duration_monotonic_s"] = (
        events["infer_end_monotonic_ns"]
        - events["infer_start_monotonic_ns"]
    ) / 1_000_000_000.0
    payload["infer_end_to_detection_publish_monotonic_s"] = (
        events["detection_publish_monotonic_ns"]
        - events["infer_end_monotonic_ns"]
    ) / 1_000_000_000.0
    return json.dumps(payload, allow_nan=False, sort_keys=True)
