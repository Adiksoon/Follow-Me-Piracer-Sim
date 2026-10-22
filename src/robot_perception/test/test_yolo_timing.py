import json

import pytest

from robot_perception.yolo_timing import build_timing_diagnostic


def _events():
    return {
        "source_stamp_ros_ns": 1_000_000_000,
        "callback_entry_ros_ns": 2_000_000_000,
        "infer_start_ros_ns": 2_050_000_000,
        "infer_end_ros_ns": 2_180_000_000,
        "detection_publish_ros_ns": 2_181_000_000,
        "callback_entry_monotonic_ns": 10_000_000_000,
        "infer_start_monotonic_ns": 10_050_000_000,
        "infer_end_monotonic_ns": 10_180_000_000,
        "detection_publish_monotonic_ns": 10_181_000_000,
    }


def test_timing_diagnostic_keeps_clocks_separate_and_computes_durations():
    payload = json.loads(build_timing_diagnostic(**_events()))

    assert payload["source_stamp_ros_ns"] == 1_000_000_000
    assert payload["callback_entry_ros_ns"] == 2_000_000_000
    assert payload["source_age_at_callback_entry_ros_s"] == pytest.approx(1.0)
    assert payload["source_age_at_infer_start_ros_s"] == pytest.approx(1.05)
    assert payload["callback_to_infer_start_monotonic_s"] == pytest.approx(0.05)
    assert payload["inference_duration_monotonic_s"] == pytest.approx(0.13)
    assert payload["infer_end_to_detection_publish_monotonic_s"] == pytest.approx(
        0.001
    )


def test_timing_diagnostic_rejects_nonmonotonic_monotonic_timestamps():
    events = _events()
    events["infer_end_monotonic_ns"] = events["infer_start_monotonic_ns"] - 1

    with pytest.raises(ValueError, match="nondecreasing"):
        build_timing_diagnostic(**events)
