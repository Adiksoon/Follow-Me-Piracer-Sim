"""Select one valid suitcase point from a synchronized RGB-D detection bundle."""

from __future__ import annotations

import math


def _confidence(estimate) -> float:
    results = estimate.source_detection.results
    if not results:
        return 0.0
    scores = [float(result.hypothesis.score) for result in results]
    finite_scores = [score for score in scores if math.isfinite(score)]
    return max(finite_scores, default=0.0)


def select_target_point(estimates, image_width, image_height):
    """Select the nearest complete detection, with confidence and index for ties."""
    valid = []
    for estimate in estimates:
        if estimate.point is None:
            continue
        box = estimate.source_detection.bbox
        x, y = box.center.position.x, box.center.position.y
        if (x - box.size_x / 2 <= 0 or x + box.size_x / 2 >= image_width
                or y - box.size_y / 2 <= 0 or y + box.size_y / 2 >= image_height):
            continue  # A clipped box's center pixel can fall on the background.
        valid.append(estimate)
    if not valid:
        return None
    selected = min(
        valid,
        key=lambda estimate: (
            math.hypot(
                estimate.point.point.x,
                estimate.point.point.y,
                estimate.point.point.z,
            ),
            -_confidence(estimate),
            estimate.source_index,
        ),
    )
    return selected.point
