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


def select_target_point(estimates):
    """Return the nearest valid 3D point, using confidence and source index for ties."""
    valid = [estimate for estimate in estimates if estimate.point is not None]
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
