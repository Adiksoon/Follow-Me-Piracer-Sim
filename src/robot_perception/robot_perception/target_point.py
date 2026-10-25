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
    """Return the highest-confidence valid point, preserving source order on ties."""
    valid = [estimate for estimate in estimates if estimate.point is not None]
    if not valid:
        return None
    selected = max(valid, key=lambda estimate: (_confidence(estimate), -estimate.source_index))
    return selected.point
