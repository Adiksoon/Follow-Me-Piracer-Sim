from types import SimpleNamespace

from robot_perception.target_point import select_target_point


def _estimate(index, score, point):
    result = SimpleNamespace(hypothesis=SimpleNamespace(score=score))
    detection = SimpleNamespace(results=[result])
    return SimpleNamespace(
        source_index=index,
        source_detection=detection,
        point=point,
    )


def test_selects_highest_confidence_valid_point_and_keeps_order_on_tie():
    low = object()
    first_tie = object()
    second_tie = object()
    selected = select_target_point(
        [
            _estimate(0, 0.9, None),
            _estimate(1, 0.7, low),
            _estimate(2, 0.8, first_tie),
            _estimate(3, 0.8, second_tie),
        ]
    )
    assert selected is first_tie


def test_returns_none_when_no_estimate_has_a_valid_point():
    assert select_target_point([_estimate(0, 0.9, None)]) is None
