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


def _point(x, y, z):
    return SimpleNamespace(point=SimpleNamespace(x=x, y=y, z=z))


def test_nearest_valid_point_wins_over_higher_confidence_and_invalid_depth():
    nearest = _point(0.0, 0.0, 2.0)
    selected = select_target_point(
        [
            _estimate(0, 0.99, _point(0.0, 0.0, 4.0)),
            _estimate(1, 1.0, None),  # Invalid center depth has no point.
            _estimate(2, 0.2, nearest),
        ]
    )
    assert selected is nearest


def test_nearest_uses_full_three_dimensional_distance():
    farther_on_z_axis = _point(0.0, 0.0, 3.0)
    nearest_in_3d = _point(2.0, 0.0, 2.0)

    selected = select_target_point(
        [
            _estimate(0, 0.5, farther_on_z_axis),
            _estimate(1, 0.5, nearest_in_3d),
        ]
    )

    assert selected is nearest_in_3d  # sqrt(8) < 3 although z is smaller.


def test_confidence_breaks_exact_distance_tie():
    lower_confidence = _point(1.0, 2.0, 2.0)
    higher_confidence = _point(0.0, 0.0, 3.0)

    selected = select_target_point(
        [
            _estimate(0, 0.4, lower_confidence),
            _estimate(1, 0.8, higher_confidence),
        ]
    )

    assert selected is higher_confidence


def test_exact_distance_and_confidence_tie_uses_source_index_deterministically():
    later_source = _point(0.0, 0.0, 3.0)
    earlier_source = _point(3.0, 0.0, 0.0)
    estimates = [
        _estimate(8, 0.8, later_source),
        _estimate(3, 0.8, earlier_source),
    ]

    assert select_target_point(estimates) is earlier_source
    assert select_target_point(list(reversed(estimates))) is earlier_source


def test_returns_none_when_no_estimate_has_a_valid_point():
    assert select_target_point(
        [_estimate(0, 0.9, None), _estimate(1, 0.7, None)]
    ) is None
