import math

import pytest

from piracer_state_manager.follow_control import (
    ACKERMANN_MAX_STEER_RAD,
    ACKERMANN_PLUGIN_MAX_SPEED_MPS,
    ACKERMANN_WHEELBASE_M,
    FollowPController,
)


def test_distance_p_drives_only_beyond_accepted_band_and_is_bounded():
    controller = FollowPController(distance_kp=0.1, max_speed_mps=0.05)

    assert controller.update(2.25, 0.0).linear_x == 0.0
    assert controller.update(2.5, 0.0).linear_x == 0.0
    assert controller.update(1.9, 0.0).linear_x == 0.0
    assert controller.update(2.6, 0.0).linear_x == pytest.approx(0.01)
    assert controller.update(3.0, 0.0).linear_x == pytest.approx(0.05)
    assert controller.update(5.0, 0.0).linear_x == 0.05


def test_direction_p_turns_toward_target_and_respects_ackermann_limit():
    controller = FollowPController(direction_kp=2.0, max_speed_mps=0.05)

    left = controller.update(3.0, 1.0)
    right = controller.update(3.0, -1.0)
    assert left.linear_x > 0.0
    assert left.steering_angle_rad > 0.0
    assert left.angular_z > 0.0
    assert right.steering_angle_rad < 0.0
    assert right.angular_z < 0.0
    assert abs(left.steering_angle_rad) <= ACKERMANN_MAX_STEER_RAD
    assert abs(left.linear_x) <= ACKERMANN_PLUGIN_MAX_SPEED_MPS

    # The existing adapter maps this yaw-rate back to the same steering angle.
    recovered_angle = math.atan(
        ACKERMANN_WHEELBASE_M * left.angular_z / abs(left.linear_x)
    )
    assert recovered_angle == pytest.approx(left.steering_angle_rad)


def test_ackermann_does_not_turn_in_place_or_reverse_for_close_or_rear_target():
    controller = FollowPController()
    for point in ((1.0, 0.0), (-3.0, 0.0), (-3.0, 2.0)):
        command = controller.update(*point)
        assert command.linear_x == 0.0
        assert command.angular_z == 0.0
        assert command.steering_angle_rad == 0.0


def test_nonfinite_target_and_plugin_limit_overrides_are_rejected():
    with pytest.raises(ValueError):
        FollowPController().update(float("nan"), 0.0)
    with pytest.raises(ValueError):
        FollowPController(max_speed_mps=0.21)
    with pytest.raises(ValueError):
        FollowPController(max_steer_rad=0.6)
