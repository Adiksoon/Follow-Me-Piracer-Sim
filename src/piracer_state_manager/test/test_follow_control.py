import math

import pytest

from piracer_state_manager.follow_control import (
    ACKERMANN_MAX_STEER_RAD,
    ACKERMANN_PLUGIN_MAX_SPEED_MPS,
    ACKERMANN_WHEELBASE_M,
    FollowPController,
)


def test_distance_p_uses_2_25_m_setpoint_without_reverse_and_is_bounded():
    controller = FollowPController(distance_kp=0.1, max_speed_mps=0.05)

    assert controller.update(2.25, 0.0).linear_x == 0.0
    assert controller.update(2.3, 0.0).linear_x == pytest.approx(0.005)
    assert controller.update(2.5, 0.0).linear_x == pytest.approx(0.025)
    assert controller.update(2.0, 0.0).linear_x == 0.0
    assert controller.update(1.9, 0.0).linear_x == 0.0
    assert controller.update(3.0, 0.0).linear_x == 0.05
    assert controller.update(5.0, 0.0).linear_x == 0.05


def test_direction_p_turns_left_right_and_center_with_ackermann_signs():
    controller = FollowPController(direction_kp=2.0, max_speed_mps=0.05)

    left = controller.update(3.0, 1.0)
    right = controller.update(3.0, -1.0)
    center = controller.update(3.0, 0.0)
    assert left.linear_x > 0.0
    assert left.steering_angle_rad > 0.0
    assert left.angular_z > 0.0
    assert right.steering_angle_rad < 0.0
    assert right.angular_z < 0.0
    assert center.steering_angle_rad == 0.0
    assert center.angular_z == 0.0
    assert abs(left.steering_angle_rad) <= ACKERMANN_MAX_STEER_RAD
    assert abs(left.linear_x) <= ACKERMANN_PLUGIN_MAX_SPEED_MPS

    # The existing adapter maps this yaw-rate back to the same steering angle.
    for command in (left, right):
        recovered_angle = math.atan(
            ACKERMANN_WHEELBASE_M * command.angular_z / abs(command.linear_x)
        )
        assert recovered_angle == pytest.approx(command.steering_angle_rad)


def test_distance_and_direction_p_calculations_are_independent():
    controller = FollowPController(
        distance_kp=0.1, direction_kp=1.0, max_speed_mps=0.2
    )
    same_bearing_near = controller.update(3.0, 1.0)
    same_bearing_far = controller.update(3.6, 1.2)
    same_distance_center = controller.update(math.sqrt(10.0), 0.0)

    assert same_bearing_near.steering_angle_rad == pytest.approx(
        same_bearing_far.steering_angle_rad
    )
    assert same_bearing_near.linear_x == pytest.approx(0.1 * (math.hypot(3, 1) - 2.25))
    assert same_bearing_far.linear_x == pytest.approx(
        0.1 * (math.hypot(3.6, 1.2) - 2.25)
    )
    assert same_distance_center.linear_x == pytest.approx(same_bearing_near.linear_x)
    assert same_distance_center.steering_angle_rad == 0.0

    distance_gain_changed = FollowPController(
        distance_kp=0.2, max_speed_mps=0.2
    ).update(3.0, 1.0)
    direction_gain_changed = FollowPController(
        direction_kp=2.0, max_speed_mps=0.2
    ).update(3.0, 1.0)
    assert distance_gain_changed.steering_angle_rad == pytest.approx(
        same_bearing_near.steering_angle_rad
    )
    assert distance_gain_changed.linear_x > same_bearing_near.linear_x
    assert direction_gain_changed.linear_x == pytest.approx(same_bearing_near.linear_x)
    assert direction_gain_changed.steering_angle_rad > same_bearing_near.steering_angle_rad


def test_direction_error_is_retained_at_distance_setpoint_without_turning_in_place():
    controller = FollowPController(direction_kp=1.0)
    at_setpoint_off_center = controller.update(1.35, 1.8)  # Horizontal range is 2.25 m.

    assert at_setpoint_off_center.linear_x == 0.0
    assert at_setpoint_off_center.steering_angle_rad > 0.0
    assert at_setpoint_off_center.angular_z == 0.0


def test_ackermann_does_not_turn_in_place_or_reverse_for_close_or_rear_target():
    controller = FollowPController()
    for point in ((1.0, 0.0), (-3.0, 0.0), (-3.0, 2.0)):
        command = controller.update(*point)
        assert command.linear_x == 0.0
        assert command.angular_z == 0.0
    close_off_center = controller.update(1.8, 0.4)
    assert close_off_center.steering_angle_rad > 0.0
    assert close_off_center.linear_x == 0.0
    assert close_off_center.angular_z == 0.0


def test_nonfinite_target_and_plugin_limit_overrides_are_rejected():
    with pytest.raises(ValueError):
        FollowPController().update(float("nan"), 0.0)
    with pytest.raises(ValueError):
        FollowPController(max_speed_mps=0.21)
    with pytest.raises(ValueError):
        FollowPController(max_steer_rad=0.6)
    with pytest.raises(ValueError):
        FollowPController(min_distance_m=2.25)
