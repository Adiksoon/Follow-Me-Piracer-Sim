import json
import math
from pathlib import Path

import pytest

from piracer_state_manager.follow_control import FollowPController, _normalized_angle_delta


def _point(bearing, distance=3.0):
    return distance * math.cos(bearing), distance * math.sin(bearing)


def _update_bearing(controller, bearing, stamp):
    return controller.update(*_point(bearing), measurement_stamp_ns=stamp)


def test_p_only_matches_pre_extension_golden_outputs_with_or_without_timestamp_argument():
    reference_path = Path(__file__).parent / "data" / "reference_p.json"
    reference = json.loads(reference_path.read_text())
    fields = ("linear_x", "angular_z", "steering_angle_rad", "distance_m", "bearing_rad")
    for gains in reference["gain_configs"]:
        controller = FollowPController(
            distance_kp=gains["distance_kp"], direction_kp=gains["direction_kp"]
        )
        for case in gains["cases"]:
            command = controller.update(case["x_m"], case["y_m"])
            stamped_command = controller.update(
                case["x_m"], case["y_m"], measurement_stamp_ns=0
            )
            for field in fields:
                assert getattr(command, field) == pytest.approx(case[field], abs=1e-14)
                assert getattr(stamped_command, field) == pytest.approx(case[field], abs=1e-14)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_derivative_sign_tracks_increasing_and_decreasing_bearing(direction):
    controller = FollowPController(direction_kd=0.5)
    _update_bearing(controller, 0.2, 1_000_000_000)
    result = _update_bearing(controller, 0.3 if direction > 0 else 0.1, 1_100_000_000)
    assert result.direction_d_filtered_radps * direction > 0.0
    assert result.direction_d_rad * direction > 0.0


def test_derivative_wraps_at_pi_and_uses_variable_measurement_dt_filter():
    controller = FollowPController(direction_kd=1.0, direction_filter_tau_s=0.3)
    assert _normalized_angle_delta(-3.13, 3.13) == pytest.approx(0.02318530717958669)
    _update_bearing(controller, 0.1, 1_000_000_000)
    result = _update_bearing(controller, 0.2, 1_100_000_000)
    wrapped_delta = 0.1
    assert result.direction_d_raw_radps == pytest.approx(wrapped_delta / 0.1)
    assert result.direction_d_filtered_radps == pytest.approx(wrapped_delta / 0.4)

    result = _update_bearing(controller, 0.4, 1_300_000_000)
    assert result.measurement_dt_s == pytest.approx(0.2)
    assert result.direction_d_filtered_radps == pytest.approx(
        0.3 / 0.5 * (wrapped_delta / 0.4) + 0.2 / 0.5
    )


def test_combined_direction_pid_sums_terms_from_one_new_measurement():
    pid = FollowPController(direction_kp=1.0, direction_ki=0.01, direction_kd=0.5)
    pd = FollowPController(direction_kp=1.0, direction_ki=0.0, direction_kd=0.5)
    pi = FollowPController(direction_kp=1.0, direction_ki=0.01, direction_kd=0.0)
    for controller in (pid, pd, pi):
        _update_bearing(controller, 0.1, 1_000_000_000)

    result = _update_bearing(pid, 0.2, 1_100_000_000)
    pd_result = _update_bearing(pd, 0.2, 1_100_000_000)
    pi_result = _update_bearing(pi, 0.2, 1_100_000_000)
    assert result.measurement_dt_s == pytest.approx(0.1)
    assert result.direction_p_rad == pytest.approx(0.2)
    assert result.direction_d_filtered_radps == pytest.approx(0.25)
    assert result.direction_d_rad == pytest.approx(0.125)
    assert result.direction_i_rad == pytest.approx(0.0002)
    assert result.steering_angle_rad == pytest.approx(0.3252)
    assert result.direction_i_rad == pytest.approx(pi_result.direction_i_rad)
    assert pd_result.direction_i_rad == 0.0
    assert result.direction_p_rad == pd_result.direction_p_rad
    assert result.direction_d_rad == pd_result.direction_d_rad
    assert pi_result.direction_p_rad + pi_result.direction_i_rad == pytest.approx(0.2002)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_combined_pid_antiwindup_accounts_for_positive_and_negative_d(sign):
    positive_d = FollowPController(direction_kp=1.0, direction_ki=0.01, direction_kd=0.5)
    _update_bearing(positive_d, 0.0, 1_000_000_000)
    saturated = _update_bearing(positive_d, sign * 0.6, 1_100_000_000)
    assert saturated.direction_d_rad * sign > 0.0
    assert saturated.direction_saturated
    assert saturated.direction_i_rad == 0.0

    negative_d = FollowPController(direction_kp=0.0, direction_ki=0.01, direction_kd=0.5)
    _update_bearing(negative_d, sign * 1.2, 1_000_000_000)
    accumulated = _update_bearing(negative_d, sign * 1.2, 2_000_000_000)
    assert accumulated.direction_i_rad == pytest.approx(sign * 0.012)
    negative_d.direction_kp = 1.0

    saturated = _update_bearing(negative_d, sign * 1.0, 2_100_000_000)
    assert saturated.direction_d_rad * sign < 0.0
    assert saturated.direction_saturated
    assert saturated.direction_i_rad == accumulated.direction_i_rad

    opposite_saturated = _update_bearing(negative_d, -sign * 0.2, 2_200_000_000)
    assert opposite_saturated.direction_saturated
    assert opposite_saturated.direction_i_rad == accumulated.direction_i_rad
    still_saturated = _update_bearing(negative_d, -sign * 0.2, 3_200_000_000)
    assert still_saturated.direction_saturated
    assert still_saturated.direction_i_rad == accumulated.direction_i_rad
    unwinding = _update_bearing(negative_d, -sign * 0.2, 4_200_000_000)
    assert not unwinding.direction_saturated
    assert abs(unwinding.direction_i_rad) < abs(accumulated.direction_i_rad)


def test_first_duplicate_and_regressing_timestamps_have_defined_history():
    controller = FollowPController(direction_kd=0.5, direction_ki=0.01, distance_ki=0.1)
    first = controller.update(3.0, 0.0, measurement_stamp_ns=2_000_000_000)
    assert first.measurement_new
    assert first.measurement_dt_s == 0.0
    assert first.direction_d_rad == 0.0
    assert first.integral_state_mps == 0.0

    newer = controller.update(3.0, 0.3, measurement_stamp_ns=2_100_000_000)
    duplicate = controller.update(3.0, 0.8, measurement_stamp_ns=2_100_000_000)
    assert newer.measurement_new and newer.measurement_dt_s == pytest.approx(0.1)
    assert newer.direction_i_rad > 0.0
    assert not duplicate.measurement_new and duplicate.measurement_dt_s == 0.0
    assert duplicate.direction_d_filtered_radps == newer.direction_d_filtered_radps
    assert duplicate.direction_i_rad == newer.direction_i_rad
    assert duplicate.integral_state_mps == newer.integral_state_mps

    regressed = controller.update(3.0, 0.6, measurement_stamp_ns=1_900_000_000)
    assert regressed.measurement_new
    assert regressed.measurement_dt_s == 0.0
    assert regressed.direction_d_rad == 0.0
    assert regressed.direction_i_rad == 0.0
    assert regressed.integral_state_mps == 0.0


@pytest.mark.parametrize("stamp", [0, -1, float("nan"), float("inf"), None, True, 1.5])
def test_invalid_timestamp_resets_and_raises_only_when_extension_is_active(stamp):
    controller = FollowPController(direction_kd=0.1, distance_ki=0.001, distance_kd=0.1)
    controller.update(2.3, 0.2, measurement_stamp_ns=1_000_000_000)
    controller.update(2.4, 0.3, measurement_stamp_ns=2_000_000_000)
    assert controller._distance_i_mps > 0.0
    assert controller._distance_d_filtered_mps > 0.0
    with pytest.raises(ValueError):
        controller.update(3.0, 0.2, measurement_stamp_ns=stamp)
    assert controller._last_measurement_stamp_ns is None
    assert controller._last_distance_error_m is None
    assert controller._distance_i_mps == 0.0
    assert controller._distance_d_filtered_mps == 0.0
    # The legacy P-only call keeps its two-argument API and no stamp requirement.
    assert FollowPController().update(3.0, 0.2).linear_x > 0.0


def test_pi_integrates_new_measurements_in_meters_per_second_and_holds_duplicates():
    controller = FollowPController(distance_ki=0.005)
    first = controller.update(2.45, 0.0, measurement_stamp_ns=1_000_000_000)
    assert first.distance_p_mps == pytest.approx(0.02)
    assert first.distance_i_mps == 0.0
    second = controller.update(2.45, 0.0, measurement_stamp_ns=2_000_000_000)
    assert second.distance_i_mps == pytest.approx(0.001)
    assert second.integral_state_mps / controller.distance_ki == pytest.approx(0.2)
    assert second.linear_x == pytest.approx(0.021)
    duplicate = controller.update(2.45, 0.0, measurement_stamp_ns=2_000_000_000)
    assert duplicate.distance_i_mps == second.distance_i_mps
    assert duplicate.linear_x == second.linear_x


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_distance_derivative_sign_filter_dt_duplicates_and_reacquisition(sign):
    controller = FollowPController(
        distance_kp=0.0, distance_kd=0.05, distance_filter_tau_s=0.3
    )
    first = controller.update(2.25 + sign * 0.1, 0.0, measurement_stamp_ns=1_000_000_000)
    assert first.distance_d_mps == 0.0
    result = controller.update(2.25 + sign * 0.2, 0.0, measurement_stamp_ns=1_100_000_000)
    assert result.distance_d_raw_mps == pytest.approx(sign * 1.0)
    assert result.distance_d_filtered_mps == pytest.approx(sign * 0.25)
    assert result.distance_d_mps == pytest.approx(sign * 0.0125)
    assert 0.0 <= result.linear_x <= controller.max_speed_mps
    assert result.angular_z == 0.0

    variable_dt = controller.update(
        2.25 + sign * 0.2, 0.0, measurement_stamp_ns=1_300_000_000
    )
    assert variable_dt.measurement_dt_s == pytest.approx(0.2)
    assert variable_dt.distance_d_raw_mps == 0.0
    assert variable_dt.distance_d_filtered_mps == pytest.approx(sign * 0.15)
    duplicate = controller.update(
        2.25 + sign * 0.1, 0.0, measurement_stamp_ns=1_300_000_000
    )
    assert not duplicate.measurement_new
    assert duplicate.distance_d_filtered_mps == variable_dt.distance_d_filtered_mps

    reacquired = controller.update(
        2.25 + sign * 0.15, 0.0, measurement_stamp_ns=1_200_000_000
    )
    assert reacquired.measurement_dt_s == 0.0
    assert reacquired.distance_d_raw_mps == reacquired.distance_d_filtered_mps == 0.0
    assert reacquired.distance_d_mps == 0.0


def test_distance_pid_sums_p_i_d_and_clamps_only_enabled_integral_contribution():
    controller = FollowPController(
        distance_kp=0.1,
        distance_ki=0.00008,
        distance_kd=0.05,
        distance_filter_tau_s=0.3,
        distance_integral_limit_mps=0.0005,
    )
    controller.update(2.35, 0.0, measurement_stamp_ns=1_000_000_000)
    result = controller.update(2.45, 0.0, measurement_stamp_ns=1_100_000_000)
    assert result.distance_p_mps == pytest.approx(0.02)
    assert result.distance_d_raw_mps == pytest.approx(1.0)
    assert result.distance_d_filtered_mps == pytest.approx(0.25)
    assert result.distance_d_mps == pytest.approx(0.0125)
    assert result.distance_i_mps == pytest.approx(0.0000016)
    assert result.distance_unsaturated_mps == pytest.approx(
        result.distance_p_mps + result.distance_i_mps + result.distance_d_mps
    )

    for index in range(1, 41):
        result = controller.update(2.45, 0.0, measurement_stamp_ns=(1_100_000_000 + index * 1_000_000_000))
    assert result.distance_i_mps == pytest.approx(0.0005)
    assert result.distance_unsaturated_mps <= controller.max_speed_mps


def test_distance_pid_antiwindup_uses_d_at_both_speed_limits_and_allows_unwind():
    upper = FollowPController(
        distance_kp=0.1, distance_ki=0.5, distance_kd=0.05,
        distance_integral_limit_mps=0.01,
    )
    upper.update(2.25, 0.0, measurement_stamp_ns=1_000_000_000)
    upper_saturated = upper.update(2.65, 0.0, measurement_stamp_ns=1_100_000_000)
    assert upper_saturated.distance_d_mps > 0.0
    assert upper_saturated.distance_saturated_high
    assert upper_saturated.distance_i_mps == 0.0

    lower = FollowPController(distance_kp=0.0, distance_ki=0.1)
    lower.update(2.35, 0.0, measurement_stamp_ns=1_000_000_000)
    lower_integral = lower.update(2.35, 0.0, measurement_stamp_ns=2_000_000_000)
    assert lower_integral.distance_i_mps == pytest.approx(0.01)
    lower.update(3.25, 0.0, measurement_stamp_ns=3_000_000_000)
    assert lower._distance_i_mps == lower_integral.distance_i_mps
    lower.distance_kp = 0.1
    lower.distance_kd = 0.05
    lower_saturated = lower.update(2.45, 0.0, measurement_stamp_ns=3_100_000_000)
    assert lower_saturated.distance_d_mps < 0.0
    assert lower_saturated.distance_saturated_low
    assert lower_saturated.distance_i_mps > lower_integral.distance_i_mps

    lower_block = FollowPController(distance_kp=0.1, distance_ki=0.5, distance_kd=0.05)
    lower_block.update(2.25, 0.0, measurement_stamp_ns=1_000_000_000)
    blocked = lower_block.update(2.05, 0.0, measurement_stamp_ns=1_100_000_000)
    assert blocked.distance_d_mps < 0.0
    assert blocked.distance_saturated_low
    assert blocked.distance_i_mps == 0.0


def test_integrator_blocks_windup_at_both_speed_limits_and_unwinds():
    upper = FollowPController(distance_ki=0.5)
    upper.update(2.3, 0.0, measurement_stamp_ns=1_000_000_000)
    upper.update(2.3, 0.0, measurement_stamp_ns=2_000_000_000)
    upper.update(8.0, 0.0, measurement_stamp_ns=3_000_000_000)
    assert upper._distance_i_mps == pytest.approx(0.025)

    lower = FollowPController(distance_ki=0.5)
    lower.update(2.1, 0.0, measurement_stamp_ns=1_000_000_000)
    lower.update(2.1, 0.0, measurement_stamp_ns=2_000_000_000)
    assert lower._distance_i_mps == 0.0
    assert lower.update(2.3, 0.0, measurement_stamp_ns=3_000_000_000).integral_state_mps > 0.0

    unwind = FollowPController(distance_ki=0.05)
    unwind.update(2.3, 0.0, measurement_stamp_ns=1_000_000_000)
    for index in range(2, 21):
        unwind.update(2.3, 0.0, measurement_stamp_ns=index * 1_000_000_000)
    before = unwind._distance_i_mps
    saturated = unwind.update(2.5, 0.0, measurement_stamp_ns=21_000_000_000)
    result = unwind.update(2.24, 0.0, measurement_stamp_ns=22_000_000_000)
    assert before > 0.0
    assert saturated.distance_saturated_high
    assert result.integral_state_mps < before


def test_pi_keeps_nonzero_integral_at_setpoint_but_hard_stops_reset_history():
    controller = FollowPController(distance_ki=0.02, direction_kd=0.5)
    controller.update(2.3, 0.0, measurement_stamp_ns=1_000_000_000)
    controller.update(2.3, 0.0, measurement_stamp_ns=2_000_000_000)
    at_setpoint = controller.update(2.25, 0.0, measurement_stamp_ns=3_000_000_000)
    assert at_setpoint.integral_state_mps > 0.0
    assert at_setpoint.linear_x > 0.0

    for x, y in ((-3.0, 0.0), (2.0, 0.0), (1.9, 0.0)):
        stopped = controller.update(x, y, measurement_stamp_ns=3_000_000_000)
        assert stopped.linear_x == 0.0
        assert stopped.angular_z == 0.0
        assert stopped.integral_state_mps == 0.0
        assert stopped.distance_i_mps == 0.0
        assert controller._last_measurement_stamp_ns is None


def test_distance_derivative_and_integral_reset_on_hard_stop():
    controller = FollowPController(distance_ki=0.001, distance_kd=0.05)
    controller.update(2.3, 0.0, measurement_stamp_ns=1_000_000_000)
    moving = controller.update(2.4, 0.0, measurement_stamp_ns=2_000_000_000)
    assert moving.distance_d_mps > 0.0
    assert moving.distance_i_mps > 0.0

    stopped = controller.update(1.9, 0.0, measurement_stamp_ns=3_000_000_000)
    assert stopped.linear_x == stopped.angular_z == 0.0
    assert stopped.distance_i_mps == 0.0
    assert stopped.distance_d_mps == 0.0
    assert stopped.distance_d_filtered_mps == 0.0
    assert controller._last_measurement_stamp_ns is None


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_direction_integral_tracks_bearing_sign_and_clamps_both_directions(sign):
    controller = FollowPController(direction_ki=0.01, direction_integral_limit_rad=0.02)
    _update_bearing(controller, 0.0, 1_000_000_000)
    result = _update_bearing(controller, sign * 0.2, 2_000_000_000)
    assert result.direction_i_rad == pytest.approx(sign * 0.002)

    for index in range(3, 20):
        result = _update_bearing(controller, sign * 0.2, index * 1_000_000_000)
    assert result.direction_i_rad == pytest.approx(sign * 0.02)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_direction_integral_blocks_windup_at_both_steering_limits_and_unwinds(sign):
    controller = FollowPController(direction_kp=0.0, direction_ki=0.01)
    _update_bearing(controller, sign * 0.2, 1_000_000_000)
    before = _update_bearing(controller, sign * 0.2, 2_000_000_000)
    assert before.direction_i_rad == pytest.approx(sign * 0.002)

    controller.direction_kp = 1.0
    saturated = _update_bearing(controller, sign * 0.6, 3_000_000_000)
    assert saturated.direction_saturated
    assert saturated.direction_i_rad == before.direction_i_rad
    unwind = _update_bearing(controller, -sign * 0.2, 4_000_000_000)
    assert abs(unwind.direction_i_rad) < abs(before.direction_i_rad)


def test_direction_integral_holds_repeated_stamps_and_resets_invalid_stamp():
    controller = FollowPController(direction_ki=0.01)
    _update_bearing(controller, 0.2, 1_000_000_000)
    newer = _update_bearing(controller, 0.2, 2_000_000_000)
    duplicate = _update_bearing(controller, 0.4, 2_000_000_000)
    assert duplicate.direction_i_rad == newer.direction_i_rad
    assert not duplicate.measurement_new and duplicate.measurement_dt_s == 0.0

    with pytest.raises(ValueError):
        controller.update(3.0, 0.2, measurement_stamp_ns=0)
    assert controller._direction_i_rad == 0.0
    assert controller._last_measurement_stamp_ns is None


def test_direction_integral_freezes_at_zero_speed_and_does_not_command_yaw():
    controller = FollowPController(direction_ki=0.01, direction_kd=0.5)
    _update_bearing(controller, 0.3, 1_000_000_000)
    moving_before = _update_bearing(controller, 0.3, 2_000_000_000)
    assert moving_before.direction_i_rad == pytest.approx(0.003)

    stopped = controller.update(*_point(-0.2, distance=2.2), measurement_stamp_ns=3_000_000_000)
    stopped_again = controller.update(
        *_point(-0.2, distance=2.2), measurement_stamp_ns=3_100_000_000
    )
    assert stopped.linear_x == stopped_again.linear_x == 0.0
    assert stopped.angular_z == stopped_again.angular_z == 0.0
    assert stopped.direction_i_rad == stopped_again.direction_i_rad == moving_before.direction_i_rad
    assert stopped.direction_d_rad < 0.0
    assert stopped_again.direction_d_rad < 0.0

    moving = controller.update(
        *_point(-0.2, distance=3.0), measurement_stamp_ns=3_200_000_000
    )
    assert 0.0 < moving.linear_x <= controller.max_speed_mps
    assert moving.measurement_dt_s == pytest.approx(0.1)
    assert moving.direction_i_rad == pytest.approx(
        moving_before.direction_i_rad + controller.direction_ki * -0.2 * 0.1
    )


def test_direction_integral_is_cleared_by_hard_stop_and_reset_for_reacquisition():
    controller = FollowPController(direction_ki=0.01, direction_kd=0.5)
    _update_bearing(controller, 0.2, 1_000_000_000)
    integrated = _update_bearing(controller, 0.3, 2_000_000_000)
    assert integrated.direction_i_rad > 0.0
    assert integrated.direction_d_filtered_radps > 0.0
    stopped = controller.update(1.9, 0.0, measurement_stamp_ns=3_000_000_000)
    assert stopped.direction_i_rad == 0.0
    assert stopped.direction_d_filtered_radps == 0.0
    assert controller._last_measurement_stamp_ns is None

    _update_bearing(controller, 0.2, 4_000_000_000)
    reacquired = _update_bearing(controller, 0.2, 5_000_000_000)
    assert reacquired.direction_i_rad > 0.0
    assert reacquired.direction_d_filtered_radps == 0.0
    controller.reset("target_reacquired")
    assert controller._direction_i_rad == 0.0
    assert controller._last_measurement_stamp_ns is None


def test_invalid_geometry_resets_d_and_i_before_raising():
    controller = FollowPController(distance_ki=0.01, direction_kd=0.2)
    controller.update(3.0, 0.2, measurement_stamp_ns=1_000_000_000)
    controller.update(3.0, 0.3, measurement_stamp_ns=2_000_000_000)
    with pytest.raises(ValueError):
        controller.update(float("nan"), 0.0, measurement_stamp_ns=3_000_000_000)
    assert controller._last_measurement_stamp_ns is None
    assert controller._distance_i_mps == 0.0


def test_bounded_target_stop_sequence_keeps_commands_forward_only():
    controller = FollowPController(distance_ki=0.005, max_speed_mps=0.05)
    distances = [3.0, 3.0, 2.8, 2.6, 2.4, 2.3, 2.25, 2.25, 2.25]
    outputs = [
        controller.update(distance, 0.0, measurement_stamp_ns=(index + 1) * 1_000_000_000)
        for index, distance in enumerate(distances)
    ]
    assert all(0.0 <= output.linear_x <= 0.05 for output in outputs)
    assert outputs[-1].integral_state_mps > 0.0
    assert outputs[-1].linear_x > 0.0
    stopped = controller.update(1.99, 0.0, measurement_stamp_ns=10_000_000_000)
    assert stopped.linear_x == 0.0
    assert stopped.integral_state_mps == 0.0
