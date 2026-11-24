"""Bounded proportional control for a visible target and Ackermann steering."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator


# Verified in piracer.urdf.xacro for the active Gazebo Ackermann plugin.
ACKERMANN_WHEELBASE_M = 0.155
ACKERMANN_MAX_STEER_RAD = 0.554
ACKERMANN_PLUGIN_MAX_SPEED_MPS = 0.20


def _normalized_angle_delta(current_rad: float, previous_rad: float) -> float:
    return math.atan2(
        math.sin(current_rad - previous_rad),
        math.cos(current_rad - previous_rad),
    )


@dataclass(frozen=True)
class FollowCommand:
    linear_x: float
    angular_z: float
    steering_angle_rad: float
    distance_m: float
    bearing_rad: float
    distance_p_mps: float = 0.0
    distance_i_mps: float = 0.0
    distance_unsaturated_mps: float = 0.0
    direction_p_rad: float = 0.0
    direction_d_rad: float = 0.0
    direction_d_raw_radps: float = 0.0
    distance_saturated_low: bool = False
    distance_saturated_high: bool = False
    direction_saturated: bool = False
    integral_state_mps: float = 0.0
    measurement_new: bool = False
    measurement_dt_s: float = 0.0
    direction_i_rad: float = 0.0
    distance_d_mps: float = 0.0
    distance_d_raw_mps: float = 0.0


class FollowPController:
    """Convert a base_link target point to bounded Ackermann-compatible Twist.

    Distance uses horizontal base_link range. Direction uses horizontal bearing.
    Both P gains are provisional and must be tuned in repeatable experiments.
    Optional D terms use the raw error difference over the measurement interval,
    held until the next measurement. Direction differences wrap at pi.
    """

    def __init__(
        self,
        *,
        target_distance_m: float = 2.25,
        min_distance_m: float = 2.0,
        distance_kp: float = 0.10,
        distance_ki: float = 0.0,
        distance_kd: float = 0.0,
        distance_integral_limit_mps: float = 0.0,
        direction_kp: float = 1.0,
        direction_kd: float = 0.0,
        direction_ki: float = 0.0,
        direction_integral_limit_rad: float = 0.02,
        max_speed_mps: float = 0.05,
        wheelbase_m: float = ACKERMANN_WHEELBASE_M,
        max_steer_rad: float = ACKERMANN_MAX_STEER_RAD,
    ) -> None:
        values = (
            target_distance_m,
            min_distance_m,
            distance_kp,
            distance_ki,
            distance_kd,
            distance_integral_limit_mps,
            direction_kp,
            direction_kd,
            direction_ki,
            direction_integral_limit_rad,
            max_speed_mps,
            wheelbase_m,
            max_steer_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Follow controller parameters must be finite")
        if not 0.0 < min_distance_m < target_distance_m:
            raise ValueError("min_distance_m must be positive and below target_distance_m")
        if (distance_kp < 0.0 or distance_ki < 0.0 or distance_kd < 0.0
                or direction_kp < 0.0
                or direction_kd < 0.0 or direction_ki < 0.0):
            raise ValueError("Controller gains must be non-negative")
        if distance_integral_limit_mps < 0.0:
            raise ValueError("distance_integral_limit_mps must be non-negative")
        if direction_integral_limit_rad <= 0.0:
            raise ValueError("direction_integral_limit_rad must be positive")
        if not 0.0 < max_speed_mps <= ACKERMANN_PLUGIN_MAX_SPEED_MPS:
            raise ValueError("max_speed_mps exceeds the Ackermann plugin limit")
        if not 0.0 < wheelbase_m or not 0.0 < max_steer_rad <= ACKERMANN_MAX_STEER_RAD:
            raise ValueError("Ackermann geometry exceeds the configured plugin limits")

        self.target_distance_m = target_distance_m
        self.min_distance_m = min_distance_m
        self.distance_kp = distance_kp
        self.distance_ki = distance_ki
        self.distance_kd = distance_kd
        self.distance_integral_limit_mps = distance_integral_limit_mps
        self.direction_kp = direction_kp
        self.direction_kd = direction_kd
        self.direction_ki = direction_ki
        self.direction_integral_limit_rad = direction_integral_limit_rad
        self.max_speed_mps = max_speed_mps
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad

        self.reset()

    def reset(self, reason: str = "reset") -> None:
        """Clear history used by the optional D and I terms."""
        self._last_measurement_stamp_ns = None
        self._last_distance_error_m = None
        self._last_direction_error_rad = None
        self._distance_d_raw_mps = 0.0
        self._direction_d_raw_radps = 0.0
        self._direction_i_rad = 0.0
        self._distance_i_mps = 0.0
        self._last_reset_reason = str(reason)

    def update(self, x_m: float, y_m: float, *, measurement_stamp_ns=None) -> FollowCommand:
        """Return one command for a target's horizontal position in ``base_link``."""
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            self.reset("invalid_geometry")
            raise ValueError("target position must be finite")

        distance_m = math.hypot(x_m, y_m)
        bearing_rad = math.atan2(y_m, x_m)

        # Positive error means the suitcase is farther than the setpoint.
        # Saturating at zero makes this forward-only: no reverse command.
        distance_error_m = distance_m - self.target_distance_m
        direction_error_rad = bearing_rad
        extensions_active = (
            self.distance_ki > 0.0 or self.distance_kd > 0.0
            or self.direction_kd > 0.0 or self.direction_ki > 0.0
        )
        measurement_new = False
        dt_s = 0.0
        if extensions_active:
            try:
                if isinstance(measurement_stamp_ns, bool):
                    raise ValueError
                stamp_ns = operator.index(measurement_stamp_ns)
                if stamp_ns <= 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                self.reset("invalid_measurement_stamp")
                raise ValueError("measurement_stamp_ns must be finite and positive when D or I is enabled")

            if self._last_measurement_stamp_ns is None:
                self._last_measurement_stamp_ns = stamp_ns
                self._last_distance_error_m = distance_error_m
                self._last_direction_error_rad = direction_error_rad
                measurement_new = True
            elif stamp_ns < self._last_measurement_stamp_ns:
                self.reset("measurement_stamp_regression")
                self._last_measurement_stamp_ns = stamp_ns
                self._last_distance_error_m = distance_error_m
                self._last_direction_error_rad = direction_error_rad
                measurement_new = True
            elif stamp_ns > self._last_measurement_stamp_ns:
                dt_s = (stamp_ns - self._last_measurement_stamp_ns) / 1_000_000_000.0
                measurement_new = True
                if self.distance_kd > 0.0:
                    delta_distance = distance_error_m - self._last_distance_error_m
                    self._distance_d_raw_mps = delta_distance / dt_s
                if self.direction_kd > 0.0:
                    delta = _normalized_angle_delta(
                        direction_error_rad, self._last_direction_error_rad
                    )
                    self._direction_d_raw_radps = delta / dt_s
                self._last_measurement_stamp_ns = stamp_ns
                self._last_distance_error_m = distance_error_m
                self._last_direction_error_rad = direction_error_rad

        distance_p_mps = self.distance_kp * distance_error_m
        distance_d_mps = self.distance_kd * self._distance_d_raw_mps
        if extensions_active and self.distance_ki > 0.0 and measurement_new and dt_s > 0.0:
            candidate_i = self._distance_i_mps + self.distance_ki * distance_error_m * dt_s
            if self.distance_integral_limit_mps > 0.0:
                candidate_i = max(
                    -self.distance_integral_limit_mps,
                    min(self.distance_integral_limit_mps, candidate_i),
                )
            candidate_u = distance_p_mps + distance_d_mps + candidate_i
            applied_delta_i = candidate_i - self._distance_i_mps
            if not (
                (candidate_u > self.max_speed_mps and applied_delta_i > 0.0)
                or (candidate_u < 0.0 and applied_delta_i < 0.0)
            ):
                self._distance_i_mps = candidate_i
        distance_unsaturated_mps = distance_p_mps + self._distance_i_mps + distance_d_mps
        speed_mps = min(self.max_speed_mps, max(0.0, distance_unsaturated_mps))
        if x_m <= 0.0 or distance_m <= self.min_distance_m:
            self.reset("hard_stop")
            distance_unsaturated_mps = distance_p_mps
            distance_d_mps = 0.0
            speed_mps = 0.0

        # Keep the steering P output independent from distance control, even
        # at the setpoint. A Twist cannot steer an Ackermann vehicle at rest.
        direction_p_rad = self.direction_kp * direction_error_rad
        direction_d_rad = self.direction_kd * self._direction_d_raw_radps
        if (self.direction_ki > 0.0 and measurement_new and dt_s > 0.0
                and speed_mps > 0.0):
            delta_i = self.direction_ki * direction_error_rad * dt_s
            candidate_i = max(
                -self.direction_integral_limit_rad,
                min(self.direction_integral_limit_rad, self._direction_i_rad + delta_i),
            )
            candidate_u = direction_p_rad + direction_d_rad + candidate_i
            applied_delta_i = candidate_i - self._direction_i_rad
            if not (
                (candidate_u > self.max_steer_rad and applied_delta_i > 0.0)
                or (candidate_u < -self.max_steer_rad and applied_delta_i < 0.0)
            ):
                self._direction_i_rad = candidate_i
        direction_unsaturated_rad = (
            direction_p_rad + direction_d_rad + self._direction_i_rad
        )
        steering_angle_rad = max(
            -self.max_steer_rad,
            min(self.max_steer_rad, direction_unsaturated_rad),
        )

        # Convert steering angle to yaw rate so the existing adapter's
        # atan(wheelbase * yaw_rate / speed) recovers this Ackermann angle.
        angular_z = speed_mps * math.tan(steering_angle_rad) / self.wheelbase_m
        return FollowCommand(
            speed_mps,
            angular_z,
            steering_angle_rad,
            distance_m,
            bearing_rad,
            distance_p_mps,
            self._distance_i_mps,
            distance_unsaturated_mps,
            direction_p_rad,
            direction_d_rad,
            self._direction_d_raw_radps,
            distance_unsaturated_mps < 0.0,
            distance_unsaturated_mps > self.max_speed_mps,
            abs(direction_unsaturated_rad) > self.max_steer_rad,
            self._distance_i_mps,
            measurement_new,
            dt_s,
            self._direction_i_rad,
            distance_d_mps,
            self._distance_d_raw_mps,
        )
