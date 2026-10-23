"""Bounded proportional control for a visible target and Ackermann steering."""

from __future__ import annotations

from dataclasses import dataclass
import math


# Verified in piracer.urdf.xacro for the active Gazebo Ackermann plugin.
ACKERMANN_WHEELBASE_M = 0.155
ACKERMANN_MAX_STEER_RAD = 0.554
ACKERMANN_PLUGIN_MAX_SPEED_MPS = 0.20


@dataclass(frozen=True)
class FollowCommand:
    linear_x: float
    angular_z: float
    steering_angle_rad: float
    distance_m: float
    bearing_rad: float


class FollowPController:
    """Convert a base_link target point to a bounded Twist command.

    Controller type and gains are provisional. P versus PI/PD/PID and final
    gain values must be selected after repeatable tuning experiments.
    """

    def __init__(
        self,
        *,
        target_distance_m: float = 2.25,
        min_distance_m: float = 2.0,
        max_distance_m: float = 2.5,
        distance_kp: float = 0.10,
        direction_kp: float = 1.0,
        max_speed_mps: float = 0.05,
        wheelbase_m: float = ACKERMANN_WHEELBASE_M,
        max_steer_rad: float = ACKERMANN_MAX_STEER_RAD,
    ) -> None:
        values = (
            target_distance_m,
            min_distance_m,
            max_distance_m,
            distance_kp,
            direction_kp,
            max_speed_mps,
            wheelbase_m,
            max_steer_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Follow controller parameters must be finite")
        if not 0.0 < min_distance_m < target_distance_m < max_distance_m:
            raise ValueError("distance limits must bracket the target distance")
        if distance_kp < 0.0 or direction_kp < 0.0:
            raise ValueError("P gains must be non-negative")
        if not 0.0 < max_speed_mps <= ACKERMANN_PLUGIN_MAX_SPEED_MPS:
            raise ValueError("max_speed_mps exceeds the Ackermann plugin limit")
        if not 0.0 < wheelbase_m or not 0.0 < max_steer_rad <= ACKERMANN_MAX_STEER_RAD:
            raise ValueError("Ackermann geometry exceeds the configured plugin limits")

        self.target_distance_m = target_distance_m
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.distance_kp = distance_kp
        self.direction_kp = direction_kp
        self.max_speed_mps = max_speed_mps
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad

    def update(self, x_m: float, y_m: float) -> FollowCommand:
        """Return one command for a target expressed in ``base_link``."""
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("target position must be finite")

        distance_m = math.hypot(x_m, y_m)
        bearing_rad = math.atan2(y_m, x_m)

        # No reverse behavior is defined yet. Stay stopped if the target is
        # too close or behind the robot; do not turn in place with Ackermann.
        if x_m <= 0.0 or distance_m < self.min_distance_m:
            return FollowCommand(0.0, 0.0, 0.0, distance_m, bearing_rad)

        # The accepted range band is 2.0-2.5 m. Drive only when farther than
        # its upper edge; target distance and gains remain provisional.
        if distance_m <= self.max_distance_m:
            return FollowCommand(0.0, 0.0, 0.0, distance_m, bearing_rad)

        # Apply P to the distance beyond the accepted band, avoiding a speed
        # step when the target first crosses the 2.5 m upper bound.
        distance_error_m = distance_m - self.max_distance_m
        speed_mps = min(self.max_speed_mps, self.distance_kp * distance_error_m)
        steering_angle_rad = max(
            -self.max_steer_rad,
            min(self.max_steer_rad, self.direction_kp * bearing_rad),
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
        )
