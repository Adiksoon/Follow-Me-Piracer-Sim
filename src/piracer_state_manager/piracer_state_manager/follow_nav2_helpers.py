"""Small pure helpers for the FOLLOW/Nav2 safety handoff."""

from __future__ import annotations

from dataclasses import dataclass
import math


DIRECT = "DIRECT"
NAV2_PENDING = "NAV2_PENDING"
NAV2 = "NAV2"
SAFE_STOP = "SAFE_STOP"

FOOTPRINT = ((-0.020, -0.1085), (0.246, -0.1085), (0.246, 0.1085), (-0.020, 0.1085))
MAX_NAV_SPEED_MPS = 0.05
MAX_STEER_RAD = 0.554
WHEELBASE_M = 0.155


@dataclass(frozen=True)
class ClearanceResult:
    stop_now: bool
    confirm_nav2: bool = False
    return_direct: bool = False
    reason: str = ""


class ClearanceHysteresis:
    """Count distinct costmap stamps; block immediately, require stable clear."""

    def __init__(self, blocked_count: int = 2, clear_count: int = 5) -> None:
        if blocked_count < 1 or clear_count < 1:
            raise ValueError("hysteresis counts must be positive")
        self.blocked_count = blocked_count
        self.clear_count = clear_count
        self.last_stamp_ns = None
        self.blocked = 0
        self.clear = 0
        self._was_blocked = False

    def reset(self) -> None:
        self.last_stamp_ns = None
        self.blocked = 0
        self.clear = 0
        self._was_blocked = False

    def observe(self, is_blocked: bool, stamp_ns: int) -> ClearanceResult:
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns <= 0:
            return ClearanceResult(True, reason="invalid_costmap_stamp")
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            # A repeated map is not new hysteresis evidence, but the current
            # pose/command can have changed since that map was first seen.
            return ClearanceResult(
                bool(is_blocked or self._was_blocked),
                reason="duplicate_costmap_stamp",
            )
        self.last_stamp_ns = stamp_ns
        if is_blocked:
            self.blocked += 1
            self.clear = 0
            first = not self._was_blocked
            self._was_blocked = True
            return ClearanceResult(
                True,
                confirm_nav2=self.blocked >= self.blocked_count,
                reason="blocked_first_map" if first else "blocked_confirmed",
            )
        self.clear += 1
        self.blocked = 0
        was_blocked = self._was_blocked
        if self.clear >= self.clear_count:
            self._was_blocked = False
        return ClearanceResult(
            stop_now=self._was_blocked,
            return_direct=was_blocked and not self._was_blocked,
            reason="clear_confirmed" if was_blocked and not self._was_blocked else "clear_candidate",
        )


def grid_is_fresh(stamp_ns: int, now_ns: int, received_age_s: float,
                  max_age_s: float, future_lead_s: float = 0.2) -> bool:
    values = (received_age_s, max_age_s, future_lead_s)
    if not all(math.isfinite(value) for value in values):
        return False
    return stamped_input_is_fresh(
        stamp_ns, now_ns, received_age_s, max_age_s, future_lead_s
    )


def stamped_input_is_fresh(stamp_ns: int, now_ns: int, received_age_s: float,
                           max_age_s: float, future_lead_s: float = 0.2) -> bool:
    values = (received_age_s, max_age_s, future_lead_s)
    if not all(math.isfinite(value) for value in values):
        return False
    age_s = (now_ns - stamp_ns) / 1_000_000_000.0
    return (
        stamp_ns > 0
        and received_age_s >= 0.0
        and received_age_s <= max_age_s
        and age_s >= -future_lead_s
        and age_s <= max_age_s
    )


def localization_is_fresh(lost: bool, stamp_ns: int, now_ns: int,
                          received_age_s: float, max_age_s: float = 1.0,
                          future_lead_s: float = 0.2) -> bool:
    return (
        not lost
        and stamped_input_is_fresh(
            stamp_ns, now_ns, received_age_s, max_age_s, future_lead_s
        )
    )


def require_distinct_command_topics(output_topic: str, nav2_input_topic: str) -> None:
    """Reject an integrated Follow loop that subscribes to its own output."""
    if output_topic == nav2_input_topic:
        raise ValueError(
            "follow_nav2_avoidance requires command_topic to differ from "
            "nav2_command_topic; both resolve to "
            f"{output_topic!r}"
        )


def _grid_coordinates(grid, x: float, y: float) -> tuple[float, float]:
    origin = grid.info.origin
    q = origin.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    dx, dy = x - origin.position.x, y - origin.position.y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _polygons_overlap_rect(poly, x0, y0, x1, y1) -> bool:
    rect = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for index, point in enumerate(poly):
        other = poly[(index + 1) % len(poly)]
        dx, dy = other[0] - point[0], other[1] - point[1]
        axes.append((-dy, dx))
    for ax, ay in axes:
        p = [x * ax + y * ay for x, y in poly]
        r = [x * ax + y * ay for x, y in rect]
        if max(p) < min(r) - 1e-12 or max(r) < min(p) - 1e-12:
            return False
    return True


def footprint_polygon(x: float, y: float, yaw: float, footprint=FOOTPRINT):
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple((x + c * px - s * py, y + s * px + c * py) for px, py in footprint)


def footprint_clear(grid, x: float, y: float, yaw: float, footprint=FOOTPRINT) -> bool:
    """Check every touched OccupancyGrid cell; unknown/outside is blocked."""
    width, height, resolution = grid.info.width, grid.info.height, grid.info.resolution
    if width <= 0 or height <= 0 or not math.isfinite(resolution) or resolution <= 0.0:
        return False
    poly = footprint_polygon(x, y, yaw, footprint)
    local = tuple(_grid_coordinates(grid, px, py) for px, py in poly)
    max_x, max_y = width * resolution, height * resolution
    if any(px < 0.0 or py < 0.0 or px > max_x or py > max_y for px, py in local):
        return False
    min_i = max(0, int(math.floor(min(p[0] for p in local) / resolution)))
    max_i = min(width - 1, int(math.floor(max(p[0] for p in local) / resolution)))
    min_j = max(0, int(math.floor(min(p[1] for p in local) / resolution)))
    max_j = min(height - 1, int(math.floor(max(p[1] for p in local) / resolution)))
    data = grid.data
    if len(data) != width * height:
        return False
    for j in range(min_j, max_j + 1):
        for i in range(min_i, max_i + 1):
            cell = data[j * width + i]
            if _polygons_overlap_rect(local, i * resolution, j * resolution,
                                      (i + 1) * resolution, (j + 1) * resolution):
                if cell < 0 or cell >= 100:
                    return False
    return True


def swept_footprint_clear(grid, x: float, y: float, yaw: float,
                          linear_x: float, angular_z: float, distance_m: float,
                          footprint=FOOTPRINT) -> tuple[bool, float]:
    """Sweep the FOLLOW command arc, at no more than half-cell spacing."""
    values = (x, y, yaw, linear_x, angular_z, distance_m)
    if not all(math.isfinite(value) for value in values) or distance_m < 0.0:
        return False, 0.0
    if distance_m <= 0.0 or linear_x <= 0.0:
        return True, math.inf
    resolution = float(grid.info.resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        return False, 0.0
    step = min(0.5 * resolution, 0.025)
    count = max(1, int(math.ceil(distance_m / step)))
    for index in range(count + 1):
        distance = distance_m * index / count
        dt = distance / linear_x
        if abs(angular_z) < 1e-9:
            px = x + distance * math.cos(yaw)
            py = y + distance * math.sin(yaw)
            pyaw = yaw
        else:
            radius = linear_x / angular_z
            pyaw = yaw + angular_z * dt
            px = x + radius * (math.sin(pyaw) - math.sin(yaw))
            py = y - radius * (math.cos(pyaw) - math.cos(yaw))
        if not footprint_clear(grid, px, py, pyaw, footprint):
            return False, distance
    return True, math.inf


def stand_off_goal(robot_x: float, robot_y: float, target_x: float, target_y: float,
                   distance_m: float = 2.25) -> tuple[float, float, float] | None:
    values = (robot_x, robot_y, target_x, target_y, distance_m)
    if not all(math.isfinite(value) for value in values) or distance_m <= 0.0:
        return None
    dx, dy = target_x - robot_x, target_y - robot_y
    distance = math.hypot(dx, dy)
    if distance <= distance_m:
        return None
    x = target_x - distance_m * dx / distance
    y = target_y - distance_m * dy / distance
    return x, y, math.atan2(target_y - y, target_x - x)


def bound_nav_twist(linear_x: float, angular_z: float) -> tuple[float, float] | None:
    """Bound speed and Ackermann steering without changing curvature."""
    if not math.isfinite(linear_x) or not math.isfinite(angular_z) or linear_x <= 0.0:
        return None
    speed = min(MAX_NAV_SPEED_MPS, linear_x)
    yaw_rate = angular_z * (speed / linear_x)
    steer = math.atan(WHEELBASE_M * yaw_rate / speed)
    if abs(steer) > MAX_STEER_RAD:
        steer = math.copysign(MAX_STEER_RAD, steer)
        yaw_rate = math.tan(steer) * speed / WHEELBASE_M
    return speed, yaw_rate


def nav2_command_policy(linear_x: float, angular_z: float,
                        received_age_s: float, motion_started: bool):
    """Return (wait|forward|fault, bounded Twist) for the owned Nav2 stream."""
    if not math.isfinite(linear_x) or not math.isfinite(angular_z):
        return "fault", None
    if linear_x < 0.0:
        return "fault", None
    if linear_x == 0.0:
        if abs(angular_z) > 1e-9:
            return "fault", None
        return "wait", (0.0, 0.0)
    if not math.isfinite(received_age_s) or received_age_s < 0.0:
        return "fault", None
    if received_age_s > 0.20:
        return ("fault" if motion_started else "wait"), None
    bounded = bound_nav_twist(linear_x, angular_z)
    return ("forward", bounded) if bounded is not None else ("fault", None)


def goal_refresh_due(target_x: float, target_y: float, goal_x: float, goal_y: float,
                     elapsed_s: float, min_displacement_m: float = 0.25,
                     min_period_s: float = 3.0) -> bool:
    values = (target_x, target_y, goal_x, goal_y, elapsed_s)
    return (
        all(math.isfinite(value) for value in values)
        and elapsed_s >= min_period_s
        and math.hypot(target_x - goal_x, target_y - goal_y) >= min_displacement_m
    )
