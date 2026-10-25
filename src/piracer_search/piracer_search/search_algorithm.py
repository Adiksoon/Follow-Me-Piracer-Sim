"""Pure-Python OccupancyGrid adapter and deterministic frontier ranking."""

from collections import deque
import math

FREE, OCCUPIED, UNKNOWN = 0, 100, -1
DIRECTIONS = ((1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3))
_DIRECTION_YAWS = (0.0, math.pi / 2, math.pi, -math.pi / 2)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def world_to_cell(grid, x, y):
    """Convert a world point to a cell using the rotated map origin."""
    origin_x, origin_y, origin_yaw = grid["origin"]
    dx, dy = x - origin_x, y - origin_y
    local_x = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
    local_y = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
    return math.floor(local_x / grid["resolution"]), math.floor(local_y / grid["resolution"])


def _read_grid(grid):
    width, height = grid.get("width"), grid.get("height")
    resolution, origin, data = grid.get("resolution"), grid.get("origin"), grid.get("data")
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0
            or not isinstance(height, int) or isinstance(height, bool) or height <= 0):
        raise ValueError("invalid map dimensions")
    if (not _number(resolution) or resolution <= 0
            or not isinstance(origin, (list, tuple)) or len(origin) != 3
            or not all(_number(value) for value in origin)):
        raise ValueError("invalid map geometry")
    if not isinstance(data, (list, tuple)) or len(data) != width * height:
        raise ValueError("map data length mismatch")
    if any(not isinstance(value, int) or isinstance(value, bool)
           or not -1 <= value <= 100 for value in data):
        raise ValueError("occupancy values must be in [-1, 100]")
    cells = {
        (index % width, index // width): (
            FREE if value == FREE else OCCUPIED if value == OCCUPIED else UNKNOWN
        )
        for index, value in enumerate(data)
    }
    return width, height, float(resolution), list(origin), cells


def _inside(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _safe_free_cells(width, height, cells):
    """Return FREE cells whose full 3x3 footprint is also FREE."""
    safe_cells = set()
    for y in range(height):
        for x in range(width):
            cell = (x, y)
            if cells[cell] != FREE:
                continue
            if all(
                    _inside((x + dx, y + dy), width, height)
                    and cells[(x + dx, y + dy)] == FREE
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
            ):
                safe_cells.add(cell)
    return safe_cells


def _reachable_edge_counts(start, width, height, safe_cells):
    """Return cardinal shortest-path edge counts over safe FREE cells."""
    if start not in safe_cells:
        return {}
    pending = deque([start])
    edge_counts = {start: 0}
    while pending:
        cell = pending.popleft()
        for dx, dy, _ in DIRECTIONS:
            neighbor = cell[0] + dx, cell[1] + dy
            if neighbor in safe_cells and neighbor not in edge_counts:
                edge_counts[neighbor] = edge_counts[cell] + 1
                pending.append(neighbor)
    return edge_counts


def _unknown_boundary_cells(width, height, cells):
    """Return UNKNOWN cells adjacent to FREE cells, in row-major order."""
    targets = []
    for y in range(height):
        for x in range(width):
            if cells[(x, y)] != UNKNOWN:
                continue
            if any(
                    _inside((x + dx, y + dy), width, height)
                    and cells[(x + dx, y + dy)] == FREE
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if dx != 0 or dy != 0
            ):
                targets.append((x, y))
    return targets


def _ray(start, end):
    """Return integer cells along a Bresenham line, including both ends."""
    x, y = start
    end_x, end_y = end
    dx, step_x = abs(end_x - x), 1 if x < end_x else -1
    dy, step_y = -abs(end_y - y), 1 if y < end_y else -1
    error, points = dx + dy, []
    while True:
        points.append((x, y))
        if (x, y) == (end_x, end_y):
            return points
        twice_error = 2 * error
        if twice_error >= dy:
            error, x = error + dy, x + step_x
        if twice_error <= dx:
            error, y = error + dx, y + step_y


def _visible_unknown(
        observer, yaw, camera, width, height, resolution, origin_yaw, cells, unknown_targets
):
    """Count UNKNOWN cells under the frozen UNKNOWN_OPAQUE visibility rule."""
    fov = camera["horizontal_fov_deg"]
    max_range = camera["max_range_m"]
    heading = math.atan2(math.sin(yaw - origin_yaw), math.cos(yaw - origin_yaw))
    visible = []
    for x, y in unknown_targets:
        dx = (x - observer[0]) * resolution
        dy = (y - observer[1]) * resolution
        distance = math.hypot(dx, dy)
        if distance == 0 or distance > max_range + 1e-12:
            continue
        angle = math.atan2(dy, dx)
        difference = math.atan2(math.sin(angle - heading), math.cos(angle - heading))
        if abs(difference) > math.radians(fov) / 2 + 1e-12:
            continue
        middle = _ray(observer, (x, y))[1:-1]
        if any(cells[cell] in (OCCUPIED, UNKNOWN) for cell in middle):
            continue
        visible.append((x, y))
    return visible


def evaluate_search(grid, robot_pose, *, camera=None):
    """Rank safe frontier poses with GAIN_COST_VISIBLE_OPAQUE_W050.

    grid contains OccupancyGrid width, height, resolution, origin [x,y,yaw],
    and row-major data. robot_pose is world (x,y,yaw). Camera requires verified
    horizontal_fov_deg and max_range_m settings; missing settings return a
    waiting status. Grid path cost is only a ranking input; Nav2 executes routes.
    """
    if grid is None:
        return {"status": "WAITING_FOR_MAP", "candidates": [], "ranking": []}
    if robot_pose is None:
        return {"status": "WAITING_FOR_LOCALIZATION", "candidates": [], "ranking": []}
    if (not isinstance(camera, dict)
            or not _number(camera.get("horizontal_fov_deg"))
            or not 0 < camera["horizontal_fov_deg"] <= 360
            or not _number(camera.get("max_range_m"))
            or camera["max_range_m"] <= 0):
        return {"status": "WAITING_FOR_CAMERA_CONFIGURATION", "candidates": [], "ranking": []}

    width, height, resolution, origin, cells = _read_grid(grid)
    if (not isinstance(robot_pose, (list, tuple)) or len(robot_pose) != 3
            or not all(_number(value) for value in robot_pose)):
        raise ValueError("robot_pose must contain finite world x, y, yaw")
    start = world_to_cell({"origin": origin, "resolution": resolution}, robot_pose[0], robot_pose[1])
    if not _inside(start, width, height):
        return {"status": "INVALID_LOCALIZATION", "candidates": [], "ranking": []}

    safe_cells = _safe_free_cells(width, height, cells)
    edge_counts = _reachable_edge_counts(start, width, height, safe_cells)
    unknown_targets = _unknown_boundary_cells(width, height, cells)

    frontiers = []
    for y in range(height):
        for x in range(width):
            if cells[(x, y)] == FREE and any(
                    _inside((x + dx, y + dy), width, height)
                    and cells[(x + dx, y + dy)] == UNKNOWN
                    for dx, dy, _ in DIRECTIONS):
                frontiers.append((x, y))

    proposals = {}
    for frontier_x, frontier_y in frontiers:
        for dx, dy, direction in DIRECTIONS:
            unknown = frontier_x + dx, frontier_y + dy
            if not _inside(unknown, width, height) or cells[unknown] != UNKNOWN:
                continue
            cell = frontier_x - dx, frontier_y - dy
            proposals.setdefault((cell[0], cell[1], direction), []).append((frontier_x, frontier_y))

    candidates = []
    for x, y, direction in sorted(proposals, key=lambda item: (item[1], item[0], item[2])):
        cell = x, y
        edges = edge_counts.get(cell)
        cost = edges * resolution if edges is not None else None
        yaw = origin[2] + _DIRECTION_YAWS[direction]
        pose_x = origin[0] + math.cos(origin[2]) * (x + 0.5) * resolution - math.sin(origin[2]) * (y + 0.5) * resolution
        pose_y = origin[1] + math.sin(origin[2]) * (x + 0.5) * resolution + math.cos(origin[2]) * (y + 0.5) * resolution
        record = {
            "id": f"x{x}_y{y}_d{direction}",
            "cell": [x, y],
            "direction": direction,
            "source_frontiers": [list(point) for point in proposals[(x, y, direction)]],
            "pose_world": [pose_x, pose_y, math.atan2(math.sin(yaw), math.cos(yaw))],
            "cost_m": cost,
        }
        if cost is not None:
            visible = _visible_unknown(
                cell, yaw, camera, width, height, resolution, origin[2], cells, unknown_targets
            )
            record["visible_unknown_cells"] = [list(point) for point in visible]
            record["visible_gain_area_m2"] = len(visible) * resolution * resolution
            candidates.append(record)

    if not candidates:
        return {
            "status": "NO_ADMISSIBLE_CANDIDATE" if frontiers else "NO_FRONTIERS",
            "frontier_cells": [list(point) for point in frontiers],
            "candidates": [],
            "ranking": [],
        }

    max_gain = max(candidate["visible_gain_area_m2"] for candidate in candidates)
    max_cost = max(candidate["cost_m"] for candidate in candidates)
    for candidate in candidates:
        gain_score = candidate["visible_gain_area_m2"] / max_gain if max_gain else 0.0
        cost_score = candidate["cost_m"] / max_cost if max_cost else 0.0
        candidate["score"] = 0.5 * gain_score - 0.5 * cost_score
    ranking = [candidate["id"] for candidate in sorted(
        candidates,
        key=lambda item: (-item["score"], item["cost_m"], item["cell"][1], item["cell"][0], item["direction"]),
    )]
    return {
        "status": "RECOMMENDATION_AVAILABLE",
        "frontier_cells": [list(point) for point in frontiers],
        "candidates": candidates,
        "ranking": ranking,
        "recommended": ranking[0],
    }
