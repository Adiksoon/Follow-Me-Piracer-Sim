"""Small, ROS-independent adapter and frontier ranking core.

Input grids use OccupancyGrid data order (row-major, bottom row first) and an
origin [x, y, yaw]. Unknown and occupied cells are never traversed by paths.
"""

from collections import deque
import math

FREE, OCCUPIED, UNKNOWN = 0, 100, -1
DIRECTIONS = ((1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3))
_YAWS = (0.0, math.pi / 2, math.pi, -math.pi / 2)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def world_to_cell(grid, x, y):
    """Convert a world point to a cell, accounting for rotated map origin."""
    ox, oy, yaw = grid["origin"]
    dx, dy = x - ox, y - oy
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.floor(local_x / grid["resolution"]), math.floor(local_y / grid["resolution"])


def _read_grid(grid):
    width, height = grid.get("width"), grid.get("height")
    resolution, origin, data = grid.get("resolution"), grid.get("origin"), grid.get("data")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0 or not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("invalid map dimensions")
    if not _number(resolution) or resolution <= 0 or not isinstance(origin, (list, tuple)) or len(origin) != 3 or not all(_number(v) for v in origin):
        raise ValueError("invalid map geometry")
    if not isinstance(data, (list, tuple)) or len(data) != width * height:
        raise ValueError("map data length mismatch")
    if any(not isinstance(v, int) or isinstance(v, bool) or v not in (FREE, OCCUPIED, UNKNOWN) for v in data):
        raise ValueError("occupancy values must be -1, 0, or 100")
    cells = {(i % width, i // width): data[i] for i in range(len(data))}
    return width, height, float(resolution), list(origin), cells


def _inside(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _safe(cell, width, height, cells):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            point = cell[0] + dx, cell[1] + dy
            if not _inside(point, width, height) or cells[point] != FREE:
                return False
    return True


def _path_cost(start, goal, width, height, resolution, cells):
    """Cardinal shortest path over safe FREE cells; return meters or None."""
    pending = deque([(start, 0)])
    seen = {start}
    while pending:
        cell, edges = pending.popleft()
        if cell == goal:
            return edges * resolution
        for dx, dy, _ in DIRECTIONS:
            nxt = cell[0] + dx, cell[1] + dy
            if nxt not in seen and _inside(nxt, width, height) and _safe(nxt, width, height, cells):
                seen.add(nxt)
                pending.append((nxt, edges + 1))
    return None


def _ray(a, b):
    x, y = a
    end_x, end_y = b
    dx, sx = abs(end_x - x), 1 if x < end_x else -1
    dy, sy = -abs(end_y - y), 1 if y < end_y else -1
    err, out = dx + dy, []
    while True:
        out.append((x, y))
        if (x, y) == (end_x, end_y):
            return out
        twice = 2 * err
        if twice >= dy:
            err, x = err + dy, x + sx
        if twice <= dx:
            err, y = err + dx, y + sy


def _visible_cells(observer, yaw, target, policy, fov_deg, range_m, width, height, resolution, origin_yaw, cells):
    heading = math.atan2(math.sin(yaw - origin_yaw), math.cos(yaw - origin_yaw))
    visible = []
    for y in range(height):
        for x in range(width):
            if cells[(x, y)] != target:
                continue
            dx, dy = (x - observer[0]) * resolution, (y - observer[1]) * resolution
            distance = math.hypot(dx, dy)
            if distance == 0 or distance > range_m + 1e-12:
                continue
            angle = math.atan2(dy, dx)
            delta = math.atan2(math.sin(angle - heading), math.cos(angle - heading))
            if abs(delta) > math.radians(fov_deg) / 2 + 1e-12:
                continue
            middle = _ray(observer, (x, y))[1:-1]
            if any(cells[p] == OCCUPIED for p in middle):
                continue
            if policy == "UNKNOWN_OPAQUE" and any(cells[p] == UNKNOWN for p in middle):
                continue
            visible.append((x, y))
    return visible


def evaluate_search(grid, robot_pose, *, camera=None, freshness=None, gain_weight=0.5):
    """Return a deterministic candidate ranking for one map and robot pose.

    robot_pose is (world_x, world_y, world_yaw). Optional camera uses
    horizontal_fov_deg and max_range_m. Freshness is a mapping from
    free-cell (x, y) to last-seen seconds and requires now_s/ttl_s.
    """
    if grid is None:
        return {"status": "WAITING_FOR_MAP", "candidates": [], "ranking": []}
    if robot_pose is None:
        return {"status": "WAITING_FOR_LOCALIZATION", "candidates": [], "ranking": []}
    width, height, resolution, origin, cells = _read_grid(grid)
    if not isinstance(robot_pose, (list, tuple)) or len(robot_pose) != 3 or not all(_number(v) for v in robot_pose):
        raise ValueError("robot_pose must contain finite world x, y, yaw")
    start = world_to_cell({"origin": origin, "resolution": resolution}, robot_pose[0], robot_pose[1])
    if not _inside(start, width, height):
        return {"status": "INVALID_LOCALIZATION", "candidates": [], "ranking": []}

    frontiers = []
    for y in range(height):
        for x in range(width):
            if cells[(x, y)] == FREE and any(
                _inside((x + dx, y + dy), width, height) and cells[(x + dx, y + dy)] == UNKNOWN
                for dx, dy, _ in DIRECTIONS
            ):
                frontiers.append((x, y))

    candidates = {}
    for fx, fy in frontiers:
        for dx, dy, direction in DIRECTIONS:
            unknown = fx + dx, fy + dy
            if not _inside(unknown, width, height) or cells[unknown] != UNKNOWN:
                continue
            cell = fx - dx, fy - dy
            candidates.setdefault((cell[0], cell[1], direction), []).append((fx, fy))

    records = []
    for x, y, direction in sorted(candidates, key=lambda c: (c[1], c[0], c[2])):
        cell = (x, y)
        identifier = f"x{x}_y{y}_d{direction}"
        cost = _path_cost(start, cell, width, height, resolution, cells) if _safe(cell, width, height, cells) else None
        yaw = origin[2] + _YAWS[direction]
        record = {"id": identifier, "cell": [x, y], "direction": direction, "source_frontiers": [list(p) for p in candidates[(x, y, direction)]], "cost_m": cost, "pose_world": [origin[0] + math.cos(origin[2]) * (x + 0.5) * resolution - math.sin(origin[2]) * (y + 0.5) * resolution, origin[1] + math.sin(origin[2]) * (x + 0.5) * resolution + math.cos(origin[2]) * (y + 0.5) * resolution, math.atan2(math.sin(yaw), math.cos(yaw))], "visible_unknown_cells": [], "visible_stale_free_cells": [], "score": None}
        if cost is None:
            record["reason"] = "UNSAFE_OR_UNREACHABLE"
        else:
            record["visible_unknown_cells"] = [list(p) for p in _visible_cells(cell, yaw, UNKNOWN, "OPTIMISTIC", camera["horizontal_fov_deg"], camera["max_range_m"], width, height, resolution, origin[2], cells)] if camera else []
            if freshness:
                now, ttl = freshness.get("now_s"), freshness.get("ttl_s")
                history = freshness.get("last_seen_s", {})
                if _number(now) and _number(ttl) and ttl >= 0:
                    known = {tuple(k) if isinstance(k, list) else k: v for k, v in history.items()}
                    record["visible_stale_free_cells"] = [list(p) for p in _visible_cells(cell, yaw, FREE, "UNKNOWN_OPAQUE", camera["horizontal_fov_deg"], camera["max_range_m"], width, height, resolution, origin[2], cells) if p in known and _number(known[p]) and known[p] <= now and now - known[p] >= ttl] if camera else []
            records.append(record)

    if not records:
        return {"status": "NO_ADMISSIBLE_CANDIDATE" if frontiers else "NO_FRONTIERS", "frontier_cells": [list(p) for p in frontiers], "candidates": [], "ranking": []}
    max_cost = max(r["cost_m"] for r in records)
    max_gain = max((len(r["visible_unknown_cells"]) + len(r["visible_stale_free_cells"]) for r in records), default=0)
    for record in records:
        gain = len(record["visible_unknown_cells"]) + len(record["visible_stale_free_cells"])
        record["score"] = gain_weight * (gain / max_gain if max_gain else 0.0) - (1.0 - gain_weight) * (record["cost_m"] / max_cost if max_cost else 0.0)
    ranking = [r["id"] for r in sorted(records, key=lambda r: (-r["score"], r["cost_m"], r["cell"][1], r["cell"][0], r["direction"]))]
    return {"status": "RECOMMENDATION_AVAILABLE", "frontier_cells": [list(p) for p in frontiers], "candidates": records, "ranking": ranking, "recommended": ranking[0]}
