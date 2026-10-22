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


def _safe(cell, width, height, cells):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor = cell[0] + dx, cell[1] + dy
            if not _inside(neighbor, width, height) or cells[neighbor] != FREE:
                return False
    return True


def _path_cost(start, goal, width, height, resolution, cells):
    """Return cardinal path length over safe FREE cells, in meters."""
    if not _safe(start, width, height, cells):
        return None
    pending = deque([(start, 0)])
    visited = {start}
    while pending:
        cell, edges = pending.popleft()
        if cell == goal:
            return edges * resolution
        for dx, dy, _ in DIRECTIONS:
            neighbor = cell[0] + dx, cell[1] + dy
            if (neighbor not in visited and _inside(neighbor, width, height)
                    and _safe(neighbor, width, height, cells)):
                visited.add(neighbor)
                pending.append((neighbor, edges + 1))
    return None


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


def _visible_unknown(observer, yaw, camera, width, height, resolution, origin_yaw, cells):
    """Count UNKNOWN cells under the frozen UNKNOWN_OPAQUE visibility rule."""
    fov = camera["horizontal_fov_deg"]
    max_range = camera["max_range_m"]
    heading = math.atan2(math.sin(yaw - origin_yaw), math.cos(yaw - origin_yaw))
    visible = []
    for y in range(height):
        for x in range(width):
            if cells[(x, y)] != UNKNOWN:
                continue
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
        cost = _path_cost(start, cell, width, height, resolution, cells) if _safe(cell, width, height, cells) else None
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
            visible = _visible_unknown(cell, yaw, camera, width, height, resolution, origin[2], cells)
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


def _cell_center_world(cell, resolution, origin):
    x, y = cell
    ox, oy, yaw = origin
    local_x, local_y = (x + 0.5) * resolution, (y + 0.5) * resolution
    return (ox + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            oy + math.sin(yaw) * local_x + math.cos(yaw) * local_y)


def _world_to_cell_geometry(x, y, resolution, origin):
    ox, oy, yaw = origin
    dx, dy = x - ox, y - oy
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.floor(local_x / resolution), math.floor(local_y / resolution)


def _polygons_intersect(first, second):
    for polygon in (first, second):
        for index, point in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            axis = (-(following[1] - point[1]), following[0] - point[0])
            first_projection = [p[0] * axis[0] + p[1] * axis[1] for p in first]
            second_projection = [p[0] * axis[0] + p[1] * axis[1] for p in second]
            if (max(first_projection) < min(second_projection) - 1e-10
                    or max(second_projection) < min(first_projection) - 1e-10):
                return False
    return True


def _footprint_cells(center, yaw, footprint, width, height, resolution, origin):
    world_polygon = [
        (center[0] + math.cos(yaw) * x - math.sin(yaw) * y,
         center[1] + math.sin(yaw) * x + math.cos(yaw) * y)
        for x, y in footprint
    ]
    local_polygon = []
    ox, oy, map_yaw = origin
    for x, y in world_polygon:
        dx, dy = x - ox, y - oy
        local_x = math.cos(map_yaw) * dx + math.sin(map_yaw) * dy
        local_y = -math.sin(map_yaw) * dx + math.cos(map_yaw) * dy
        local_polygon.append((local_x / resolution, local_y / resolution))
    min_x = math.floor(min(point[0] for point in local_polygon))
    max_x = math.floor(max(point[0] for point in local_polygon))
    min_y = math.floor(min(point[1] for point in local_polygon))
    max_y = math.floor(max(point[1] for point in local_polygon))
    cells = []
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            if not (0 <= cell_x < width and 0 <= cell_y < height):
                return None
            square = [(cell_x, cell_y), (cell_x + 1, cell_y),
                      (cell_x + 1, cell_y + 1), (cell_x, cell_y + 1)]
            if _polygons_intersect(local_polygon, square):
                cells.append((cell_x, cell_y))
    return cells


def _costmap_value(world_x, world_y, costmap):
    width, height, resolution, origin, data = costmap
    cell = _world_to_cell_geometry(world_x, world_y, resolution, origin)
    if not (0 <= cell[0] < width and 0 <= cell[1] < height):
        return None
    return data[cell[1] * width + cell[0]]


def evaluate_frontier_candidates(
    raw_grid,
    navigation_grid,
    costmap,
    robot_pose,
    *,
    footprint,
    camera,
    min_group_cells=3,
):
    """Explain frontier candidates without issuing navigation or motion commands.

    Grid dictionaries use OccupancyGrid values and origin [x, y, yaw]. Costmap
    dictionaries use Nav2 uint8 costs and the same geometry fields. The returned
    connectivity is a center-cell grid check, not an Ackermann path guarantee.
    """
    if navigation_grid is None or costmap is None:
        return {"status": "WAITING_FOR_NAVIGATION_COSTMAP", "groups": [], "selected": None}
    if (not isinstance(min_group_cells, int) or isinstance(min_group_cells, bool)
            or min_group_cells < 1):
        raise ValueError("min_group_cells must be a positive integer")
    if (not isinstance(footprint, (list, tuple)) or len(footprint) < 3
            or any(not isinstance(point, (list, tuple)) or len(point) != 2
                   or not all(_number(value) for value in point) for point in footprint)):
        raise ValueError("footprint must contain at least three finite 2D points")
    if (not isinstance(robot_pose, (list, tuple)) or len(robot_pose) != 3
            or not all(_number(value) for value in robot_pose)):
        raise ValueError("robot_pose must contain finite world x, y, yaw")
    if (not isinstance(camera, dict)
            or not _number(camera.get("horizontal_fov_deg"))
            or not 0 < camera["horizontal_fov_deg"] <= 360
            or not _number(camera.get("max_range_m"))
            or camera["max_range_m"] <= 0):
        raise ValueError("camera configuration must contain a valid FOV and range")

    raw_width, raw_height, raw_resolution, raw_origin, raw_cells = _read_grid(raw_grid)
    width, height, resolution, origin, cells = _read_grid(navigation_grid)
    if (raw_width, raw_height) != (width, height) or not math.isclose(
        raw_resolution, resolution, rel_tol=0.0, abs_tol=1e-9
    ) or any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
             for a, b in zip(raw_origin, origin)):
        return {"status": "MAP_GEOMETRY_MISMATCH", "groups": [], "selected": None}
    if any((raw_cells[cell] == FREE and cells[cell] != FREE)
           or (raw_cells[cell] == OCCUPIED and cells[cell] != OCCUPIED)
           or (raw_cells[cell] == UNKNOWN and cells[cell] == OCCUPIED)
           for cell in raw_cells):
        return {"status": "MAP_SEMANTIC_MISMATCH", "groups": [], "selected": None}
    if (not isinstance(costmap, dict) or not isinstance(costmap.get("width"), int)
            or not isinstance(costmap.get("height"), int)
            or not _number(costmap.get("resolution"))
            or costmap["resolution"] <= 0
            or not isinstance(costmap.get("origin"), (list, tuple))
            or len(costmap["origin"]) != 3
            or not all(_number(value) for value in costmap["origin"])
            or not isinstance(costmap.get("data"), (list, tuple))
            or len(costmap["data"]) != costmap["width"] * costmap["height"]
            or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
                   for value in costmap["data"])):
        raise ValueError("invalid Nav2 costmap")
    costs = (costmap["width"], costmap["height"], float(costmap["resolution"]),
             list(costmap["origin"]), costmap["data"])
    if (not math.isclose(costs[2], resolution, rel_tol=0.0, abs_tol=1e-9)
            or any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
                   for a, b in zip(costs[3], origin))):
        return {"status": "COSTMAP_GEOMETRY_MISMATCH", "groups": [], "selected": None}
    # Nav2 cost values 253, 254 and 255 denote inscribed, lethal and unknown.

    frontier_cells = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if cells[(x, y)] == FREE and any(
            _inside((x + dx, y + dy), width, height)
            and cells[(x + dx, y + dy)] == UNKNOWN
            for dx, dy, _ in DIRECTIONS
        )
    }
    groups = []
    unassigned = set(frontier_cells)
    while unassigned:
        seed = min(unassigned, key=lambda cell: (cell[1], cell[0]))
        pending, group_cells = [seed], set()
        unassigned.remove(seed)
        while pending:
            cell = pending.pop()
            group_cells.add(cell)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in unassigned:
                        unassigned.remove(neighbor)
                        pending.append(neighbor)
        groups.append(sorted(group_cells, key=lambda cell: (cell[1], cell[0])))

    start = _world_to_cell_geometry(robot_pose[0], robot_pose[1], resolution, origin)
    if not _inside(start, width, height) or cells[start] != FREE:
        return {"status": "INVALID_LOCALIZATION", "frontier_cells": [list(c) for c in sorted(frontier_cells)],
                "groups": [], "selected": None}

    def path_cost_at(cell):
        wx, wy = _cell_center_world(cell, resolution, origin)
        value = _costmap_value(wx, wy, costs)
        return value if value is not None and value < 253 else None

    start_cost = path_cost_at(start)
    if start_cost is None:
        return {"status": "START_BLOCKED_BY_COSTMAP", "frontier_cells": [list(c) for c in sorted(frontier_cells)],
                "groups": [], "selected": None}
    distances = {start: 0}
    assumption_path = {start: raw_cells[start] != FREE}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for dx, dy, _ in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if (neighbor in distances or not _inside(neighbor, width, height)
                    or cells[neighbor] != FREE or path_cost_at(neighbor) is None):
                continue
            distances[neighbor] = distances[current] + 1
            assumption_path[neighbor] = assumption_path[current] or raw_cells[neighbor] != FREE
            pending.append(neighbor)

    max_forward = max(point[0] for point in footprint)
    inward_cells = max(1, math.ceil((max_forward + resolution) / resolution))
    candidate_records = []
    group_records = []
    for group_index, group in enumerate(groups):
        record = {
            "id": f"frontier_{group_index}",
            "cells": [list(cell) for cell in group],
            "status": "rejected",
            "reason": "too_small" if len(group) < min_group_cells else None,
            "source_is_assumption_based": any(raw_cells[cell] != FREE for cell in group),
            "candidates": [],
        }
        if len(group) < min_group_cells:
            group_records.append(record)
            continue
        if not any(cell in distances for cell in group):
            record["reason"] = "disconnected_from_robot_free_space"
            group_records.append(record)
            continue

        options = []
        for boundary in group:
            for dx, dy, direction in DIRECTIONS:
                unknown = (boundary[0] + dx, boundary[1] + dy)
                if not _inside(unknown, width, height) or cells[unknown] != UNKNOWN:
                    continue
                target = (boundary[0] - dx * inward_cells,
                          boundary[1] - dy * inward_cells)
                if not _inside(target, width, height) or cells[target] != FREE:
                    continue
                if target not in distances:
                    continue
                yaw = origin[2] + _DIRECTION_YAWS[direction]
                center = _cell_center_world(target, resolution, origin)
                covered = _footprint_cells(center, yaw, footprint, width, height,
                                           resolution, origin)
                if covered is None or any(cells[cell] != FREE for cell in covered):
                    continue
                footprint_costs = []
                footprint_synthetic = False
                blocked = False
                for cell in covered:
                    cell_center = _cell_center_world(cell, resolution, origin)
                    cost = _costmap_value(cell_center[0], cell_center[1], costs)
                    if cost is None or cost >= 253:
                        blocked = True
                        break
                    footprint_costs.append(cost)
                    footprint_synthetic = footprint_synthetic or raw_cells[cell] != FREE
                if blocked:
                    continue
                visible = _visible_unknown(target, yaw, camera, width, height,
                                           resolution, origin[2], cells)
                gain = len(visible) * resolution * resolution
                record_candidate = {
                    "cell": list(target),
                    "pose_world": [center[0], center[1],
                                   math.atan2(math.sin(yaw), math.cos(yaw))],
                    "frontier_cell": list(boundary),
                    "grid_connected": True,
                    "footprint_clear": True,
                    "costmap_checked": True,
                    "max_footprint_cost": max(footprint_costs, default=0),
                    "path_uses_start_zone_assumption": assumption_path[target],
                    "footprint_uses_start_zone_assumption": footprint_synthetic,
                    "connectivity_status": "ASSUMPTION_BASED" if assumption_path[target]
                    else "OBSERVED_FREE_PATH",
                    "ackermann_feasibility": "NOT_EVALUATED",
                    "visible_unknown_cells": [list(point) for point in visible],
                    "visible_gain_area_m2": gain,
                    "grid_distance_m": distances[target] * resolution,
                }
                options.append(record_candidate)
        if not options:
            record["reason"] = "no_connected_standoff_pose_passes_footprint_and_costmap_checks"
            group_records.append(record)
            continue
        record["status"] = "candidate_available"
        record["reason"] = "connected_free_side_pose_passes_endpoint_footprint_and_costmap_checks"
        record["candidates"] = options
        for candidate in options:
            candidate["frontier_group"] = record["id"]
            candidate_records.append(candidate)
        group_records.append(record)

    max_gain = max((item["visible_gain_area_m2"] for item in candidate_records), default=0.0)
    max_distance = max((item["grid_distance_m"] for item in candidate_records), default=0.0)
    for candidate in candidate_records:
        gain_score = candidate["visible_gain_area_m2"] / max_gain if max_gain else 0.0
        distance_score = candidate["grid_distance_m"] / max_distance if max_distance else 0.0
        candidate["score_terms"] = {
            "normalized_visible_gain": gain_score,
            "normalized_grid_distance": distance_score,
            "max_footprint_cost_tiebreak": candidate["max_footprint_cost"],
            "weights": {"gain": 0.50, "distance": -0.50},
        }
        candidate["score"] = 0.5 * gain_score - 0.5 * distance_score
    candidate_records.sort(key=lambda item: (-item["score"], item["max_footprint_cost"],
                                               item["grid_distance_m"],
                                               item["cell"][1], item["cell"][0],
                                               item["frontier_cell"][1], item["frontier_cell"][0]))
    selected = candidate_records[0] if candidate_records else None
    for group in group_records:
        group_options = [item for item in group["candidates"]]
        if group_options:
            group["recommended"] = min(
                group_options,
                key=lambda item: (-item["score"], item["max_footprint_cost"],
                                  item["grid_distance_m"], item["cell"][1], item["cell"][0]),
            )
    if selected is not None:
        selected["selected"] = True
        selected["selection_reason"] = (
            "highest normalized visible-gain minus grid-distance score; ties use "
            "lower maximum footprint cost, shorter distance, then cell order"
        )
    return {
        "status": "CANDIDATES_AVAILABLE" if selected else ("NO_FRONTIERS" if not groups else "NO_ADMISSIBLE_CANDIDATE"),
        "frontier_cells": [list(cell) for cell in sorted(frontier_cells, key=lambda cell: (cell[1], cell[0]))],
        "groups": group_records,
        "candidates": candidate_records,
        "selected": selected,
        "connectivity_basis": "4-connected navigation-grid FREE center cells with non-lethal known costmap cells",
        "ackermann_feasibility": "NOT_EVALUATED",
    }
