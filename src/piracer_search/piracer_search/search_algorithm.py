"""Pure-Python OccupancyGrid adapter and deterministic frontier ranking."""

from collections import deque
import math

FREE, OCCUPIED, UNKNOWN = 0, 100, -1
DIRECTIONS = ((1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3))
_DIRECTION_YAWS = (0.0, math.pi / 2, math.pi, -math.pi / 2)
# Keep this in sync with the global and local costmap footprint in
# piracer_navigation/config/nav2_params.yaml. Padding is zero in that config.
PIRACER_FOOTPRINT = (
    (-0.020, -0.1085), (0.246, -0.1085),
    (0.246, 0.1085), (-0.020, 0.1085),
)
# Keep fallback local to about one PiRacer body length.
MAX_VIEWPOINT_SHIFT_M = 0.30
_GEOMETRY_EPS = 1e-10


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def world_to_cell(grid, x, y):
    """Convert a world point to a cell using the rotated map origin."""
    origin_x, origin_y, origin_yaw = grid["origin"]
    dx, dy = x - origin_x, y - origin_y
    local_x = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
    local_y = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
    return math.floor(local_x / grid["resolution"]), math.floor(local_y / grid["resolution"])


def match_active_frontiers(grid, candidate, result):
    """Return current clusters overlapping the saved active-frontier cells."""
    try:
        resolution = grid["resolution"]
        points = candidate.get("source_frontiers_world") or [
            candidate["source_frontier_world"]
        ]
        points = [point for point in points if isinstance(point, (list, tuple))
                  and len(point) == 2 and all(_number(value) for value in point)]
        if not points or not _number(resolution) or resolution <= 0:
            return None
        source = {world_to_cell(grid, *point) for point in points}
        nearby = {(x + dx, y + dy) for x, y in source
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        matches = []
        for cluster in result.get("cluster_candidates", []):
            frontier_points = cluster.get("frontiers_world", [])
            cells = {world_to_cell(grid, *point) for point in frontier_points
                     if isinstance(point, (list, tuple)) and len(point) == 2
                     and all(_number(value) for value in point)}
            if nearby.intersection(cells):
                if (not _number(cluster.get("score"))
                        or not isinstance(cluster.get("cluster_id"), str)
                        or not cluster["cluster_id"]):
                    return None
                matches.append(cluster)
        return matches or None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


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


def _cells_under_footprint(width, height, resolution, origin, pose, yaw, footprint):
    """Return map cells touched by the full rotated footprint, or None if out of map."""
    origin_x, origin_y, origin_yaw = origin
    pose_x, pose_y = pose
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_origin, sin_origin = math.cos(origin_yaw), math.sin(origin_yaw)
    polygon = []
    for local_x, local_y in footprint:
        world_x = pose_x + cos_yaw * local_x - sin_yaw * local_y
        world_y = pose_y + sin_yaw * local_x + cos_yaw * local_y
        dx, dy = world_x - origin_x, world_y - origin_y
        map_x = (cos_origin * dx + sin_origin * dy) / resolution
        map_y = (-sin_origin * dx + cos_origin * dy) / resolution
        polygon.append((map_x, map_y))

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    if (min_x < -_GEOMETRY_EPS or min_y < -_GEOMETRY_EPS
            or max_x > width + _GEOMETRY_EPS
            or max_y > height + _GEOMETRY_EPS):
        return None

    # Include boundary-touching cells conservatively. Clamp to map bounds because
    # a footprint that only touches the outer map edge is still inside the map.
    first_x = max(0, math.floor(min_x) - 1)
    last_x = min(width, math.ceil(max_x) + 1)
    first_y = max(0, math.floor(min_y) - 1)
    last_y = min(height, math.ceil(max_y) + 1)
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        edge_x = next_point[0] - point[0]
        edge_y = next_point[1] - point[1]
        axes.append((-edge_y, edge_x))

    touched = []
    for cell_y in range(first_y, last_y):
        for cell_x in range(first_x, last_x):
            square = (
                (cell_x, cell_y), (cell_x + 1, cell_y),
                (cell_x + 1, cell_y + 1), (cell_x, cell_y + 1),
            )
            separated = False
            for axis_x, axis_y in axes:
                polygon_projection = [
                    x * axis_x + y * axis_y for x, y in polygon
                ]
                square_projection = [
                    x * axis_x + y * axis_y for x, y in square
                ]
                if (max(polygon_projection) < min(square_projection) - _GEOMETRY_EPS
                        or max(square_projection)
                        < min(polygon_projection) - _GEOMETRY_EPS):
                    separated = True
                    break
            if not separated:
                touched.append((cell_x, cell_y))
    return touched or None


def footprint_is_free(grid, pose, *, footprint=PIRACER_FOOTPRINT):
    """Return whether the complete footprint at world (x, y, yaw) is known FREE."""
    width, height, resolution, origin, cells = _read_grid(grid)
    if (not isinstance(pose, (list, tuple)) or len(pose) != 3
            or not all(_number(value) for value in pose)):
        raise ValueError("pose must contain finite world x, y, yaw")
    touched = _cells_under_footprint(
        width, height, resolution, origin, pose[:2], pose[2], footprint
    )
    return touched is not None and all(cells[cell] == FREE for cell in touched)


def _approach_yaw(robot_pose, pose_x, pose_y):
    dx, dy = pose_x - robot_pose[0], pose_y - robot_pose[1]
    yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else robot_pose[2]
    return math.atan2(math.sin(yaw), math.cos(yaw))


def _nearby_reachable_cells(seed, width, height, resolution, safe_cells, edge_counts):
    """List nearby reachable FREE cells nearest first, within one robot length."""
    radius = math.ceil(MAX_VIEWPOINT_SHIFT_M / resolution)
    limit_squared = (MAX_VIEWPOINT_SHIFT_M / resolution) ** 2 + _GEOMETRY_EPS
    options = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cell = seed[0] + dx, seed[1] + dy
            distance_squared = dx * dx + dy * dy
            if (distance_squared <= limit_squared
                    and _inside(cell, width, height)
                    and cell in safe_cells and cell in edge_counts):
                options.append((distance_squared, cell[1], cell[0], cell))
    return [item[3] for item in sorted(options)]


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


def _frontier_clusters(frontiers):
    remaining = set(frontiers)
    clusters = []
    while remaining:
        first = min(remaining, key=lambda cell: (cell[1], cell[0]))
        remaining.remove(first)
        pending, cluster = [first], []
        while pending:
            cell = pending.pop()
            cluster.append(cell)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        clusters.append(sorted(cluster, key=lambda cell: (cell[1], cell[0])))
    return clusters


def _cell_world(origin, resolution, cell):
    lx, ly = (cell[0] + 0.5) * resolution, (cell[1] + 0.5) * resolution
    return [origin[0] + math.cos(origin[2]) * lx - math.sin(origin[2]) * ly,
            origin[1] + math.sin(origin[2]) * lx + math.cos(origin[2]) * ly]


def _record_id(point):
    return f"x{point[0]:.6f}_y{point[1]:.6f}"


def _frontiers(width, height, cells):
    return [(x, y) for y in range(height) for x in range(width)
            if cells[(x, y)] == FREE and any(
                _inside((x + dx, y + dy), width, height)
                and cells[(x + dx, y + dy)] == UNKNOWN for dx, dy, _ in DIRECTIONS)]


def _cluster_options(cluster, width, height, resolution, origin, cells, safe_cells, edge_counts):
    options = []
    for fx, fy in cluster:
        for dx, dy, direction in DIRECTIONS:
            unknown = fx + dx, fy + dy
            if not _inside(unknown, width, height) or cells[unknown] != UNKNOWN:
                continue
            seed = (fx - dx, fy - dy)
            seed_world = _cell_world(origin, resolution, seed)
            for cell in _nearby_reachable_cells(seed, width, height, resolution, safe_cells, edge_counts):
                point = _cell_world(origin, resolution, cell)
                dist = math.hypot(point[0] - seed_world[0], point[1] - seed_world[1])
                options.append((edge_counts[cell] * resolution, dist, cell[1], cell[0],
                                cell, (fx, fy), direction, seed, seed_world))
    # Keep one deterministic access source per reachable point before footprint work.
    by_cell = {}
    for option in sorted(options):
        by_cell.setdefault(option[4], option)
    return sorted(by_cell.values())


def _candidate_for(cluster_record, option, robot_pose, width, height, resolution, origin, cells):
    _, _, _, _, cell, frontier, direction, seed, seed_world = option
    point = _cell_world(origin, resolution, cell)
    yaw = _approach_yaw(robot_pose, *point)
    touched = _cells_under_footprint(width, height, resolution, origin, point, yaw, PIRACER_FOOTPRINT)
    if touched is None or any(cells[c] != FREE for c in touched):
        return None
    cost = cluster_record["edge_counts"][cell] * resolution
    yaw_direction = math.atan2(math.sin(origin[2] + _DIRECTION_YAWS[direction]),
                               math.cos(origin[2] + _DIRECTION_YAWS[direction]))
    return {"id": _record_id(point), "cell": list(cell), "direction": direction,
            "pose_world": [point[0], point[1], yaw_direction], "goal_yaw": yaw,
            "cost_m": cluster_record["distance_m"], "goal_cost_m": cost,
            "cluster_access_distance_m": cluster_record["distance_m"],
            "cluster_size": cluster_record["size"], "score": cluster_record["score"],
            "cluster_id": cluster_record["cluster_id"],
            "source_frontier_world": _cell_world(origin, resolution, frontier),
            "source_frontier_cell": list(frontier), "source_seed_world": seed_world,
            "source_frontiers_world": cluster_record["frontiers_world"]}


def evaluate_search(grid, robot_pose):
    """Rank frontier clusters by boundary-cell gain and BFS cluster-access distance."""
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
    safe_cells = _safe_free_cells(width, height, cells)
    edge_counts = _reachable_edge_counts(start, width, height, safe_cells)
    frontiers = _frontiers(width, height, cells)
    coarse = []
    for cluster in _frontier_clusters(frontiers):
        options = _cluster_options(cluster, width, height, resolution, origin, cells, safe_cells, edge_counts)
        if options:
            access = min(item[0] for item in options)
            coarse.append({"size": len(cluster), "distance_m": access, "options": options,
                           "frontiers_world": [_cell_world(origin, resolution, c) for c in cluster],
                           "anchor_world": _cell_world(origin, resolution, cluster[0])})
    if not coarse:
        return {"status": "NO_ADMISSIBLE_CANDIDATE" if frontiers else "NO_FRONTIERS",
                "frontier_cells": [list(c) for c in frontiers], "candidates": [], "ranking": [],
                "clustered": True, "cluster_candidates": []}
    max_gain = max(c["size"] for c in coarse)
    max_distance = max(c["distance_m"] for c in coarse)
    for c in coarse:
        c["score"] = 0.5 * (c["size"] / max_gain if max_gain else 0.0) - 0.5 * (c["distance_m"] / max_distance if max_distance else 0.0)
        c["cluster_id"] = _record_id(c["anchor_world"])
        best = c["options"][0]
        point = _cell_world(origin, resolution, best[4])
        c["id"] = _record_id(point)
        c["cell"] = list(best[4])
        c["direction"] = best[6]
        c["pose_world"] = [point[0], point[1], _approach_yaw(robot_pose, *point)]
        c["goal_yaw"] = c["pose_world"][2]
        c["goal_cost_m"] = best[0]
        c["cost_m"] = c["distance_m"]
        c["cluster_size"] = c["size"]
        c["source_frontier_world"] = _cell_world(origin, resolution, best[5])
        c["source_seed_world"] = list(best[8])
    ranked = sorted(coarse, key=lambda c: (-c["score"], c["distance_m"], c["anchor_world"][1], c["anchor_world"][0]))
    cluster_candidates = []
    for c in ranked:
        public = {k: v for k, v in c.items() if k != "options"}
        public["access_options"] = []
        seen_access = set()
        for option in c["options"]:
            _, _, _, _, cell, frontier, direction, seed, seed_world = option
            key = (tuple(seed_world), tuple(_cell_world(origin, resolution, frontier)), direction)
            if key in seen_access:
                continue
            seen_access.add(key)
            public["access_options"].append({"source_seed_world": list(seed_world),
                                             "source_frontier_world": _cell_world(origin, resolution, frontier),
                                             "direction": direction})
        cluster_candidates.append(public)
    ready = []
    for c in ranked:
        private = {"edge_counts": edge_counts}
        for option in c["options"]:
            record = _candidate_for(private | c, option, robot_pose, width, height, resolution, origin, cells)
            if record is not None:
                if record["id"] in {item["id"] for item in ready}:
                    continue
                ready.append(record)
                break
        if len(ready) >= 3:
            break
    return {"status": "RECOMMENDATION_AVAILABLE" if ready else "NO_ADMISSIBLE_CANDIDATE",
            "frontier_cells": [list(c) for c in frontiers], "candidates": ready,
            "ranking": [c["id"] for c in ready], "recommended": ready[0]["id"] if ready else None,
            "clustered": True, "cluster_candidates": cluster_candidates}


def revalidate_candidate(grid, robot_pose, candidate):
    """Recheck a cached candidate against current map geometry and local frontier metadata."""
    try:
        if not isinstance(candidate, dict) or not all(k in candidate for k in
                ("source_frontier_world", "source_seed_world", "pose_world", "score", "cluster_size")):
            return None
        if grid is None or robot_pose is None:
            return None
        width, height, resolution, origin, cells = _read_grid(grid)
        if not isinstance(robot_pose, (list, tuple)) or len(robot_pose) != 3 or not all(_number(v) for v in robot_pose):
            return None
        if not all(_number(v) for v in candidate["source_frontier_world"] + candidate["source_seed_world"]):
            return None
        safe = _safe_free_cells(width, height, cells)
        start = world_to_cell({"origin": origin, "resolution": resolution}, robot_pose[0], robot_pose[1])
        if not _inside(start, width, height):
            return None
        edges = _reachable_edge_counts(start, width, height, safe)
        source_options = candidate.get("access_options") or [candidate]
        for source in source_options:
            seed_world = source.get("source_seed_world")
            frontier_world = source.get("source_frontier_world")
            if not isinstance(seed_world, (list, tuple)) or not isinstance(frontier_world, (list, tuple)):
                continue
            source_cell = world_to_cell({"origin": origin, "resolution": resolution}, *frontier_world)
            if (not _inside(source_cell, width, height) or cells[source_cell] != FREE
                    or not any(_inside((source_cell[0]+dx, source_cell[1]+dy), width, height)
                               and cells[(source_cell[0]+dx, source_cell[1]+dy)] == UNKNOWN
                               for dx, dy, _ in DIRECTIONS)):
                continue
            seed = world_to_cell({"origin": origin, "resolution": resolution}, *seed_world)
            choices = _nearby_reachable_cells(seed, width, height, resolution, safe, edges)
            for cell in choices:
                point = _cell_world(origin, resolution, cell)
                if math.dist(point, seed_world) > MAX_VIEWPOINT_SHIFT_M + 1e-9:
                    continue
                yaw = _approach_yaw(robot_pose, *point)
                touched = _cells_under_footprint(width, height, resolution, origin, point, yaw, PIRACER_FOOTPRINT)
                if touched is None or any(cells[c] != FREE for c in touched):
                    continue
                fresh = dict(candidate)
                fresh.update({"id": _record_id(point), "cell": list(cell), "pose_world": [*point, yaw],
                              "goal_yaw": yaw, "goal_cost_m": edges[cell] * resolution,
                              "source_seed_world": list(seed_world),
                              "source_frontier_world": list(frontier_world)})
                return fresh
        return None
    except (KeyError, TypeError, ValueError, IndexError):
        return None
