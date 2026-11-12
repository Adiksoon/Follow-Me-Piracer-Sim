"""Match the active upstream frontier to clusters in the current grid."""

import math


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _world_to_cell(grid, x, y):
    origin_x, origin_y, origin_yaw = grid["origin"]
    dx, dy = x - origin_x, y - origin_y
    local_x = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
    local_y = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
    return math.floor(local_x / grid["resolution"]), math.floor(local_y / grid["resolution"])


def match_active_frontiers(grid, candidate, result):
    """Return upstream clusters overlapping the saved active frontier cells."""
    try:
        resolution = grid["resolution"]
        points = candidate.get("source_frontiers_world") or candidate.get(
            "frontiers_world", []
        )
        if not points or not _number(resolution) or resolution <= 0:
            return None
        source = {_world_to_cell(grid, *point) for point in points
                  if isinstance(point, (list, tuple)) and len(point) == 2
                  and all(_number(value) for value in point)}
        nearby = {(x + dx, y + dy) for x, y in source
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        matches = []
        for cluster in result.get("cluster_candidates", []):
            cells = {_world_to_cell(grid, *point)
                     for point in cluster.get("frontiers_world", [])
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
