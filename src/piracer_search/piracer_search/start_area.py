"""Pure unknown-only seeding for one fixed Search startup area."""

import math


START_AREA_BOUNDS = (-0.25, 0.70, -0.25, 0.25)


def merge_initial_start_area(grid, initial_pose, bounds=START_AREA_BOUNDS):
    """Return copied grid data with wholly contained UNKNOWN cells set FREE.

    Bounds are expressed in the robot frame at the fixed initial pose in the
    map frame. Positive occupancy values and cells outside the rectangle are
    preserved. The input grid and its data are never modified.
    """
    width, height = grid["width"], grid["height"]
    resolution = grid["resolution"]
    origin_x, origin_y, origin_yaw = grid["origin"]
    pose_x, pose_y, pose_yaw = initial_pose
    min_x, max_x, min_y, max_y = bounds
    data = grid["data"]
    if (not isinstance(width, int) or width <= 0
            or not isinstance(height, int) or height <= 0
            or not isinstance(data, (list, tuple)) or len(data) != width * height
            or not math.isfinite(resolution) or resolution <= 0
            or not all(math.isfinite(value) for value in (
                origin_x, origin_y, origin_yaw, pose_x, pose_y, pose_yaw,
                min_x, max_x, min_y, max_y,
            ))
            or min_x >= max_x or min_y >= max_y):
        raise ValueError("invalid start-area map geometry")

    origin_cos, origin_sin = math.cos(origin_yaw), math.sin(origin_yaw)
    pose_cos, pose_sin = math.cos(pose_yaw), math.sin(pose_yaw)

    def in_bounds(grid_x, grid_y):
        local_x = grid_x * resolution
        local_y = grid_y * resolution
        map_x = origin_x + origin_cos * local_x - origin_sin * local_y
        map_y = origin_y + origin_sin * local_x + origin_cos * local_y
        dx, dy = map_x - pose_x, map_y - pose_y
        base_x = pose_cos * dx + pose_sin * dy
        base_y = -pose_sin * dx + pose_cos * dy
        epsilon = 1e-9
        return (min_x - epsilon <= base_x <= max_x + epsilon
                and min_y - epsilon <= base_y <= max_y + epsilon)

    merged = list(data)
    whole_cells = 0
    changed = 0
    positive_cells = 0
    for y in range(height):
        for x in range(width):
            if not all(in_bounds(x + dx, y + dy)
                       for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))):
                continue
            whole_cells += 1
            index = y * width + x
            value = data[index]
            if value == -1:
                merged[index] = 0
                changed += 1
            elif value > 0:
                positive_cells += 1
    return tuple(merged), {
        "whole_cells_inside": whole_cells,
        "unknown_changed_to_free": changed,
        "positive_cells_preserved_inside": positive_cells,
    }
