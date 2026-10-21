"""Apply the approved initial free-space assumption to an OccupancyGrid."""

from copy import copy
from math import atan2, cos, isfinite, sin


def _origin_yaw(origin):
    q = origin.orientation
    values = (q.x, q.y, q.z, q.w)
    if not all(isfinite(value) for value in values):
        raise ValueError("map origin quaternion must be finite")
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0.0:
        raise ValueError("map origin quaternion must be non-zero")
    x, y, z, w = (value / norm for value in values)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def apply_initial_free_zone(grid, center_x, center_y, radius):
    """Return a copied grid with unknown cell centers in a fixed circle set free.

    Existing free and occupied values are preserved. The input message and its
    metadata are not modified.
    """
    if not all(isfinite(value) for value in (center_x, center_y, radius)):
        raise ValueError("zone center and radius must be finite")
    if radius < 0.0:
        raise ValueError("zone radius must be non-negative")
    info = grid.info
    if info.width <= 0 or info.height <= 0 or not isfinite(info.resolution):
        raise ValueError("map dimensions and resolution must be valid")
    if info.resolution <= 0.0 or len(grid.data) != info.width * info.height:
        raise ValueError("map data length or resolution is invalid")

    yaw = _origin_yaw(info.origin)
    origin_x = info.origin.position.x
    origin_y = info.origin.position.y
    if not isfinite(origin_x) or not isfinite(origin_y):
        raise ValueError("map origin position must be finite")
    c, s = cos(yaw), sin(yaw)
    radius_squared = radius * radius
    data = list(grid.data)

    for row in range(info.height):
        local_y = (row + 0.5) * info.resolution
        for column in range(info.width):
            index = row * info.width + column
            if data[index] != -1:
                continue
            local_x = (column + 0.5) * info.resolution
            cell_x = origin_x + c * local_x - s * local_y
            cell_y = origin_y + s * local_x + c * local_y
            if (cell_x - center_x) ** 2 + (cell_y - center_y) ** 2 <= radius_squared:
                data[index] = 0

    result = copy(grid)
    result.data = data
    return result
