"""Observation-goal geometry and structural Nav2 path checks."""

import math


def observation_goal(robot, target, standoff, max_step, index, count, radius, angle, first_turn=1):
    """Choose an endpoint, leaving all path planning to Nav2.

    Reserve some approach clearance for each remaining maneuver. The second
    and third endpoints lie on forward circular arcs, toward first_turn
    (+1 left, -1 right) then the opposite side, with equal capped sweep angles.
    Arc length bounds the reduction in distance to the remembered suitcase.
    The first arc goal faces along its endpoint chord. The final goal faces
    halfway between chord and arc tangent, a
    bounded pose heuristic for the observed forward-only planner lattice.
    """
    values = (*robot, *target, standoff, max_step, radius, angle)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite observation goal input")
    if (not 1 <= count <= 3 or not 0 <= index < count
            or standoff <= 0 or max_step <= 0 or radius < 0.5 or not 0 < angle <= 0.4):
        raise ValueError("invalid observation maneuver geometry")
    if first_turn not in (-1, 1):
        raise ValueError("first turn must be +1 (left) or -1 (right)")
    x, y, yaw = robot
    dx, dy = target[0] - x, target[1] - y
    if dx * math.cos(yaw) + dy * math.sin(yaw) <= 0:
        return None
    distance = math.hypot(dx, dy)
    step = min(max_step, (distance - standoff) / (count - index))
    # The Acquire position goal checker accepts positions within 5 cm.
    # Do not request a maneuver whose entire extent is that tolerance.
    if step <= 0.10:
        return None
    if index == 0:
        return x + step * dx / distance, y + step * dy / distance, math.atan2(dy, dx)
    turn = min(angle, step / radius)
    if radius * turn <= 0.10:
        return None
    turn *= first_turn if index == 1 else -first_turn
    curvature = (1.0 if turn > 0 else -1.0) / radius
    end_yaw = yaw + turn
    end_x = x + (math.sin(end_yaw) - math.sin(yaw)) / curvature
    end_y = y + (math.cos(yaw) - math.cos(end_yaw)) / curvature
    if index == 2:
        goal_yaw = yaw + 0.75 * turn
        return end_x, end_y, math.atan2(math.sin(goal_yaw), math.cos(goal_yaw))
    return end_x, end_y, math.atan2(end_y - y, end_x - x)


def path_length(path, frame):
    """Check every pose and return the 3D polyline length in metres."""
    if path.header.frame_id != frame or len(path.poses) < 2:
        raise ValueError("empty path or incorrect frame")
    points = []
    for stamped in path.poses:
        if stamped.header.frame_id not in ("", frame):
            raise ValueError("path pose frame mismatch")
        p, q = stamped.pose.position, stamped.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite path pose")
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if abs(norm - 1.0) > 0.01:
            raise ValueError("invalid path orientation")
        points.append((p.x, p.y, p.z))
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))
