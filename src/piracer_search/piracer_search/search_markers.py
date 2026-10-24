"""Marker helpers shared by the active Search node."""

import math

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


MAX_DISPLAYED_CANDIDATES = 6


def _score_diameters(candidates, minimum=0.08, maximum=0.35):
    """Map the current candidate scores linearly to marker diameters."""
    if not candidates:
        return []
    scores = [float(candidate["score"]) for candidate in candidates]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("candidate scores must be finite")
    low, high = min(scores), max(scores)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        diameter = (minimum + maximum) / 2.0
        return [diameter] * len(scores)
    return [minimum + (score - low) * (maximum - minimum) / (high - low)
            for score in scores]


def _candidate_markers(
    candidates, selected, map_frame, stamp, max_labels=10, selected_frame=None
):
    """Create markers for the six highest-scoring candidates and selected ring."""
    markers = []
    ranked = sorted(
        enumerate(candidates), key=lambda item: (-item[1]["score"], item[0])
    )[:MAX_DISPLAYED_CANDIDATES]
    displayed_candidates = [candidate for _, candidate in ranked]
    diameters = _score_diameters(displayed_candidates)
    selected_diameter = None
    label_ids = {index for index, _ in ranked[:max_labels]}
    for (index, candidate), diameter in zip(ranked, diameters):
        x, y, yaw = candidate["pose_world"]
        if candidate is selected:
            selected_diameter = diameter
        marker = Marker()
        marker.header.frame_id = map_frame
        marker.header.stamp = stamp
        marker.ns = "frontier_candidates"
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y = x, y
        marker.pose.position.z = diameter / 2.0 + 0.02
        marker.pose.orientation.z = math.sin(yaw / 2.0)
        marker.pose.orientation.w = math.cos(yaw / 2.0)
        marker.scale.x = marker.scale.y = marker.scale.z = diameter
        marker.color.r, marker.color.g, marker.color.b = (0.1, 0.85, 0.65)
        marker.color.a = 0.9
        markers.append(marker)

        if index in label_ids:
            label = Marker()
            label.header.frame_id = map_frame
            label.header.stamp = stamp
            label.ns = "candidate_scores"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = x, y
            label.pose.position.z = 0.28
            label.pose.orientation.w = 1.0
            label.scale.z = 0.065
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"score={candidate['score']:.3f}"
            markers.append(label)

    if selected is not None:
        x, y = selected["pose_world"][:2]
        diameter = selected_diameter or 0.35
        ring = Marker()
        ring.header.frame_id = selected_frame or map_frame
        ring.header.stamp = stamp
        ring.ns = "selected_candidate_ring"
        ring.id = 0
        ring.type = Marker.LINE_STRIP
        ring.action = Marker.ADD
        ring.pose.orientation.w = 1.0
        ring.scale.x = 0.025
        ring.color.r, ring.color.g, ring.color.b, ring.color.a = (1.0, 0.72, 0.05, 1.0)
        ring_radius = diameter / 2.0 + 0.045
        ring.pose.position.z = 0.045
        for point_index in range(33):
            angle = 2.0 * math.pi * point_index / 32
            point = Point()
            point.x = x + ring_radius * math.cos(angle)
            point.y = y + ring_radius * math.sin(angle)
            ring.points.append(point)
        markers.append(ring)
    return markers


def _stale_marker_deletions(previous_keys, current_keys, map_frame, stamp):
    """Delete stale markers previously published by this Search node."""
    deletions = []
    for namespace, marker_id in sorted(previous_keys - current_keys):
        marker = Marker()
        marker.header.frame_id = map_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        deletions.append(marker)
    return deletions
