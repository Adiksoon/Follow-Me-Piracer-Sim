"""Read-only frontier evaluation and RViz marker publisher."""

import math

import rclpy
from geometry_msgs.msg import Point
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .search_algorithm import evaluate_frontier_candidates

DEFAULT_COSTMAP_TOPIC = "/global_costmap/costmap_raw"


def _map_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _costmap_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _grid_dict(message):
    origin = message.info.origin
    return {
        "width": message.info.width,
        "height": message.info.height,
        "resolution": message.info.resolution,
        "origin": [origin.position.x, origin.position.y, _yaw(origin.orientation)],
        "data": tuple(message.data),
    }


def _costmap_dict(message):
    origin = message.metadata.origin
    return {
        "width": message.metadata.size_x,
        "height": message.metadata.size_y,
        "resolution": message.metadata.resolution,
        "origin": [origin.position.x, origin.position.y, _yaw(origin.orientation)],
        "data": tuple(message.data),
    }


def _stamp_key(message):
    return message.header.stamp.sec, message.header.stamp.nanosec


def _score_diameters(candidates, minimum=0.08, maximum=0.35):
    """Map the current set of evaluator scores linearly to marker diameters."""
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


def _candidate_markers(candidates, selected, map_frame, stamp, max_labels=10):
    """Create score-sized candidate markers, bounded labels, and selected ring."""
    markers = []
    if not candidates:
        return markers
    diameters = _score_diameters(candidates)
    ranked = sorted(enumerate(candidates), key=lambda item: (-item[1]["score"], item[0]))
    label_ids = {index for index, _ in ranked[:max_labels]}
    for index, (candidate, diameter) in enumerate(zip(candidates, diameters)):
        x, y, yaw = candidate["pose_world"]
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

        if candidate is selected:
            ring = Marker()
            ring.header.frame_id = map_frame
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
                point.x, point.y = x + ring_radius * math.cos(angle), y + ring_radius * math.sin(angle)
                ring.points.append(point)
            markers.append(ring)

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
    return markers


def _stale_marker_deletions(previous_keys, current_keys, map_frame, stamp):
    """Delete only stale markers from this visualizer's own namespaces."""
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


class FrontierVisualizer(Node):
    """Evaluate one current map/costmap snapshot and publish RViz markers only."""

    def __init__(self):
        super().__init__("piracer_frontier_visualizer")
        self.declare_parameter("raw_map_topic", "/map")
        self.declare_parameter("navigation_map_topic", "/nav2/map")
        self.declare_parameter("costmap_topic", DEFAULT_COSTMAP_TOPIC)
        self.declare_parameter("marker_topic", "/piracer/frontier_markers")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("min_group_cells", 3)
        self.declare_parameter("max_candidate_labels", 10)
        self.declare_parameter("camera.horizontal_fov_deg", 65.36016047954865)
        self.declare_parameter("camera.max_range_m", 5.0)

        self._raw_map = None
        self._navigation_map = None
        self._costmap = None
        self._published_marker_keys = set()
        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._markers = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 10
        )
        self._raw_sub = self.create_subscription(
            OccupancyGrid, str(self.get_parameter("raw_map_topic").value),
            self._on_raw_map, _map_qos()
        )
        self._navigation_sub = self.create_subscription(
            OccupancyGrid, str(self.get_parameter("navigation_map_topic").value),
            self._on_navigation_map, _map_qos()
        )
        self._costmap_sub = self.create_subscription(
            Costmap, str(self.get_parameter("costmap_topic").value),
            self._on_costmap, _costmap_qos()
        )
        self._timer = self.create_timer(1.0, self._update)

    def _on_raw_map(self, message):
        self._raw_map = message

    def _on_navigation_map(self, message):
        self._navigation_map = message

    def _on_costmap(self, message):
        self._costmap = message

    def _update(self):
        if self._raw_map is None or self._navigation_map is None or self._costmap is None:
            return
        if self._raw_map.header.frame_id != self._navigation_map.header.frame_id:
            self.get_logger().warning("Raw and navigation maps use different frames")
            return
        if _stamp_key(self._raw_map) != _stamp_key(self._navigation_map):
            self.get_logger().debug("Waiting for raw and derived maps with matching stamps")
            return
        map_frame = str(self.get_parameter("map_frame").value)
        if (self._navigation_map.header.frame_id != map_frame
                or self._costmap.header.frame_id != map_frame):
            self.get_logger().warning("Map and global costmap must use the configured map frame")
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                map_frame, str(self.get_parameter("base_frame").value), Time()
            )
        except TransformException as error:
            self.get_logger().debug(f"Waiting for current robot pose in map: {error}")
            return

        translation = transform.transform.translation
        robot_pose = (translation.x, translation.y, _yaw(transform.transform.rotation))
        footprint = ((-0.020, -0.1085), (0.246, -0.1085),
                     (0.246, 0.1085), (-0.020, 0.1085))
        camera = {
            "horizontal_fov_deg": float(
                self.get_parameter("camera.horizontal_fov_deg").value
            ),
            "max_range_m": float(self.get_parameter("camera.max_range_m").value),
        }
        try:
            result = evaluate_frontier_candidates(
                _grid_dict(self._raw_map), _grid_dict(self._navigation_map),
                _costmap_dict(self._costmap), robot_pose,
                footprint=footprint, camera=camera,
                min_group_cells=int(self.get_parameter("min_group_cells").value),
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f"Frontier evaluation unavailable: {error}")
            return
        self._publish_markers(result, robot_pose, map_frame)

    def _publish_markers(self, result, robot_pose, map_frame):
        now = self.get_clock().now().to_msg()
        array = MarkerArray()

        resolution = self._navigation_map.info.resolution
        origin = _grid_dict(self._navigation_map)["origin"]

        def center(cell):
            x, y = cell
            local_x, local_y = (x + 0.5) * resolution, (y + 0.5) * resolution
            return (origin[0] + math.cos(origin[2]) * local_x - math.sin(origin[2]) * local_y,
                    origin[1] + math.sin(origin[2]) * local_x + math.cos(origin[2]) * local_y)

        marker_id = 0
        for group in result.get("groups", []):
            marker = Marker()
            marker.header.frame_id = map_frame
            marker.header.stamp = now
            marker.ns = "frontier_groups"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = max(0.015, resolution * 0.65)
            marker.scale.z = 0.025
            assumption = group.get("source_is_assumption_based", False)
            if group["status"] == "candidate_available":
                marker.color.r, marker.color.g, marker.color.b = ((1.0, 0.55, 0.05)
                    if assumption else (0.1, 0.9, 0.2))
            else:
                marker.color.r, marker.color.g, marker.color.b = (0.95, 0.1, 0.1)
            marker.color.a = 0.95
            for cell in group["cells"]:
                x, y = center(cell)
                point = Point()
                point.x, point.y, point.z = x, y, 0.04
                marker.points.append(point)
            array.markers.append(marker)

            label_pose = group.get("recommended")
            label_x, label_y = (center(label_pose["cell"]) if label_pose else center(group["cells"][0]))
            label = Marker()
            label.header.frame_id = map_frame
            label.header.stamp = now
            label.ns = "frontier_reasons"
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = label_x, label_y
            label.pose.position.z = 0.18
            label.pose.orientation.w = 1.0
            label.scale.z = 0.075
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"{group['id']}: {group['reason']}"
            array.markers.append(label)

        candidates = result.get("candidates", [])
        array.markers.extend(_candidate_markers(
            candidates, result.get("selected"), map_frame, now,
            max_labels=int(self.get_parameter("max_candidate_labels").value),
        ))

        robot = Marker()
        robot.header.frame_id = map_frame
        robot.header.stamp = now
        robot.ns = "robot_pose"
        robot.id = 0
        robot.type = Marker.ARROW
        robot.action = Marker.ADD
        robot.pose.position.x, robot.pose.position.y = robot_pose[:2]
        robot.pose.orientation.z = math.sin(robot_pose[2] / 2.0)
        robot.pose.orientation.w = math.cos(robot_pose[2] / 2.0)
        robot.scale.x, robot.scale.y, robot.scale.z = (0.30, 0.06, 0.06)
        robot.color.r, robot.color.g, robot.color.b, robot.color.a = (0.1, 0.8, 1.0, 1.0)
        array.markers.append(robot)
        current_keys = {(marker.ns, marker.id) for marker in array.markers
                        if marker.action == Marker.ADD}
        array.markers.extend(_stale_marker_deletions(
            self._published_marker_keys, current_keys, map_frame, now
        ))
        self._published_marker_keys = current_keys
        self._markers.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
