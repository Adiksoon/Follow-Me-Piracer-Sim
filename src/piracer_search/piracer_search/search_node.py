"""ROS adapter that owns Search START/STOP and one Nav2 goal at a time."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener

from .search_algorithm import evaluate_search


def _map_subscription_qos():
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class SearchNode(Node):
    def __init__(self):
        super().__init__("piracer_search")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("start_stop_service", "/search/enable")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("behavior_tree", "")
        self.declare_parameter("base_frame", "base_link")
        # Simulation settings only; physical camera calibration is TBD.
        self.declare_parameter("camera.horizontal_fov_deg", 65.36016047954865)
        self.declare_parameter("camera.max_range_m", 5.0)

        self._enabled = False
        self._generation = 0
        self._map = None
        self._frontier_key = None
        self._attempted = set()
        self._accept_pending = False
        self._goal_handle = None
        self._goal_id = None
        self._cancel_requested = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action = ActionClient(
            self, NavigateToPose, self.get_parameter("navigate_to_pose_action").value
        )
        self._map_subscription = self.create_subscription(
            OccupancyGrid,
            self.get_parameter("map_topic").value,
            self._map_callback,
            _map_subscription_qos(),
        )
        self._start_stop = self.create_service(
            SetBool, self.get_parameter("start_stop_service").value, self._set_search
        )
        self._timer = self.create_timer(1.0, self._search_tick)

    def _map_callback(self, message):
        self._map = message

    def _set_search(self, request, response):
        self._generation += 1
        if request.data:
            self._enabled = True
            self._attempted.clear()
            response.message = "Search enabled"
        else:
            self._enabled = False
            if self._goal_handle is not None and not self._cancel_requested:
                self._cancel_requested = True
                self._goal_handle.cancel_goal_async()
            response.message = "Search disabled"
        response.success = True
        return response

    def _robot_pose(self, map_frame):
        transform = self._tf_buffer.lookup_transform(map_frame, self.get_parameter("base_frame").value, Time())
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        return translation.x, translation.y, yaw

    def _grid(self):
        info = self._map.info
        origin = info.origin
        orientation = origin.orientation
        origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        return {
            "width": info.width,
            "height": info.height,
            "resolution": info.resolution,
            "origin": [origin.position.x, origin.position.y, origin_yaw],
            "data": tuple(self._map.data),
        }

    def _search_tick(self):
        if (not self._enabled or self._map is None or self._accept_pending
                or self._goal_handle is not None or not self._action.server_is_ready()):
            return
        behavior_tree = self.get_parameter("behavior_tree").value
        if not behavior_tree:
            return
        camera = {
            "horizontal_fov_deg": self.get_parameter("camera.horizontal_fov_deg").value,
            "max_range_m": self.get_parameter("camera.max_range_m").value,
        }
        try:
            result = evaluate_search(self._grid(), self._robot_pose(self._map.header.frame_id), camera=camera)
        except Exception as error:
            self.get_logger().debug(f"Search waits for current map/TF: {error}")
            return
        frontier_key = tuple(tuple(cell) for cell in result.get("frontier_cells", []))
        if frontier_key != self._frontier_key:
            self._frontier_key = frontier_key
            self._attempted.clear()
        if result["status"] != "RECOMMENDATION_AVAILABLE":
            return
        candidate = self._next_ranked_candidate(result)
        if candidate is None:
            return
        self._attempted.add(candidate["id"])
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._map.header.frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = candidate["pose_world"][:2]
        goal.pose.pose.position.z = 0.0
        yaw = candidate["pose_world"][2]
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.behavior_tree = behavior_tree
        generation = self._generation
        candidate_id = candidate["id"]
        self._accept_pending = True
        future = self._action.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self._goal_response(completed, generation, candidate_id)
        )

    def _next_ranked_candidate(self, result):
        by_id = {candidate["id"]: candidate for candidate in result["candidates"]}
        return next(
            (by_id[candidate_id] for candidate_id in result["ranking"]
             if candidate_id not in self._attempted),
            None,
        )

    def _goal_response(self, future, generation, candidate_id):
        self._accept_pending = False
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().warning(f"Nav2 goal request failed: {error}")
            return
        if not handle.accepted:
            self.get_logger().info(f"Nav2 rejected Search candidate {candidate_id}")
            return
        self._goal_handle = handle
        self._goal_id = candidate_id
        self._cancel_requested = False
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self._goal_result(completed, candidate_id)
        )
        if not self._enabled or generation != self._generation:
            self._cancel_active_goal()

    def _cancel_active_goal(self):
        if self._goal_handle is not None and not self._cancel_requested:
            self._cancel_requested = True
            self._goal_handle.cancel_goal_async()

    def _goal_result(self, future, candidate_id):
        if self._goal_id != candidate_id:
            return
        try:
            future.result()
        except Exception as error:
            self.get_logger().warning(f"Nav2 result failed for {candidate_id}: {error}")
        self._goal_handle = None
        self._goal_id = None
        self._cancel_requested = False
        # The timer selects the next candidate. The attempted set prevents
        # immediate retries while the frontier cells remain unchanged.


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
