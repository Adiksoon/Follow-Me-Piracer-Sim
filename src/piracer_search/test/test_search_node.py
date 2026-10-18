import importlib
import sys
import types
import unittest
from unittest.mock import patch


class Future:
    def __init__(self, value=None):
        self.value = value
        self.callback = None

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callback = callback


class FailingFuture(Future):
    def result(self):
        raise RuntimeError("terminal failure")


class FakeHandle:
    accepted = True

    def __init__(self):
        self.cancel_count = 0
        self.result_future = Future(object())

    def cancel_goal_async(self):
        self.cancel_count += 1
        return Future()

    def get_result_async(self):
        return self.result_future


class SearchNodeCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy = types.ModuleType("rclpy")
        rclpy.__path__ = []
        rclpy.init = lambda **kwargs: None
        rclpy.spin = lambda node: None
        rclpy.shutdown = lambda: None
        node_module = types.ModuleType("rclpy.node")
        node_module.Node = type("Node", (), {})
        class FakeQoSProfile:
            def __init__(self, *, depth, reliability, durability):
                self.depth = depth
                self.reliability = reliability
                self.durability = durability

        qos_module = types.ModuleType("rclpy.qos")
        qos_module.QoSProfile = FakeQoSProfile
        qos_module.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="reliable")
        qos_module.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient_local")
        action_module = types.ModuleType("rclpy.action")
        action_module.ActionClient = type("ActionClient", (), {})
        time_module = types.ModuleType("rclpy.time")
        time_module.Time = type("Time", (), {})
        geometry_module = types.ModuleType("geometry_msgs.msg")
        class FakePoseStamped:
            def __init__(self):
                self.header = types.SimpleNamespace(frame_id=None, stamp=None)
                self.pose = types.SimpleNamespace(
                    position=types.SimpleNamespace(x=None, y=None, z=None),
                    orientation=types.SimpleNamespace(z=None, w=None))

        geometry_module.PoseStamped = FakePoseStamped
        geometry_parent = types.ModuleType("geometry_msgs")
        geometry_parent.__path__ = []
        nav2_action = type("NavigateToPose", (), {"Goal": type("Goal", (), {})})
        nav2_module = types.ModuleType("nav2_msgs.action")
        nav2_module.NavigateToPose = nav2_action
        nav2_parent = types.ModuleType("nav2_msgs")
        nav2_parent.__path__ = []
        nav_module = types.ModuleType("nav_msgs.msg")
        nav_module.OccupancyGrid = type("OccupancyGrid", (), {})
        nav_parent = types.ModuleType("nav_msgs")
        nav_parent.__path__ = []
        service_module = types.ModuleType("std_srvs.srv")
        service_module.SetBool = type("SetBool", (), {})
        service_parent = types.ModuleType("std_srvs")
        service_parent.__path__ = []
        tf_module = types.ModuleType("tf2_ros")
        tf_module.Buffer = type("Buffer", (), {})
        tf_module.TransformListener = type("TransformListener", (), {})
        cls.modules = {
            "rclpy": rclpy,
            "rclpy.node": node_module,
            "rclpy.qos": qos_module,
            "rclpy.action": action_module,
            "rclpy.time": time_module,
            "geometry_msgs": geometry_parent,
            "geometry_msgs.msg": geometry_module,
            "nav2_msgs": nav2_parent,
            "nav2_msgs.action": nav2_module,
            "nav_msgs": nav_parent,
            "nav_msgs.msg": nav_module,
            "std_srvs": service_parent,
            "std_srvs.srv": service_module,
            "tf2_ros": tf_module,
        }
        with patch.dict(sys.modules, cls.modules):
            cls.module = importlib.import_module("piracer_search.search_node")

    def make_node(self):
        node = object.__new__(self.module.SearchNode)
        node._enabled = True
        node._generation = 0
        node._frontier_key = None
        node._accept_pending = True
        node._attempted = set()
        node._goal_handle = None
        node._goal_id = None
        node._cancel_requested = False
        node._logger = type("Logger", (), {"warning": lambda *args: None, "info": lambda *args: None})()
        node.get_logger = lambda: node._logger
        return node

    def test_terminal_failure_releases_owned_goal(self):
        node = self.make_node()
        handle = FakeHandle()
        node._goal_handle = handle
        node._goal_id = "x4_y5_d1"
        node._goal_result(FailingFuture(), "x4_y5_d1")
        self.assertIsNone(node._goal_handle)
        self.assertIsNone(node._goal_id)

    def test_map_subscription_uses_reliable_transient_local_depth_one_qos(self):
        qos = self.module._map_subscription_qos()
        self.assertEqual(1, qos.depth)
        self.assertEqual("reliable", qos.reliability)
        self.assertEqual("transient_local", qos.durability)

    def test_search_dispatches_configured_tree_with_map_pose_and_sim_stamp(self):
        node = self.make_node()
        node._accept_pending = False
        node._map = types.SimpleNamespace(header=types.SimpleNamespace(frame_id="map"))
        node.get_parameter = lambda name: types.SimpleNamespace(value={
            "behavior_tree": "/share/search_navigate_to_pose.xml",
            "camera.horizontal_fov_deg": 65.0,
            "camera.max_range_m": 5.0,
            "base_frame": "base_link",
        }[name])
        sent = []
        node._action = types.SimpleNamespace(server_is_ready=lambda: True,
                                             send_goal_async=lambda goal: (sent.append(goal) or Future()))
        node._grid = lambda: {"width": 1, "height": 1, "resolution": 1.0,
                              "origin": [0.0, 0.0, 0.0], "data": (0,)}
        node._robot_pose = lambda frame: (1.0, 2.0, 0.5)
        node.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(to_msg=lambda: "sim-stamp"))
        with patch.object(self.module, "evaluate_search", return_value={
                "status": "RECOMMENDATION_AVAILABLE", "frontier_cells": [],
                "ranking": ["x0_y0_d0"], "candidates": [{
                    "id": "x0_y0_d0", "pose_world": [1.0, 2.0, 0.5]}]}):
            node._search_tick()
        goal = sent[0]
        self.assertEqual("/share/search_navigate_to_pose.xml", goal.behavior_tree)
        self.assertEqual("map", goal.pose.header.frame_id)
        self.assertEqual("sim-stamp", goal.pose.header.stamp)
        self.assertEqual(1.0, goal.pose.pose.position.x)
        self.assertEqual(2.0, goal.pose.pose.position.y)
        self.assertTrue(node._accept_pending)

    def test_next_candidate_follows_score_ranking_not_coordinate_order(self):
        node = self.make_node()
        result = {
            "ranking": ["score-first", "coordinate-first"],
            "candidates": [
                {"id": "coordinate-first", "cell": [1, 1]},
                {"id": "score-first", "cell": [5, 4]},
            ],
        }
        self.assertEqual("score-first", node._next_ranked_candidate(result)["id"])

    def test_stop_cancels_owned_goal(self):
        node = self.make_node()
        handle = FakeHandle()
        node._goal_handle = handle
        response = types.SimpleNamespace(success=False, message="")
        returned = node._set_search(types.SimpleNamespace(data=False), response)
        self.assertIs(returned, response)
        self.assertFalse(node._enabled)
        self.assertEqual(1, handle.cancel_count)

    def test_stale_accepted_goal_is_cancelled_and_does_not_enable_search(self):
        node = self.make_node()
        node._enabled = False
        node._generation = 1
        handle = FakeHandle()
        node._goal_response(Future(handle), generation=0, candidate_id="x4_y5_d1")
        self.assertEqual(1, handle.cancel_count)
        self.assertFalse(node._enabled)
        self.assertIs(node._goal_handle, handle)
        node._goal_result(handle.result_future, candidate_id="x4_y5_d1")
        self.assertIsNone(node._goal_handle)
        self.assertFalse(node._enabled)
