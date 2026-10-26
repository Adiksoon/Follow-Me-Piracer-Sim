"""Offline checks for the single background frontier calculation."""

import copy
import inspect
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import test_search_node as fixtures


class SearchWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SearchNodeCallbackTest.setUpClass()
        cls.module = fixtures.SearchNodeCallbackTest.module

    @classmethod
    def tearDownClass(cls):
        fixtures.SearchNodeCallbackTest.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.SearchNodeCallbackTest()
        self.node = self.fixture.make_node()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.node._evaluation_executor = self.pool
        self.release = threading.Event()
        self.started = threading.Event()
        self.evaluation = self.fixture.result([
            self.fixture.candidate("frontier", score=0.8)
        ])
        self.calls = []

    def tearDown(self):
        self.release.set()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def compute(self, grid, pose, *, camera):
        self.calls.append((copy.deepcopy(grid), pose, dict(camera), threading.get_ident()))
        self.started.set()
        if not self.release.wait(timeout=3.0):
            raise RuntimeError("offline worker was not released")
        return self.evaluation

    def start(self):
        self.node._search_tick()
        self.assertTrue(self.started.wait(timeout=1.0))
        return self.node._evaluation_future

    def finish(self, future):
        self.release.set()
        future.result(timeout=2.0)
        self.node._search_tick()

    def test_pending_job_returns_and_odom_info_runs_before_worker_finishes(self):
        with patch.object(self.module, "evaluate_search", side_effect=self.compute):
            main_thread = threading.get_ident()
            future = self.start()
            self.assertFalse(future.done())
            self.node._odom_info_callback(self.fixture.odom_info(9_900_000_000))
            self.assertEqual(9_900_000_000, self.node._stamp_ns(self.node._odom_info[-1]))
            for _ in range(5):
                self.node._search_tick()
            self.assertEqual(1, len(self.calls))
            self.assertNotEqual(main_thread, self.calls[0][3])
            self.assertEqual([], self.node._action.sent)
            self.finish(future)
        self.assertEqual(1, len(self.node._action.sent))
        self.assertEqual(1, len(self.calls))

    def test_worker_does_not_change_navigation_even_after_completion(self):
        with patch.object(self.module, "evaluate_search", side_effect=self.compute):
            future = self.start()
            self.release.set()
            future.result(timeout=2.0)
            self.assertEqual([], self.node._action.sent)
            self.assertEqual([], self.node._markers.messages)
            self.assertIsNone(self.node._pending_goal)
            self.node._search_tick()
        self.assertEqual(1, len(self.node._action.sent))

    def test_snapshot_does_not_share_map_origin_or_camera(self):
        grid = {"width": 1, "height": 1, "resolution": 1.0,
                "origin": [0.0, 0.0, 0.0], "data": (0,)}
        # _grid creates fresh tuple/list values from the ROS message in production.
        self.node._grid = lambda: copy.deepcopy(grid)
        with patch.object(self.module, "evaluate_search", side_effect=self.compute):
            future = self.start()
            grid["origin"][0] = 5.0
            self.assertEqual([0.0, 0.0, 0.0], self.calls[0][0]["origin"])
            self.assertIsInstance(self.calls[0][1], tuple)
            self.finish(future)
        self.assertEqual([], self.node._action.sent)

    def test_invalidated_context_cannot_dispatch_even_after_recovery(self):
        for change in ("disable", "reenable", "loss_recovery", "clock_reset",
                       "map", "frame", "camera"):
            with self.subTest(change=change):
                node = self.fixture.make_node()
                node._evaluation_executor = self.pool
                self.node = node
                self.calls.clear()
                self.started.clear()
                self.release.clear()
                with patch.object(self.module, "evaluate_search", side_effect=self.compute):
                    future = self.start()
                    if change in ("disable", "reenable"):
                        request = type("Request", (), {"data": False})()
                        response = type("Response", (), {})()
                        node._set_search(request, response)
                        if change == "reenable":
                            request.data = True
                            node._set_search(request, response)
                    elif change == "loss_recovery":
                        node._odom_info_callback(self.fixture.odom_info(9_900_000_000, lost=True))
                        node._odom_info_callback(self.fixture.odom_info(10_000_000_000))
                    elif change == "clock_reset":
                        node._handle_clock_regression(1_000_000_000)
                    elif change == "map":
                        node._grid = lambda: {"changed": True}
                    elif change == "frame":
                        node._map.header.frame_id = "other_map"
                    elif change == "camera":
                        original = node.get_parameter
                        node.get_parameter = lambda name: (
                            type("Param", (), {"value": 4.0})()
                            if name == "camera.max_range_m" else original(name)
                        )
                    self.finish(future)
                    self.assertEqual([], node._action.sent)
                    self.assertEqual(1, len(self.calls))

    def test_pending_job_retained_when_disabled_no_second_job_after_reenable(self):
        with patch.object(self.module, "evaluate_search", side_effect=self.compute):
            future = self.start()
            self.node._set_search(type("R", (), {"data": False})(), type("S", (), {})())
            self.node._search_tick()
            self.assertIs(future, self.node._evaluation_future)
            self.node._set_search(type("R", (), {"data": True})(), type("S", (), {})())
            self.node._search_tick()
            self.assertIs(future, self.node._evaluation_future)
            self.assertEqual(1, len(self.calls))
            self.finish(future)
        self.assertEqual([], self.node._action.sent)

    def test_completed_result_requires_fresh_localization(self):
        with patch.object(self.module, "evaluate_search", side_effect=self.compute):
            future = self.start()
            # Top-of-tick succeeds; mandatory consumption check must still reject.
            poses = iter([(1.0, 2.0, 0.5), None])
            self.node._localization_pose = lambda frame: next(poses)
            self.finish(future)
        self.assertEqual([], self.node._action.sent)

    def test_identical_republished_map_is_compatible(self):
        with patch.object(self.module, "evaluate_search", side_effect=self.compute):
            future = self.start()
            self.node._map = copy.deepcopy(self.node._map)
            self.finish(future)
        self.assertEqual(1, len(self.node._action.sent))

    def test_worker_failure_is_discarded_without_navigation(self):
        def broken(grid, pose, *, camera):
            raise ValueError("bad snapshot")
        with patch.object(self.module, "evaluate_search", side_effect=broken):
            self.node._search_tick()
            future = self.node._evaluation_future
            if future is not None:
                self.assertIsInstance(future.exception(timeout=2.0), ValueError)
                self.node._search_tick()
        self.assertEqual([], self.node._action.sent)

    def test_pure_worker_has_no_node_reference_and_matches_algorithm(self):
        function = self.module._evaluate_frontier_snapshot
        self.assertEqual(["grid", "robot_pose", "camera", "trace", "tick_sequence"],
                         list(inspect.signature(function).parameters))
        grid = {"width": 6, "height": 6, "resolution": 1.0,
                "origin": [0.0, 0.0, 0.0],
                "data": tuple(0 if 1 <= x <= 4 and 1 <= y <= 4 else -1
                              for y in range(6) for x in range(6))}
        pose = (2.5, 2.5, 0.0)
        camera = {"horizontal_fov_deg": 90.0, "max_range_m": 5.0}
        original = copy.deepcopy((grid, pose, camera))
        expected = self.module.evaluate_search(grid, pose, camera=camera)
        actual = self.pool.submit(function, grid, pose, camera, None, 1).result(timeout=2.0)
        self.assertEqual(expected, actual)
        self.assertEqual(original, (grid, pose, camera))


if __name__ == "__main__":
    unittest.main()
