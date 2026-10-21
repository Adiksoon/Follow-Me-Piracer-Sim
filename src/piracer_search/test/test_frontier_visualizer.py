import math
import unittest

from piracer_search.search_algorithm import _footprint_cells, evaluate_frontier_candidates


FOOTPRINT = ((-0.020, -0.1085), (0.246, -0.1085),
             (0.246, 0.1085), (-0.020, 0.1085))
CAMERA = {"horizontal_fov_deg": 65.36016047954865, "max_range_m": 5.0}


def grid(width=40, height=30, origin=(0.0, 0.0, 0.0), fill=-1):
    return {"width": width, "height": height, "resolution": 0.05,
            "origin": list(origin), "data": [fill] * (width * height)}


def set_rect(value, x0, x1, y0, y1):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            value["data"][y * value["width"] + x] = 0


def costmap_like(value, fill=0):
    return {"width": value["width"], "height": value["height"],
            "resolution": value["resolution"], "origin": list(value["origin"]),
            "data": [fill] * (value["width"] * value["height"])}


def pose_for_cell(value, cell):
    x, y = cell
    ox, oy, yaw = value["origin"]
    lx, ly = (x + 0.5) * value["resolution"], (y + 0.5) * value["resolution"]
    return (ox + math.cos(yaw) * lx - math.sin(yaw) * ly,
            oy + math.sin(yaw) * lx + math.cos(yaw) * ly, 0.0)


def evaluate(raw, nav=None, cost=None, start=(10, 10), min_cells=3):
    nav = nav or raw
    cost = cost or costmap_like(nav)
    return evaluate_frontier_candidates(
        raw, nav, cost, pose_for_cell(nav, start), footprint=FOOTPRINT,
        camera=CAMERA, min_group_cells=min_cells,
    )


class FrontierEvaluationTests(unittest.TestCase):
    def test_frontier_goal_uses_free_side_and_checks_footprint_and_costmap(self):
        raw = grid()
        set_rect(raw, 4, 31, 4, 25)
        result = evaluate(raw)
        self.assertEqual("CANDIDATES_AVAILABLE", result["status"])
        self.assertTrue(result["candidates"])
        candidate = result["selected"]
        self.assertTrue(candidate["grid_connected"])
        self.assertTrue(candidate["footprint_clear"])
        self.assertTrue(candidate["costmap_checked"])
        self.assertEqual("NOT_EVALUATED", candidate["ackermann_feasibility"])
        self.assertEqual(0, candidate["max_footprint_cost"])
        self.assertIn("normalized_visible_gain", candidate["score_terms"])
        self.assertIn("selection_reason", result["selected"])
        self.assertEqual(0, raw["data"][candidate["cell"][1] * raw["width"] + candidate["cell"][0]])

    def test_start_zone_only_frontier_is_explicitly_assumption_based(self):
        raw = grid()
        nav = grid()
        set_rect(nav, 4, 31, 4, 25)
        result = evaluate(raw, nav)
        self.assertEqual("CANDIDATES_AVAILABLE", result["status"])
        self.assertTrue(all(group["source_is_assumption_based"] for group in result["groups"]))
        self.assertEqual("ASSUMPTION_BASED", result["selected"]["connectivity_status"])
        self.assertTrue(result["selected"]["path_uses_start_zone_assumption"])

    def test_one_meter_initial_free_zone_produces_labeled_border_candidates(self):
        raw = grid(width=60, height=60)
        nav = grid(width=60, height=60)
        center_x = center_y = 1.5
        for y in range(60):
            for x in range(60):
                wx, wy = pose_for_cell(nav, (x, y))[:2]
                if (wx - center_x) ** 2 + (wy - center_y) ** 2 <= 1.0 ** 2:
                    nav["data"][y * 60 + x] = 0
        start = (30, 30)
        result = evaluate(raw, nav, costmap_like(nav), start=start)
        self.assertEqual("CANDIDATES_AVAILABLE", result["status"])
        self.assertTrue(result["selected"]["path_uses_start_zone_assumption"])
        self.assertTrue(all(group["source_is_assumption_based"] for group in result["groups"]))

    def test_disconnected_free_island_is_rejected_and_explained(self):
        raw = grid()
        set_rect(raw, 3, 11, 3, 11)
        set_rect(raw, 22, 34, 5, 24)
        result = evaluate(raw)
        reasons = [group["reason"] for group in result["groups"]]
        self.assertIn("disconnected_from_robot_free_space", reasons)
        self.assertIsNotNone(result["selected"])

    def test_tiny_frontier_group_is_retained_but_rejected(self):
        raw = grid()
        raw["data"][10 * raw["width"] + 10] = 0
        result = evaluate(raw, start=(10, 10), min_cells=3)
        self.assertEqual("NO_ADMISSIBLE_CANDIDATE", result["status"])
        self.assertEqual(1, len(result["groups"]))
        self.assertEqual("too_small", result["groups"][0]["reason"])

    def test_no_frontier_is_reported(self):
        raw = grid(fill=0)
        result = evaluate(raw)
        self.assertEqual("NO_FRONTIERS", result["status"])
        self.assertEqual([], result["groups"])

    def test_lethal_costmap_barrier_rejects_candidates_across_barrier(self):
        raw = grid()
        set_rect(raw, 4, 34, 4, 25)
        costs = costmap_like(raw)
        for y in range(raw["height"]):
            costs["data"][y * raw["width"] + 20] = 254
        result = evaluate(raw, cost=costs)
        self.assertEqual("CANDIDATES_AVAILABLE", result["status"])
        self.assertTrue(result["candidates"])
        self.assertTrue(all(candidate["cell"][0] < 20 for candidate in result["candidates"]))

    def test_occupied_cell_under_candidate_footprint_rejects_that_pose(self):
        raw = grid()
        set_rect(raw, 4, 31, 4, 25)
        first = evaluate(raw)
        candidate = first["selected"]
        footprint_cells = _footprint_cells(
            candidate["pose_world"][:2], candidate["pose_world"][2], FOOTPRINT,
            raw["width"], raw["height"], raw["resolution"], raw["origin"],
        )
        obstacle = next(cell for cell in footprint_cells if cell != tuple(candidate["cell"]))
        updated = {**raw, "data": list(raw["data"])}
        updated["data"][obstacle[1] * updated["width"] + obstacle[0]] = 100
        result = evaluate(updated)
        pose_key = (candidate["cell"], round(candidate["pose_world"][2], 6))
        self.assertNotIn(pose_key, [
            (item["cell"], round(item["pose_world"][2], 6))
            for item in result["candidates"]
        ])

    def test_rotated_map_origin_preserves_candidate_world_cell_center(self):
        raw = grid(origin=(1.3, -0.7, math.pi / 2))
        set_rect(raw, 4, 31, 4, 25)
        result = evaluate(raw)
        self.assertIsNotNone(result["selected"])
        expected_x, expected_y = pose_for_cell(raw, result["selected"]["cell"])[:2]
        actual_x, actual_y = result["selected"]["pose_world"][:2]
        self.assertAlmostEqual(expected_x, actual_x)
        self.assertAlmostEqual(expected_y, actual_y)

    def test_new_map_is_recomputed_without_retaining_old_frontiers(self):
        first = grid()
        set_rect(first, 8, 16, 8, 16)
        second = grid()
        set_rect(second, 4, 31, 4, 25)
        first_result = evaluate(first)
        second_result = evaluate(second)
        self.assertNotEqual(first_result["frontier_cells"], second_result["frontier_cells"])
        self.assertEqual(1, len(second_result["groups"]))

    def test_map_geometry_mismatch_waits_instead_of_comparing_unpaired_maps(self):
        raw = grid()
        nav = grid(origin=(0.1, 0.0, 0.0))
        result = evaluate(raw, nav)
        self.assertEqual("MAP_GEOMETRY_MISMATCH", result["status"])

    def test_missing_global_costmap_does_not_claim_candidates_are_safe(self):
        raw = grid()
        set_rect(raw, 4, 31, 4, 25)
        result = evaluate_frontier_candidates(
            raw, raw, None, pose_for_cell(raw, (10, 10)),
            footprint=FOOTPRINT, camera=CAMERA,
        )
        self.assertEqual("WAITING_FOR_NAVIGATION_COSTMAP", result["status"])
        self.assertIsNone(result["selected"])


if __name__ == "__main__":
    unittest.main()
