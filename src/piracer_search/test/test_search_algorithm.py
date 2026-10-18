import json
import math
from pathlib import Path
import unittest

from piracer_search.search_algorithm import evaluate_search, world_to_cell


ROOT = Path(__file__).resolve().parents[3]


class SearchAlgorithmTest(unittest.TestCase):
    def test_goal02_visible_gain_cost_matches_independent_expected_geometry(self):
        fixtures = ROOT / "experiments/search_offline_goal02/fixtures"
        source = json.loads((fixtures / "search_goal02_v5_cases.json").read_text())
        expected = json.loads((fixtures / "search_goal02_v5_expected.json").read_text())
        case = next(item for item in source["search_cases"] if item["case_id"] == "S01_TWO_FRONTIERS")
        map_data = source["maps"][case["map_ref"]]
        symbols = {".": 0, "#": 100, "?": -1}
        occupancy = {
            "width": map_data["width"],
            "height": map_data["height"],
            "resolution": map_data["resolution"],
            "origin": map_data["origin"],
            "data": [symbols[symbol] for row in map_data["rows"] for symbol in row],
        }
        # World coordinates are calculated from the fixture robot cell center.
        rx, ry = case["robot"]["cell"]
        resolution = occupancy["resolution"]
        robot = ((rx + 0.5) * resolution, (ry + 0.5) * resolution, case["robot"]["yaw_world"])
        result = evaluate_search(occupancy, robot, camera=case["camera"])

        # These are hand-authored Goal02 expected values. Costs are independently
        # checked from the fixture's explicit cardinal path cell counts.
        self.assertIs(expected["expected_results_hand_authored"], True)
        oracle = next(item for item in expected["search_cases"] if item["case_id"] == case["case_id"])
        self.assertEqual([[6, 4], [4, 6]], oracle["frontier_cells"])
        self.assertEqual([[5, 4], [4, 5]], [item["cell"] for item in oracle["candidates"]])
        self.assertEqual(oracle["frontier_cells"], result["frontier_cells"])
        self.assertEqual([item["cell"] for item in oracle["candidates"]], [item["cell"] for item in result["candidates"]])
        for expected_candidate in oracle["candidates"]:
            candidate_id = expected_candidate["id"]
            path_edges = len(case["paths_by_candidate"][candidate_id]) - 1
            self.assertEqual(path_edges * resolution, expected_candidate["cost_m"])
            actual = next(item for item in result["candidates"] if item["id"] == candidate_id)
            self.assertEqual(expected_candidate["cost_m"], actual["cost_m"])
            self.assertEqual(expected_candidate["unknown_opaque_cells"], actual["visible_unknown_cells"])
        score_oracle = oracle["strategies"]["GAIN_COST_VISIBLE_OPAQUE_W050"]
        expected_ranking = score_oracle["ranking"]
        self.assertEqual(expected_ranking, result["ranking"])
        expected_scores = score_oracle["scores"]
        for candidate in result["candidates"]:
            self.assertAlmostEqual(
                expected_scores[candidate["id"]], candidate["score"], delta=1e-6
            )

    def test_rotated_origin_world_to_cell(self):
        grid = {"origin": [4.0, -2.0, math.pi / 2], "resolution": 0.5}
        self.assertEqual((2, 3), world_to_cell(grid, 2.25, -0.75))

    def test_no_safe_candidate(self):
        width = height = 5
        data = [100] * (width * height)
        for x, y in {(2, 2), (3, 2)}:
            data[y * width + x] = 0
        data[2 * width + 4] = -1
        grid = {"width": width, "height": height, "resolution": 1.0, "origin": [0, 0, 0], "data": data}
        result = evaluate_search(grid, (2.5, 2.5, 0.0), camera={"horizontal_fov_deg": 90, "max_range_m": 1})
        self.assertEqual("NO_ADMISSIBLE_CANDIDATE", result["status"])
        self.assertEqual([], result["ranking"])

    def test_intermediate_occupancy_blocks_safe_footprint(self):
        width = height = 8
        data = [100] * (width * height)
        for y in range(1, 7):
            for x in range(1, 7):
                data[y * width + x] = 0
        data[3 * width + 6] = -1
        data[2 * width + 3] = 50
        grid = {"width": width, "height": height, "resolution": 1.0, "origin": [0, 0, 0], "data": data}
        result = evaluate_search(grid, (2.5, 3.5, 0.0), camera={"horizontal_fov_deg": 90, "max_range_m": 2})
        self.assertIn([3, 3], result["frontier_cells"])
        self.assertNotIn("x4_y3_d0", {item["id"] for item in result["candidates"]})

    def test_missing_map_localization_or_camera_configuration(self):
        self.assertEqual("WAITING_FOR_MAP", evaluate_search(None, (0, 0, 0))["status"])
        grid = {"width": 3, "height": 3, "resolution": 1.0, "origin": [0, 0, 0], "data": [100] * 9}
        self.assertEqual("WAITING_FOR_LOCALIZATION", evaluate_search(grid, None)["status"])
        self.assertEqual("WAITING_FOR_CAMERA_CONFIGURATION", evaluate_search(grid, (1.5, 1.5, 0))["status"])
