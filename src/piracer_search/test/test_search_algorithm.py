import math
import unittest

from piracer_search.search_algorithm import evaluate_search, world_to_cell


def grid(width, height, free_cells, unknown_cells=(), origin=(0.0, 0.0, 0.0), resolution=1.0):
    data = [100] * (width * height)
    for x, y in free_cells:
        data[y * width + x] = 0
    for x, y in unknown_cells:
        data[y * width + x] = -1
    return {"width": width, "height": height, "resolution": resolution, "origin": list(origin), "data": data}


class SearchAlgorithmTest(unittest.TestCase):
    def test_cp1_compatible_frontier_candidate_and_cost_order(self):
        # Candidate IDs and edge counts match the frozen CP1 S01 fixture.
        free = {(x, y) for x in range(2, 8) for y in range(3, 7)}
        unknown = {(7, 4), (4, 7)}
        result = evaluate_search(grid(10, 10, free, unknown, resolution=0.25), (0.875, 1.375, 0.0))
        by_id = {item["id"]: item for item in result["candidates"]}
        self.assertEqual(result["status"], "RECOMMENDATION_AVAILABLE")
        self.assertEqual(["x4_y5_d1", "x5_y4_d0"], result["ranking"][:2])
        self.assertEqual(0.25, by_id["x4_y5_d1"]["cost_m"])
        self.assertEqual(0.75, by_id["x5_y4_d0"]["cost_m"])

    def test_world_to_cell_with_rotated_origin(self):
        g = {"origin": [4.0, -2.0, math.pi / 2], "resolution": 0.5}
        self.assertEqual((2, 3), world_to_cell(g, 2.25, -0.75))

    def test_no_safe_candidate(self):
        g = grid(5, 5, {(2, 2), (3, 2)}, {(4, 2)})
        result = evaluate_search(g, (2.5, 2.5, 0.0))
        self.assertEqual("NO_ADMISSIBLE_CANDIDATE", result["status"])
        self.assertEqual([], result["ranking"])

    def test_missing_map_or_localization(self):
        self.assertEqual("WAITING_FOR_MAP", evaluate_search(None, (0, 0, 0))["status"])
        g = grid(3, 3, {(1, 1)})
        self.assertEqual("WAITING_FOR_LOCALIZATION", evaluate_search(g, None)["status"])
