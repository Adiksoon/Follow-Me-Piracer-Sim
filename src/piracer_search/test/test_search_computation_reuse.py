import unittest
from unittest.mock import patch
from piracer_search import search_algorithm

class SearchComputationReuseTest(unittest.TestCase):
    def test_one_bfs_no_raycast_and_limited_detail_checks(self):
        width = height = 30
        data = [-1] * (width * height)
        for y in range(2, 28):
            for x in range(2, 28): data[y * width + x] = 0
        for x in (27, 26, 25): data[10 * width + x] = -1
        grid = {"width": width, "height": height, "resolution": 0.05,
                "origin": [0., 0., 0.], "data": data}
        with (patch.object(search_algorithm, "_reachable_edge_counts", wraps=search_algorithm._reachable_edge_counts) as bfs,
              patch.object(search_algorithm, "_cells_under_footprint", wraps=search_algorithm._cells_under_footprint) as checks):
            result = search_algorithm.evaluate_search(grid, (0.5, 0.5, 0.0))
        self.assertEqual("RECOMMENDATION_AVAILABLE", result["status"])
        self.assertEqual(1, bfs.call_count)
        self.assertLessEqual(len(result["candidates"]), 3)
        self.assertLess(checks.call_count, len(result["cluster_candidates"]) * 100)

if __name__ == "__main__": unittest.main()
