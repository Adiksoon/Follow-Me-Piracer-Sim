import unittest
from unittest.mock import patch

from piracer_search import search_algorithm


class SearchComputationReuseTest(unittest.TestCase):
    def test_reuses_one_bfs_and_one_unknown_target_list_per_evaluation(self):
        width = height = 6
        data = [-1] * (width * height)
        for y in range(1, 5):
            for x in range(1, 5):
                data[y * width + x] = 0
        grid = {
            "width": width,
            "height": height,
            "resolution": 1.0,
            "origin": [0.0, 0.0, 0.0],
            "data": data,
        }
        camera = {"horizontal_fov_deg": 90.0, "max_range_m": 5.0}
        visible_unknown = search_algorithm._visible_unknown
        target_list_ids = []

        def capture_target_list(*args):
            target_list_ids.append(id(args[-1]))
            return visible_unknown(*args)

        with (
            patch.object(
                search_algorithm,
                "_safe_free_cells",
                wraps=search_algorithm._safe_free_cells,
            ) as safe_cells,
            patch.object(
                search_algorithm,
                "_reachable_edge_counts",
                wraps=search_algorithm._reachable_edge_counts,
            ) as bfs,
            patch.object(
                search_algorithm,
                "_unknown_boundary_cells",
                wraps=search_algorithm._unknown_boundary_cells,
            ) as targets,
            patch.object(
                search_algorithm,
                "_visible_unknown",
                side_effect=capture_target_list,
            ),
        ):
            result = search_algorithm.evaluate_search(
                grid, (2.5, 2.5, 0.0), camera=camera
            )

        self.assertEqual("RECOMMENDATION_AVAILABLE", result["status"])
        self.assertGreater(len(result["candidates"]), 0)
        self.assertEqual(1, safe_cells.call_count)
        self.assertEqual(1, bfs.call_count)
        self.assertEqual(1, targets.call_count)
        self.assertGreater(len(target_list_ids), 1)
        self.assertEqual(1, len(set(target_list_ids)))


if __name__ == "__main__":
    unittest.main()
