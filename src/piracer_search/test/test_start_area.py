import math
import unittest

from piracer_search.start_area import merge_initial_start_area


class StartAreaMergeTest(unittest.TestCase):
    def test_only_whole_unknown_cells_change_and_positive_cells_remain(self):
        grid = {
            "width": 3,
            "height": 1,
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
            "data": [-1, 80, -1],
        }
        data, summary = merge_initial_start_area(
            grid, (0.0, 0.0, 0.0), bounds=(0.0, 0.2, 0.0, 0.1)
        )
        self.assertEqual((0, 80, -1), data)
        self.assertEqual([-1, 80, -1], grid["data"])
        self.assertEqual(2, summary["whole_cells_inside"])
        self.assertEqual(1, summary["unknown_changed_to_free"])
        self.assertEqual(1, summary["positive_cells_preserved_inside"])

    def test_partial_cells_outside_rectangle_are_not_seeded(self):
        grid = {
            "width": 2,
            "height": 1,
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
            "data": [-1, -1],
        }
        data, summary = merge_initial_start_area(
            grid, (0.0, 0.0, 0.0), bounds=(0.01, 0.19, 0.0, 0.1)
        )
        self.assertEqual((-1, -1), data)
        self.assertEqual(0, summary["whole_cells_inside"])

    def test_rotated_map_origin_and_initial_pose(self):
        grid = {
            "width": 3,
            "height": 1,
            "resolution": 0.1,
            "origin": [0.0, 0.0, math.pi / 2],
            "data": [-1, 75, -1],
        }
        data, summary = merge_initial_start_area(
            grid,
            (0.0, 0.0, math.pi / 2),
            bounds=(-0.01, 0.21, -0.01, 0.11),
        )
        self.assertEqual((0, 75, -1), data)
        self.assertEqual(2, summary["whole_cells_inside"])
        self.assertEqual(1, summary["positive_cells_preserved_inside"])


if __name__ == "__main__":
    unittest.main()
