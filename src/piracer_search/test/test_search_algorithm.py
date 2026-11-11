import ast
import json
import math
from pathlib import Path
import re
import unittest

from piracer_search.search_algorithm import (
    MAX_VIEWPOINT_SHIFT_M, PIRACER_FOOTPRINT, _cells_under_footprint,
    _read_grid, _frontier_clusters, evaluate_search, footprint_is_free,
    match_active_frontiers, revalidate_candidate, world_to_cell,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class SearchAlgorithmTest(unittest.TestCase):
    def test_active_frontier_matches_all_overlapping_clusters_in_shifted_grid(self):
        grid = {"width": 20, "height": 20, "resolution": 0.5,
                "origin": [10.0, -3.0, 0.0], "data": [0] * 400}
        old = {"source_frontiers_world": [[11.25, 0.25], [11.75, 0.25]]}
        current = {"cluster_id": "active", "score": 0.2,
                   "frontiers_world": [[11.3, 0.3]]}
        result = {"cluster_candidates": [
            {"cluster_id": "other", "score": 0.9, "frontiers_world": [[14.0, 1.0]]},
            current,
        ]}
        split = {"cluster_id": "split", "score": 0.8,
                 "frontiers_world": [[11.4, 0.2]]}
        self.assertEqual([current], match_active_frontiers(grid, old, result))
        result["cluster_candidates"].append(split)
        matches = match_active_frontiers(grid, old, result)
        self.assertEqual([current, split], matches)
        self.assertEqual(0.8, max(cluster["score"] for cluster in matches))

    def test_active_frontier_fails_closed_for_invalid_overlapping_score(self):
        grid = {"width": 20, "height": 20, "resolution": 0.5,
                "origin": [10.0, -3.0, 0.0], "data": [0] * 400}
        old = {"source_frontiers_world": [[11.25, 0.25]]}
        result = {"cluster_candidates": [
            {"cluster_id": "valid", "score": 0.2,
             "frontiers_world": [[11.3, 0.3]]},
            {"cluster_id": "invalid", "score": float("nan"),
             "frontiers_world": [[11.4, 0.2]]},
        ]}
        self.assertIsNone(match_active_frontiers(grid, old, result))

    def test_active_frontier_with_no_overlap_returns_none(self):
        grid = {"width": 20, "height": 20, "resolution": 0.5,
                "origin": [10.0, -3.0, 0.0], "data": [0] * 400}
        old = {"source_frontiers_world": [[11.25, 0.25]]}
        result = {"cluster_candidates": [
            {"cluster_id": "other", "score": 0.9,
             "frontiers_world": [[14.0, 1.0]]},
        ]}
        self.assertIsNone(match_active_frontiers(grid, old, result))

    def test_goal02_frontier_cost_with_independent_expected_geometry(self):
        fixtures = FIXTURES
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
        result = evaluate_search(occupancy, robot)

        # Preserve the hand-authored Goal02 frontier geometry.
        self.assertIs(expected["expected_results_hand_authored"], True)
        oracle = next(item for item in expected["search_cases"] if item["case_id"] == case["case_id"])
        self.assertEqual([[6, 4], [4, 6]], oracle["frontier_cells"])
        self.assertEqual([[5, 4], [4, 5]], [item["cell"] for item in oracle["candidates"]])
        self.assertEqual(oracle["frontier_cells"], result["frontier_cells"])
        self.assertTrue(result["clustered"])
        self.assertLessEqual(len(result["candidates"]), 3)
        clusters = result["cluster_candidates"]
        self.assertEqual([1, 1], [c["cluster_size"] for c in clusters])
        self.assertEqual([0.0, 0.5], [c["cost_m"] for c in clusters])
        self.assertAlmostEqual(-0.25 * (3.0 * math.sqrt(0.125) - 1.0), clusters[0]["score"])
        self.assertAlmostEqual(-0.25 * (3.0 * math.sqrt(0.625) - 1.0), clusters[1]["score"])
        self.assertTrue(all("cluster_access_distance_m" in c for c in result["candidates"]))
        self.assertEqual([c["id"] for c in result["candidates"]], result["ranking"])

    def test_larger_cluster_can_win_at_greater_euclidean_distance_in_rotated_map(self):
        width, height = 20, 18
        data = [0] * (width * height)
        for x, y in ((4, 8), (13, 8), (13, 9)):
            data[y * width + x] = -1
        grid = {"width": width, "height": height, "resolution": 0.25,
                "origin": [4.0, -2.0, math.pi / 2], "data": data}
        # Local robot XY is (2.0, 2.125), between cell centres along X.
        result = evaluate_search(grid, (1.875, 0.0, 0.0))
        large, small = result["cluster_candidates"]
        self.assertEqual([6, 4], [large["cluster_size"], small["cluster_size"]])
        self.assertAlmostEqual(-0.25 * (3.0 * 1.125 - 6.0), large["score"])
        self.assertAlmostEqual(-0.25 * (3.0 * 0.625 - 4.0), small["score"])
        self.assertGreater(large["distance_m"], small["distance_m"])
        self.assertEqual([large["cluster_id"], small["cluster_id"]],
                         [c["cluster_id"] for c in result["candidates"]])
        self.assertEqual(result["candidates"][0]["id"], result["recommended"])

    def test_equal_frontier_costs_use_bfs_access_then_anchor(self):
        width, height = 20, 18
        data = [0] * (width * height)
        data[8 * width + 3] = data[8 * width + 15] = -1
        grid = {"width": width, "height": height, "resolution": 0.25,
                "origin": [0.0, 0.0, 0.0], "data": data}
        robot = (2.375, 2.125, 0.0)
        first, second = evaluate_search(grid, robot)["cluster_candidates"]
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["distance_m"], second["distance_m"])
        self.assertEqual([[0.875, 1.875], [3.875, 1.875]],
                         [first["anchor_world"], second["anchor_world"]])

        data[8 * width + 7] = 100  # Force a detour to the left cluster.
        first, second = evaluate_search(grid, robot)["cluster_candidates"]
        self.assertEqual(first["score"], second["score"])
        self.assertLess(first["distance_m"], second["distance_m"])
        self.assertEqual([[3.875, 1.875], [0.875, 1.875]],
                         [first["anchor_world"], second["anchor_world"]])

    def test_reachable_viewpoints_remain_distinct_when_clusters_share_best_point(self):
        width, height = 20, 18
        data = [0] * (width * height)
        data[8 * width + 3] = data[8 * width + 7] = -1
        grid = {"width": width, "height": height, "resolution": 0.25,
                "origin": [0.0, 0.0, 0.0], "data": data}
        result = evaluate_search(grid, (1.375, 2.125, 0.0))
        self.assertEqual([[5, 8], [5, 8]], [c["cell"] for c in result["cluster_candidates"]])
        self.assertEqual([[5, 8], [5, 7]], [c["cell"] for c in result["candidates"]])

    def test_bfs_excludes_nearer_larger_cluster_behind_wall(self):
        width, height = 24, 20
        data = [0] * (width * height)
        for x, y in ((3, 13), (10, 8), (10, 9)):
            data[y * width + x] = -1
        grid = {"width": width, "height": height, "resolution": 0.25,
                "origin": [0.0, 0.0, 0.0], "data": data}
        robot = (1.625, 2.125, 0.0)
        open_result = evaluate_search(grid, robot)
        nearer, farther = open_result["cluster_candidates"]
        self.assertEqual([2.625, 1.875], nearer["anchor_world"])
        for y in range(height):
            data[y * width + 8] = 100
        result = evaluate_search(grid, robot)
        self.assertIn([9, 8], result["frontier_cells"])
        self.assertEqual([farther["cluster_id"]], [c["cluster_id"] for c in result["cluster_candidates"]])
        self.assertEqual([farther["cluster_id"]], [c["cluster_id"] for c in result["candidates"]])

    def test_cluster_scoring_and_world_metadata_revalidation(self):
        w = h = 24; res = 0.05
        data = [100] * (w*h)
        for y in range(3, 21):
            for x in range(3, 21): data[y*w+x] = 0
        for x in (20, 19, 18): data[10*w+x] = -1
        grid = {"width":w,"height":h,"resolution":res,"origin":[0,0,0],"data":data}
        result = evaluate_search(grid, (0.5,0.5,0))
        self.assertTrue(result["clustered"])
        self.assertTrue(result["candidates"])
        c = result["candidates"][0]
        self.assertGreaterEqual(c["cluster_size"], 1)
        self.assertEqual(c["score"], result["cluster_candidates"][0]["score"])
        fresh = __import__("piracer_search.search_algorithm", fromlist=["revalidate_candidate"]).revalidate_candidate(grid, (0.5,0.5,0), c)
        self.assertIsNotNone(fresh)
        self.assertEqual(c["score"], fresh["score"])
        blocked = dict(grid); changed = list(data)
        fx, fy = c["source_frontier_cell"]; changed[fy*w+fx] = 100; blocked["data"] = changed
        self.assertIsNone(__import__("piracer_search.search_algorithm", fromlist=["revalidate_candidate"]).revalidate_candidate(blocked, (0.5,0.5,0), c))

    def test_eight_neighbor_components_and_zero_bfs_access_distance(self):
        self.assertEqual([[(0, 0), (1, 1)], [(5, 4)]], _frontier_clusters([(5, 4), (1, 1), (0, 0)]))
        # The helper verifies diagonal frontier cells form one 8-neighbor component.
        self.assertEqual([[(0, 0), (1, 1)]], _frontier_clusters([(0, 0), (1, 1)]))
        w = h = 28; res = 0.05; data = [100] * (w*h)
        for y in range(2, 26):
            for x in range(2, 26): data[y*w+x] = 0
        data[12*w+20] = -1
        grid = {"width":w,"height":h,"resolution":res,"origin":[0,0,0],"data":data}
        result = evaluate_search(grid, ((18.5)*res, (12.5)*res, 0))
        self.assertEqual(1, len(result["cluster_candidates"]))
        self.assertEqual(0.0, result["cluster_candidates"][0]["cost_m"])
        self.assertAlmostEqual(0.1925, result["cluster_candidates"][0]["score"])

    def test_continue_after_three_unsafe_cluster_points(self):
        from unittest.mock import patch
        from piracer_search.search_algorithm import _candidate_for as real_candidate_for
        w = h = 36; data = [100] * (w*h)
        for y in range(2, 34):
            for x in range(2, 34): data[y*w+x] = 0
        for x in (32, 27, 22, 17): data[10*w+x] = -1
        grid = {"width":w,"height":h,"resolution":0.05,"origin":[0,0,0],"data":data}
        ranked = evaluate_search(grid, (0.5, 0.5, 0))["cluster_candidates"]
        failed_clusters = {c["cluster_id"] for c in ranked[:3]}
        passed_clusters = []
        def reject_first_three(record, *args):
            cluster_id = record["cluster_id"]
            if cluster_id in failed_clusters:
                return None
            passed_clusters.append(cluster_id)
            return real_candidate_for(record, *args)
        with patch("piracer_search.search_algorithm._candidate_for", side_effect=reject_first_three):
            result = evaluate_search(grid, (0.5, 0.5, 0))
        self.assertEqual(3, len(failed_clusters))
        self.assertTrue(result["candidates"])
        self.assertNotIn(result["candidates"][0]["cluster_id"], failed_clusters)

    def test_revalidation_rejects_wall_and_accepts_shifted_origin_by_world_metadata(self):
        w = h = 24; res = 0.05; data = [100]*(w*h)
        for y in range(3,21):
            for x in range(3,21): data[y*w+x] = 0
        data[10*w+20] = -1
        grid = {"width":w,"height":h,"resolution":res,"origin":[0,0,0],"data":data}
        rec = evaluate_search(grid, (0.5,0.5,0))["cluster_candidates"][0]
        self.assertIsNotNone(revalidate_candidate(grid, (0.5,0.5,0), rec))
        wall = dict(grid); changed = list(data); changed[8*w+8] = 100; wall["data"] = changed
        # An unrelated wall need not invalidate an alternate still-safe point.
        self.assertIsNotNone(revalidate_candidate(wall, (0.5,0.5,0), rec))
        consumed = dict(grid); changed = list(data)
        fx, fy = world_to_cell({"origin":grid["origin"],"resolution":res}, *rec["source_frontier_world"])
        for world in rec["frontiers_world"]:
            fx, fy = world_to_cell({"origin":grid["origin"],"resolution":res}, *world)
            changed[fy*w+fx] = 100
        consumed["data"] = changed
        self.assertIsNone(revalidate_candidate(consumed, (0.5,0.5,0), rec))
        # Changing map origin while retaining world coordinates must not trust old indices.
        shifted = dict(grid); shifted["origin"] = [0.1, 0, 0]
        self.assertIsNone(revalidate_candidate(shifted, (0.5,0.5,0), rec))

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
        result = evaluate_search(grid, (2.5, 2.5, 0.0))
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
        result = evaluate_search(grid, (2.5, 3.5, 0.0))
        self.assertIn([3, 3], result["frontier_cells"])
        self.assertNotIn("x4_y3_d0", {item["id"] for item in result["candidates"]})

    def test_missing_map_or_localization(self):
        self.assertEqual("WAITING_FOR_MAP", evaluate_search(None, (0, 0, 0))["status"])
        grid = {"width": 3, "height": 3, "resolution": 1.0, "origin": [0, 0, 0], "data": [100] * 9}
        self.assertEqual("WAITING_FOR_LOCALIZATION", evaluate_search(grid, None)["status"])
        self.assertEqual("NO_FRONTIERS", evaluate_search(grid, (1.5, 1.5, 0))["status"])

    def test_search_footprint_matches_nav2_global_and_local_costmaps(self):
        config = (ROOT / "src/piracer_navigation/config/nav2_params.yaml").read_text()
        configured = [ast.literal_eval(value) for value in re.findall(
            r'^\s+footprint:\s+"(\[\[.*?\]\])"\s*$', config, re.MULTILINE
        )]
        self.assertEqual(2, len(configured))
        expected = [list(point) for point in PIRACER_FOOTPRINT]
        self.assertEqual([expected, expected], configured)

    def test_complete_known_free_footprint_is_accepted_and_any_unknown_is_rejected(self):
        grid = {
            "width": 40, "height": 40, "resolution": 0.05,
            "origin": [-1.0, -1.0, 0.0], "data": [0] * 1600,
        }
        pose = (0.0, 0.0, 0.0)
        self.assertTrue(footprint_is_free(grid, pose))

        width, height, resolution, origin, _ = _read_grid(grid)
        touched = _cells_under_footprint(
            width, height, resolution, origin, pose[:2], pose[2], PIRACER_FOOTPRINT
        )
        self.assertGreater(len(touched), 4)
        changed = dict(grid)
        data = list(grid["data"])
        x, y = touched[len(touched) // 2]
        data[y * width + x] = -1
        changed["data"] = data
        self.assertFalse(footprint_is_free(changed, pose))

    def test_heading_changes_the_cells_checked_under_asymmetric_footprint(self):
        width = height = 40
        resolution, origin, pose = 0.05, [-1.0, -1.0, 0.0], (0.0, 0.0)
        cells0 = set(_cells_under_footprint(
            width, height, resolution, origin, pose, 0.0, PIRACER_FOOTPRINT
        ))
        cells90 = set(_cells_under_footprint(
            width, height, resolution, origin, pose, math.pi / 2, PIRACER_FOOTPRINT
        ))
        only_at_zero = sorted(cells0 - cells90)
        self.assertTrue(only_at_zero)

        grid = {"width": width, "height": height, "resolution": resolution,
                "origin": origin, "data": [0] * (width * height)}
        data = list(grid["data"])
        x, y = only_at_zero[0]
        data[y * width + x] = -1
        grid["data"] = data
        self.assertFalse(footprint_is_free(grid, (*pose, 0.0)))
        self.assertTrue(footprint_is_free(grid, (*pose, math.pi / 2)))

    def test_footprint_outside_map_is_rejected(self):
        grid = {
            "width": 40, "height": 40, "resolution": 0.05,
            "origin": [0.0, 0.0, 0.0], "data": [0] * 1600,
        }
        self.assertFalse(footprint_is_free(grid, (0.01, 1.0, 0.0)))

    def test_unknown_cell_touching_footprint_edge_or_corner_is_rejected(self):
        width = height = 40
        origin, resolution = [-1.0, -1.0, 0.0], 0.05
        for pose, unknown_cell in (
            ((0.004, 0.0, 0.0), (25, 20)),
            ((0.004, 0.0415, 0.0), (25, 23)),
        ):
            with self.subTest(unknown_cell=unknown_cell):
                data = [0] * (width * height)
                data[unknown_cell[1] * width + unknown_cell[0]] = -1
                grid = {"width": width, "height": height, "resolution": resolution,
                        "origin": origin, "data": data}
                self.assertFalse(footprint_is_free(grid, pose))

    def test_footprint_transform_respects_rotated_map_origin(self):
        rotated = {
            "width": 40, "height": 40, "resolution": 0.05,
            "origin": [2.0, -1.0, math.pi / 2], "data": [0] * 1600,
        }
        plain = {"width": 40, "height": 40, "resolution": 0.05,
                 "origin": [0.0, 0.0, 0.0], "data": [0] * 1600}
        # Both poses map to the same cell-center and robot orientation in grid coordinates.
        rotated_pose = (0.975, 0.025, math.pi / 2)
        plain_pose = (1.025, 1.025, 0.0)
        rotated_cells = set(_cells_under_footprint(
            40, 40, 0.05, rotated["origin"], rotated_pose[:2], rotated_pose[2],
            PIRACER_FOOTPRINT,
        ))
        plain_cells = set(_cells_under_footprint(
            40, 40, 0.05, plain["origin"], plain_pose[:2], plain_pose[2],
            PIRACER_FOOTPRINT,
        ))
        self.assertEqual(plain_cells, rotated_cells)
        self.assertTrue(footprint_is_free(rotated, rotated_pose))

        data = list(rotated["data"])
        x, y = sorted(rotated_cells)[0]
        data[y * rotated["width"] + x] = -1
        rotated["data"] = data
        self.assertFalse(footprint_is_free(rotated, rotated_pose))

    def test_cp40_goal2_costmap_footprint_unknown_cells_are_rejected(self):
        evidence = json.loads(
            (FIXTURES / "cp40_goal2_footprint.json").read_text()
        )
        goal = evidence["goal"]
        self.assertEqual([list(point) for point in PIRACER_FOOTPRINT], goal["footprint"])
        for topic in ("/nav2/map", "/global_costmap/costmap"):
            with self.subTest(topic=topic):
                snapshot = evidence["snapshots"][topic]
                data = [0] * (snapshot["width"] * snapshot["height"])
                for cell in snapshot["intersected_cells"]:
                    data[cell["y"] * snapshot["width"] + cell["x"]] = cell["value"]
                grid = {
                    "width": snapshot["width"], "height": snapshot["height"],
                    "resolution": snapshot["resolution"], "origin": snapshot["origin"],
                    "data": data,
                }
                self.assertEqual(3, len(snapshot["center_unknown_cells"]))
                self.assertEqual(6, snapshot["counts"]["-1"])
                self.assertFalse(footprint_is_free(
                    grid, (*goal["position"], goal["yaw"])
                ))

        alternative = evidence["cp40_reevaluation"]["nearest_safe_alternative_same_frontier"]
        self.assertEqual("x30_y91_d3", alternative["id"])
        self.assertAlmostEqual(0.1414, alternative["shift_m"], places=3)
        self.assertIn([28, 87], alternative["source_unknown_visible"])
        for topic, cells in alternative["intersected_cells"].items():
            with self.subTest(alternative_topic=topic):
                snapshot = evidence["snapshots"][topic]
                data = [0] * (snapshot["width"] * snapshot["height"])
                for cell in cells:
                    data[cell["y"] * snapshot["width"] + cell["x"]] = cell["value"]
                grid = {
                    "width": snapshot["width"], "height": snapshot["height"],
                    "resolution": snapshot["resolution"], "origin": snapshot["origin"],
                    "data": data,
                }
                x, y = alternative["pose_world"][:2]
                self.assertTrue(footprint_is_free(
                    grid, (x, y, alternative["goal_yaw"])
                ))

    def test_unsafe_observation_point_uses_nearby_reachable_viewpoint(self):
        width, height = 20, 15
        resolution = 0.05
        data = [100] * (width * height)
        for y in range(4, 11):
            for x in range(4, 13):
                data[y * width + x] = 0
        data[7 * width + 13] = -1
        grid = {"width": width, "height": height, "resolution": resolution,
                "origin": [0.0, 0.0, 0.0], "data": data}
        robot_pose = ((5.5) * resolution, (7.5) * resolution, 0.0)
        result = evaluate_search(grid, robot_pose)

        self.assertEqual(0.30, MAX_VIEWPOINT_SHIFT_M)
        self.assertEqual("RECOMMENDATION_AVAILABLE", result["status"])
        chosen = result["candidates"][0]
        self.assertIn("source_frontier_world", chosen)
        self.assertIn("source_seed_world", chosen)
        self.assertIn("cluster_size", chosen)
        self.assertTrue(footprint_is_free(
            grid, (*chosen["pose_world"][:2], chosen["goal_yaw"])
        ))
