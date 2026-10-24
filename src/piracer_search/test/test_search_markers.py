import unittest

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from piracer_search.search_markers import (
    _candidate_markers,
    _score_diameters,
    _stale_marker_deletions,
)

class FrontierMarkerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"score": -0.2, "pose_world": [1.0, 2.0, 0.0]},
            {"score": 0.0, "pose_world": [3.0, 4.0, 0.5]},
            {"score": 0.4, "pose_world": [5.0, 6.0, 1.0]},
        ]

    def test_higher_score_gives_larger_marker(self):
        sizes = _score_diameters(self.candidates)
        self.assertEqual(0.08, sizes[0])
        self.assertAlmostEqual(0.17, sizes[1])
        self.assertEqual(0.35, sizes[2])

    def test_equal_scores_keep_equal_marker_sizes(self):
        equal = [{"score": 0.25}, {"score": 0.25}, {"score": 0.25}]
        self.assertEqual([0.215, 0.215, 0.215], _score_diameters(equal))

    def test_all_candidates_have_map_markers_and_selected_has_gold_ring(self):
        markers = _candidate_markers(
            self.candidates, self.candidates[2], "map", Time(), max_labels=3
        )
        candidate_markers = [m for m in markers if m.ns == "frontier_candidates"]
        rings = [m for m in markers if m.ns == "selected_candidate_ring"]
        labels = [m for m in markers if m.ns == "candidate_scores"]
        self.assertEqual(3, len(candidate_markers))
        self.assertTrue(all(m.header.frame_id == "map" for m in candidate_markers))
        self.assertEqual((5.0, 6.0), (rings[0].points[0].x - (0.35 / 2 + 0.045), rings[0].points[0].y))
        self.assertEqual(1, len(rings))
        self.assertEqual(3, len(labels))
        self.assertIn("score=0.400", {label.text for label in labels})
        self.assertEqual(Marker.LINE_STRIP, rings[0].type)
        self.assertEqual(
            0.35,
            next(marker.scale.x for marker in candidate_markers if marker.id == 2),
        )

    def test_labels_are_bounded_to_highest_scoring_candidates(self):
        markers = _candidate_markers(
            self.candidates, self.candidates[2], "map", Time(), max_labels=2
        )
        labels = [m for m in markers if m.ns == "candidate_scores"]
        self.assertEqual(2, len(labels))
        self.assertEqual({"score=0.000", "score=0.400"}, {m.text for m in labels})

    def test_only_six_highest_scoring_candidates_are_rendered(self):
        candidates = [
            {"score": float(index), "pose_world": [float(index), 0.0, 0.0]}
            for index in range(8)
        ]
        markers = _candidate_markers(
            candidates, candidates[-1], "map", Time()
        )
        candidate_markers = [m for m in markers if m.ns == "frontier_candidates"]
        labels = [m for m in markers if m.ns == "candidate_scores"]
        rings = [m for m in markers if m.ns == "selected_candidate_ring"]

        self.assertEqual(6, len(candidate_markers))
        self.assertEqual({2, 3, 4, 5, 6, 7}, {m.id for m in candidate_markers})
        self.assertEqual({"score=2.000", "score=3.000", "score=4.000",
                          "score=5.000", "score=6.000", "score=7.000"},
                         {m.text for m in labels})
        self.assertEqual(1, len(rings))

    def test_selected_goal_ring_remains_visible_outside_top_six(self):
        candidates = [
            {"score": float(index), "pose_world": [float(index), 0.0, 0.0]}
            for index in range(8)
        ]
        selected = candidates[0]
        markers = _candidate_markers(candidates, selected, "map", Time())
        candidate_markers = [m for m in markers if m.ns == "frontier_candidates"]
        rings = [m for m in markers if m.ns == "selected_candidate_ring"]

        self.assertEqual(6, len(candidate_markers))
        self.assertEqual(1, len(rings))
        self.assertAlmostEqual(selected["pose_world"][0] + 0.35 / 2 + 0.045,
                               rings[0].points[0].x)

    def test_selected_goal_ring_is_rendered_when_current_candidates_are_empty(self):
        selected = {"score": 0.2, "pose_world": [9.0, -2.0, 0.0]}
        markers = _candidate_markers(
            [], selected, "current_map", Time(), selected_frame="submitted_map"
        )

        self.assertEqual([], [m for m in markers if m.ns == "frontier_candidates"])
        rings = [m for m in markers if m.ns == "selected_candidate_ring"]
        self.assertEqual(1, len(rings))
        self.assertEqual("submitted_map", rings[0].header.frame_id)
        self.assertAlmostEqual(9.0 + 0.35 / 2 + 0.045, rings[0].points[0].x)

    def test_empty_candidates_without_selection_emit_no_add_markers(self):
        self.assertEqual([], _candidate_markers([], None, "map", Time()))

    def test_removed_frontier_markers_are_deleted_without_deleteall(self):
        previous = {("frontier_candidates", 0), ("candidate_scores", 0),
                    ("another_component", 4)}
        current = {("frontier_candidates", 0)}
        deletions = _stale_marker_deletions(previous, current, "map", Time())
        self.assertEqual({("candidate_scores", 0), ("another_component", 4)},
                         {(m.ns, m.id) for m in deletions})
        self.assertTrue(all(m.action == Marker.DELETE for m in deletions))

if __name__ == '__main__':
    unittest.main()
