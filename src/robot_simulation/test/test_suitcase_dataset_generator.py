import importlib.util
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "suitcase_dataset_generator.py"
SPEC = importlib.util.spec_from_file_location("suitcase_dataset_generator", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class IndependentCameraPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = generator.ROOT / generator.WORLD_REL
        cls.mesh_path = generator.ROOT / generator.MESH_REL
        cls.mesh = generator.load_collada_geometry(cls.mesh_path)
        cls.target_position, cls.target_q = generator.target_model_pose(cls.world)
        cls.target_center = generator.suitcase_visual_center(
            cls.target_position, cls.target_q, cls.mesh,
        )
        cls.samples = generator.pilot_samples(cls.target_center)

    def test_pilot_covers_full_azimuth_at_two_ranges(self):
        self.assertEqual(20, len(self.samples))
        train = [sample for sample in self.samples if sample.split == "train"]
        val = [sample for sample in self.samples if sample.split == "val"]
        self.assertEqual(12, len(train))
        self.assertEqual(8, len(val))
        self.assertEqual(set(range(0, 360, 30)), {sample.azimuth_deg for sample in train})
        self.assertEqual(set(generator.PILOT_VAL_AZIMUTHS_DEG), {sample.azimuth_deg for sample in val})
        self.assertEqual({2.4}, {sample.camera_distance_m for sample in train})
        self.assertEqual({3.0}, {sample.camera_distance_m for sample in val})
        self.assertEqual(20, len({sample.sample_id for sample in self.samples}))
        for sample in self.samples:
            position = np.asarray((sample.camera_x_m, sample.camera_y_m, sample.camera_z_m))
            center = np.asarray(self.target_center)
            self.assertAlmostEqual(sample.camera_distance_m, np.linalg.norm((position - center)[:2]))
            self.assertAlmostEqual(
                generator.PILOT_CAMERA_HEIGHT_ABOVE_TARGET_M,
                position[2] - center[2],
            )

    def test_camera_link_positive_x_points_at_dae_visual_center(self):
        for sample in self.samples:
            position = (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m)
            quaternion = generator.camera_link_orientation(position, self.target_center)
            forward = generator.quaternion_matrix_xyzw(quaternion)[:, 0]
            target_direction = np.asarray(self.target_center) - np.asarray(position)
            target_direction /= np.linalg.norm(target_direction)
            self.assertAlmostEqual(1.0, float(np.dot(forward, target_direction)), places=7)

    def test_optical_frame_forward_axis_matches_camera_link_forward_axis(self):
        sample = self.samples[3]
        link_q = generator.camera_link_orientation(
            (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m),
            self.target_center,
        )
        _, optical_q = generator.camera_optical_pose(
            (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m), link_q,
        )
        link_forward = generator.quaternion_matrix_xyzw(link_q)[:, 0]
        optical_forward = generator.quaternion_matrix_xyzw(optical_q)[:, 2]
        np.testing.assert_allclose(link_forward, optical_forward, atol=1e-8)

    def test_actual_camera_pose_guard_checks_position_range_and_look_direction(self):
        sample = self.samples[0]
        position = (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m)
        quaternion = generator.camera_link_orientation(position, self.target_center)
        distance, position_delta = generator.validate_camera_sample_pose(
            sample, position, quaternion, self.target_center,
        )
        self.assertAlmostEqual(sample.camera_distance_m, distance)
        self.assertAlmostEqual(0.0, position_delta)
        wrong_position = (position[0] + 0.2, position[1], position[2])
        with self.assertRaisesRegex(ValueError, "range|look-at"):
            generator.validate_camera_sample_pose(
                sample, wrong_position, quaternion, self.target_center,
            )

    def test_all_planned_mesh_boxes_fit_frame_and_camera_path_is_clear(self):
        boxes = generator.projected_pilot_boxes(
            self.samples, self.target_position, self.target_q,
            self.target_center, self.mesh,
        )
        self.assertEqual(20, len(boxes))
        for x1, y1, x2, y2 in boxes.values():
            self.assertGreaterEqual(x1, 2.0)
            self.assertGreaterEqual(y1, 2.0)
            self.assertLess(x2, generator.CAMERA_WIDTH - 2.0)
            self.assertLess(y2, generator.CAMERA_HEIGHT - 2.0)
        obstacles = generator.validate_camera_clearance(self.world, self.samples)
        self.assertEqual(88, len(obstacles))


class CaptureSafeguardTests(unittest.TestCase):
    def test_set_entity_state_moves_camera_and_zeros_twist(self):
        vector = lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0)
        request = SimpleNamespace(state=SimpleNamespace(
            name="",
            reference_frame="",
            pose=SimpleNamespace(
                position=vector(),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            twist=SimpleNamespace(linear=vector(), angular=vector()),
        ))
        sample = generator.pilot_samples((3.0, 0.0, 0.5))[0]
        result = generator.populate_set_entity_state_request(
            request, generator.CAMERA_ENTITY, sample,
        )
        self.assertEqual("dataset_camera", result.state.name)
        self.assertEqual("world", result.state.reference_frame)
        self.assertEqual(
            (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m),
            (result.state.pose.position.x, result.state.pose.position.y, result.state.pose.position.z),
        )
        self.assertAlmostEqual(1.0, np.linalg.norm((
            result.state.pose.orientation.x, result.state.pose.orientation.y,
            result.state.pose.orientation.z, result.state.pose.orientation.w,
        )))
        self.assertEqual((0.0, 0.0, 0.0), (
            result.state.twist.linear.x, result.state.twist.linear.y,
            result.state.twist.linear.z,
        ))

    def test_rgb_and_depth_must_follow_the_camera_state_timestamp(self):
        self.assertTrue(generator.rgbd_timestamps_are_fresh(10.02, 10.03, 10.01, 0.05))
        self.assertFalse(generator.rgbd_timestamps_are_fresh(10.00, 10.03, 10.01, 0.05))
        self.assertFalse(generator.rgbd_timestamps_are_fresh(10.03, 10.00, 10.01, 0.05))
        self.assertFalse(generator.rgbd_timestamps_are_fresh(10.02, 10.10, 10.01, 0.05))

    def test_capture_pose_check_compares_full_translation_and_orientation(self):
        position_delta, orientation_delta = generator.capture_pose_delta(
            (2.7, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0),
            (2.7, 0.0, 0.07), (0.0, 0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(0.02, position_delta)
        self.assertAlmostEqual(0.0, orientation_delta)
        with self.assertRaisesRegex(ValueError, "invalid"):
            generator._unit_quaternion_difference((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def test_generator_refuses_nonapproved_target_before_ros_capture(self):
        with redirect_stderr(StringIO()):
            result = generator.main(["--target-entity", "piracer"])
        self.assertEqual(2, result)


class GeometryAndCameraModelTests(unittest.TestCase):
    def test_approved_dae_and_camera_model_settings(self):
        mesh = generator.load_collada_geometry(generator.ROOT / generator.MESH_REL)
        self.assertEqual(5, mesh.geometry_count)
        self.assertEqual(83104, mesh.vertices.shape[0])
        self.assertEqual(124944, mesh.triangle_count)
        self.assertAlmostEqual(1.0065, mesh.local_bounds_max[2], places=3)

        model_path = generator.SIMULATION_SHARE / "models/dataset_camera/model.sdf"
        root = ET.parse(model_path).getroot()
        camera = root.find(".//sensor/camera")
        self.assertEqual("640", camera.findtext("image/width"))
        self.assertEqual("360", camera.findtext("image/height"))
        self.assertAlmostEqual(1.14075, float(camera.findtext("horizontal_fov")))
        self.assertAlmostEqual(5.0, float(root.findtext(".//sensor/update_rate")))
        self.assertEqual("R8G8B8", camera.findtext("image/format"))
        self.assertEqual("0.05", camera.findtext("clip/near"))
        self.assertEqual("20.0", camera.findtext("clip/far"))
        self.assertEqual("/dataset_camera", root.findtext(".//plugin/ros/namespace"))
        self.assertEqual("dataset_camera_optical_frame", root.findtext(".//plugin/frame_name"))

        launch_source = (PACKAGE / "launch" / "dataset_capture.launch.py").read_text()
        self.assertIn('DeclareLaunchArgument("gui", default_value="false")', launch_source)
        self.assertNotIn("piracer_sim.launch.py", launch_source)
        self.assertNotIn("robot_state_publisher", launch_source)
        self.assertNotIn("yolo_node", launch_source)

    def test_identity_projection_normalization_and_clipped_box_rejection(self):
        camera_points = np.array([
            [-0.5, -0.25, 2.0], [0.5, 0.25, 2.0], [0.0, 0.0, 2.0],
        ])
        bbox = generator.projected_bbox(camera_points, (100.0, 100.0, 100.0, 50.0), 200, 100)
        self.assertEqual((75.0, 37.5, 125.0, 62.5), bbox)
        np.testing.assert_allclose((0.5, 0.5, 0.25, 0.25), generator.yolo_bbox(bbox, 200, 100))
        clipped = np.array([[-2.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.5, 0.5, 2.0]])
        with self.assertRaisesRegex(generator.CaptureRejected, "CLIPPED"):
            generator.projected_bbox(clipped, (100.0, 100.0, 100.0, 50.0), 200, 100)

    def test_aligned_depth_confirms_visible_geometry_and_detects_occlusion(self):
        intrinsics = (100.0, 100.0, 20.0, 20.0)
        pixels = np.array([(x, y) for y in range(5, 15) for x in range(5, 15)], dtype=np.float64)
        z = np.full(pixels.shape[0], 2.0)
        points = np.column_stack(((pixels[:, 0] - 20) * z / 100,
                                  (pixels[:, 1] - 20) * z / 100, z))
        visible_depth = np.full((40, 40), 2.0, dtype=np.float32)
        visible = generator.depth_visibility(points, intrinsics, visible_depth)
        self.assertEqual(1.0, visible["visible_fraction"])
        occluding_depth = visible_depth.copy()
        occluding_depth[5:15, 5:15] = 1.0
        occluded = generator.depth_visibility(points, intrinsics, occluding_depth)
        self.assertEqual(0.0, occluded["visible_fraction"])


if __name__ == "__main__":
    unittest.main()
