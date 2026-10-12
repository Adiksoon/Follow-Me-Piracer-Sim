import importlib.util
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
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
        self.assertEqual(16, len(train))
        self.assertEqual(4, len(val))
        self.assertTrue(all(
            any(low <= sample.azimuth_deg < high
                for low, high in generator.TRAIN_AZIMUTH_INTERVALS_DEG)
            for sample in train
        ))
        self.assertTrue(all(
            any(low <= sample.azimuth_deg < high
                for low, high in generator.PILOT_VALIDATION_AZIMUTH_INTERVALS_DEG)
            for sample in val
        ))
        self.assertTrue(all(
            generator.CAMERA_DISTANCE_RANGE_M[0] <= sample.camera_distance_m
            <= generator.CAMERA_DISTANCE_RANGE_M[1]
            for sample in self.samples
        ))
        self.assertTrue(all(
            generator.CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[0]
            <= sample.camera_height_above_target_m
            <= generator.CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[1]
            for sample in self.samples
        ))
        self.assertEqual(20, len({sample.sample_id for sample in self.samples}))
        self.assertEqual(set(range(0, 20)), {int(sample.sample_id.rsplit("_", 1)[1]) for sample in self.samples})


class IndependentCameraPlanAndDiagnosticTests(unittest.TestCase):
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

    def test_loads_exact_test_pose_and_binds_source_manifest(self):
        sample = generator.PilotSample(
            sample_id="seed20261012_view_0004", split="test",
            scenario_id="test_azimuth_band_00_234_270",
            camera_x_m=2.4, camera_y_m=-3.7, camera_z_m=1.3,
            camera_yaw_deg=94.2, camera_pitch_deg=13.1,
            azimuth_deg=261.5, camera_distance_m=3.82,
            camera_height_above_target_m=0.81, random_seed=20261012,
            image_target_x_fraction=0.67, image_target_y_fraction=0.46,
            camera_link_quaternion_xyzw=(0.1, 0.2, 0.3, 0.9),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "capture_manifest.json"
            source.write_text(json.dumps({
                "world_sha256": "world-hash", "mesh_sha256": "mesh-hash",
                "run_token": "source-run", "pilot_plan": [generator.asdict(sample)],
            }), encoding="utf-8")
            loaded, provenance = generator.diagnostic_sample_from_manifest(
                source, sample.sample_id, "world-hash", "mesh-hash",
            )
            self.assertEqual(sample, loaded)
            self.assertEqual(sample.sample_id, provenance["sample_id"])
            self.assertEqual("source-run", provenance["source_run_token"])
            self.assertEqual(generator.sha256_file(source), provenance["source_manifest_sha256"])

    def test_rejects_manifest_for_different_scene(self):
        sample = generator.PilotSample(
            sample_id="view", split="test", scenario_id="test_azimuth_band_00_234_270",
            camera_x_m=2.4, camera_y_m=-3.7, camera_z_m=1.3,
            camera_yaw_deg=94.2, camera_pitch_deg=13.1,
            azimuth_deg=261.5, camera_distance_m=3.82,
            camera_height_above_target_m=0.81, random_seed=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "capture_manifest.json"
            source.write_text(json.dumps({
                "world_sha256": "different-world", "mesh_sha256": "mesh-hash",
                "pilot_plan": [generator.asdict(sample)],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different approved world"):
                generator.diagnostic_sample_from_manifest(
                    source, sample.sample_id, "world-hash", "mesh-hash",
                )

    def test_camera_maps_suitcase_center_to_planned_image_point(self):
        for sample in self.samples:
            position = (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m)
            _, optical_q = generator.camera_optical_pose(
                position, sample.camera_link_quaternion_xyzw,
            )
            target_camera = (
                np.asarray(self.target_center) - np.asarray(position)
            ) @ generator.quaternion_matrix_xyzw(optical_q)
            focal = generator.CAMERA_WIDTH / (
                2.0 * np.tan(generator.CAMERA_HORIZONTAL_FOV_RAD / 2.0)
            )
            projected_x = focal * target_camera[0] / target_camera[2] + generator.CAMERA_WIDTH / 2.0
            projected_y = focal * target_camera[1] / target_camera[2] + generator.CAMERA_HEIGHT / 2.0
            self.assertAlmostEqual(
                sample.image_target_x_fraction * generator.CAMERA_WIDTH,
                projected_x, delta=1e-6,
            )
            self.assertAlmostEqual(
                sample.image_target_y_fraction * generator.CAMERA_HEIGHT,
                projected_y, delta=1e-6,
            )

    def test_optical_frame_forward_axis_matches_camera_link_forward_axis(self):
        sample = self.samples[3]
        link_q = sample.camera_link_quaternion_xyzw
        _, optical_q = generator.camera_optical_pose(
            (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m), link_q,
        )
        link_forward = generator.quaternion_matrix_xyzw(link_q)[:, 0]
        optical_forward = generator.quaternion_matrix_xyzw(optical_q)[:, 2]
        np.testing.assert_allclose(link_forward, optical_forward, atol=1e-8)

    def test_actual_camera_pose_guard_checks_position_range_and_look_direction(self):
        sample = self.samples[0]
        position = (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m)
        quaternion = sample.camera_link_quaternion_xyzw
        distance, position_delta = generator.validate_camera_sample_pose(
            sample, position, quaternion, self.target_center,
        )
        self.assertAlmostEqual(sample.camera_distance_m, distance)
        self.assertAlmostEqual(0.0, position_delta)
        wrong_position = (position[0] + 0.2, position[1], position[2])
        with self.assertRaisesRegex(ValueError, "range|look-at|planned pose"):
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

    def test_full_dataset_and_candidate_pool_use_disjoint_azimuth_splits(self):
        world = generator.ROOT / generator.WORLD_REL
        mesh = generator.load_collada_geometry(generator.ROOT / generator.MESH_REL)
        target_position, target_q = generator.target_model_pose(world)
        target_center = generator.suitcase_visual_center(target_position, target_q, mesh)
        final_plan = generator.pilot_samples(
            target_center, generator.AUTHORIZED_FULL_DATASET_SAMPLES,
            generator.DEFAULT_RANDOM_SEED, world,
        )
        candidate_plan = generator.pilot_samples(
            target_center,
            generator.AUTHORIZED_FULL_DATASET_SAMPLES * generator.CAPTURE_CANDIDATE_MULTIPLIER,
            generator.DEFAULT_RANDOM_SEED, world,
        )
        self.assertEqual({"train": 320, "val": 40, "test": 40},
                         generator.dataset_split_counts(len(final_plan)))
        self.assertEqual({"train": 640, "val": 80, "test": 80},
                         generator.dataset_split_counts(len(candidate_plan)))
        self.assertEqual({"train": 240, "val": 30, "test": 30},
                         generator.dataset_split_counts(300))
        self.assertEqual({"train": 80, "val": 10, "test": 10},
                         generator.dataset_split_counts(100))
        self.assertEqual(600, generator.capture_candidate_count(300))
        self.assertEqual(200, generator.capture_candidate_count(100))
        self.assertEqual(20, generator.capture_candidate_count(20))
        self.assertEqual(
            ((150.0, 210.0),),
            generator.split_azimuth_intervals(400)["test"],
        )
        for samples, expected_counts in (
            (final_plan, {"train": 320, "val": 40, "test": 40}),
            (candidate_plan, {"train": 640, "val": 80, "test": 80}),
        ):
            self.assertEqual(expected_counts, {
                split: sum(sample.split == split for sample in samples)
                for split in expected_counts
            })
            self.assertEqual(len(samples), len({sample.sample_id for sample in samples}))
            summary = generator.pilot_coverage_summary(samples)
            self.assertTrue(all(not values for values in summary["split_scenario_overlap"].values()))
            for sample in samples:
                intervals = generator.split_azimuth_intervals(len(samples))[sample.split]
                self.assertTrue(any(low <= sample.azimuth_deg < high for low, high in intervals))
                self.assertGreaterEqual(sample.image_target_x_fraction, generator.IMAGE_TARGET_X_FRACTION_RANGE[0])
                self.assertLessEqual(sample.image_target_x_fraction, generator.IMAGE_TARGET_X_FRACTION_RANGE[1])
                self.assertGreaterEqual(sample.image_target_y_fraction, generator.IMAGE_TARGET_Y_FRACTION_RANGE[0])
                self.assertLessEqual(sample.image_target_y_fraction, generator.IMAGE_TARGET_Y_FRACTION_RANGE[1])
        boxes = generator.projected_pilot_boxes(
            candidate_plan, target_position, target_q, target_center, mesh,
        )
        self.assertEqual(len(candidate_plan), len(boxes))

    def test_empty_scene_is_pinned_and_reuses_the_approved_target_pose(self):
        empty_world = generator.ROOT / generator.EMPTY_WORLD_REL
        self.assertEqual("empty", generator.scene_variant_for_world(empty_world))
        self.assertEqual(generator.EMPTY_WORLD_SHA256, generator.sha256_file(empty_world))
        self.assertEqual([], generator.production_collision_boxes(empty_world))
        production_world = generator.ROOT / generator.WORLD_REL
        self.assertEqual(
            generator.target_model_pose(production_world),
            generator.target_model_pose(empty_world),
        )
        mesh_path = generator.ROOT / generator.MESH_REL
        world_hash, mesh_hash = generator.check_scene_files(empty_world, mesh_path)
        self.assertEqual(generator.EMPTY_WORLD_SHA256, world_hash)
        self.assertEqual(generator.MESH_SHA256, mesh_hash)

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
        self.assertEqual((124944, 3), mesh.triangles.shape)
        self.assertGreaterEqual(int(mesh.triangles.min()), 0)
        self.assertLess(int(mesh.triangles.max()), mesh.vertices.shape[0])
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
        self.assertIn('DeclareLaunchArgument("scene", default_value="production_room")', launch_source)
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
        pixels = np.array(((5, 5), (14, 5), (14, 14), (5, 14)), dtype=np.float64)
        z = np.full(pixels.shape[0], 2.0)
        points = np.column_stack(((pixels[:, 0] - 20) * z / 100,
                                  (pixels[:, 1] - 20) * z / 100, z))
        triangles = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        visible_depth = np.full((40, 40), 2.0, dtype=np.float32)
        visible = generator.depth_visibility(points, triangles, intrinsics, visible_depth)
        self.assertEqual(1.0, visible["visible_fraction"])
        self.assertEqual(100, visible["projected_mesh_pixels"])
        occluding_depth = visible_depth.copy()
        occluding_depth[5:15, 5:15] = 1.0
        occluded = generator.depth_visibility(points, triangles, intrinsics, occluding_depth)
        self.assertEqual(0.0, occluded["visible_fraction"])

    def test_surface_depth_rasterizer_uses_perspective_depth_and_nearest_surface(self):
        intrinsics = (100.0, 100.0, 0.0, 0.0)
        screen = np.array(((10.0, 10.0), (20.0, 10.0), (10.0, 20.0)))
        depths = np.array((1.0, 2.0, 4.0))
        slanted = np.column_stack((
            screen[:, 0] * depths / intrinsics[0],
            screen[:, 1] * depths / intrinsics[1],
            depths,
        ))
        expected, _ = generator.rasterize_front_surface_depth(
            slanted, np.array(((0, 1, 2),)), intrinsics, 32, 32,
        )
        self.assertAlmostEqual(1.0 / (0.6 / 1.0 + 0.2 / 2.0 + 0.2 / 4.0),
                               float(expected[12, 12]), places=5)

        near = np.column_stack((screen / 100.0, np.ones(3)))
        far = np.column_stack((screen * 3.0 / 100.0, np.full(3, 3.0)))
        overlapping, _ = generator.rasterize_front_surface_depth(
            np.concatenate((far, near)), np.array(((0, 1, 2), (3, 4, 5))),
            intrinsics, 32, 32,
        )
        self.assertAlmostEqual(1.0, float(overlapping[12, 12]), places=6)

    def test_surface_depth_rasterizer_rejects_unsupported_distortion(self):
        points = np.array(((-0.1, -0.1, 1.0), (0.1, -0.1, 1.0), (0.0, 0.1, 1.0)))
        with self.assertRaisesRegex(generator.CaptureRejected, "NONZERO_CAMERA_DISTORTION"):
            generator.rasterize_front_surface_depth(
                points, np.array(((0, 1, 2),)), (100.0, 100.0, 16.0, 16.0),
                32, 32, (0.01, 0.0, 0.0, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
