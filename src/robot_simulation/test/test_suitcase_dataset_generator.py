import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "suitcase_dataset_generator.py"
SPEC = importlib.util.spec_from_file_location("suitcase_dataset_generator", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class PilotPlanTests(unittest.TestCase):
    def test_pilot_has_20_samples_and_keeps_distance_groups_separate(self):
        samples = generator.pilot_samples()
        self.assertEqual(20, len(samples))
        self.assertEqual(10, sum(sample.split == "train" for sample in samples))
        self.assertEqual(10, sum(sample.split == "val" for sample in samples))
        self.assertEqual({2.4}, {sample.camera_forward_m for sample in samples if sample.split == "train"})
        self.assertEqual({3.0}, {sample.camera_forward_m for sample in samples if sample.split == "val"})
        self.assertEqual(set(generator.PILOT_RELATIVE_YAWS_DEG), {sample.relative_yaw_deg for sample in samples})
        self.assertEqual(20, len({sample.sample_id for sample in samples}))

    def test_camera_relative_plan_uses_measured_heading_and_preserves_clearance(self):
        base_yaw = np.deg2rad(-5.043388934)
        base_quaternion = (
            0.0, 0.0, np.sin(base_yaw / 2.0), np.cos(base_yaw / 2.0)
        )
        base_to_optical, _ = generator.fixed_transform_between_links(
            generator.robot_description_xacro_path(), "base_link", "sim_camera_optical_frame"
        )
        camera_world = generator._matrix_from_pose(
            (0.10402347, 0.00283562, 0.01900026), base_quaternion,
        ) @ base_to_optical
        camera_position = tuple(float(value) for value in camera_world[:3, 3])
        camera_quaternion = generator._quaternion_xyzw_from_matrix(camera_world[:3, :3])
        camera_forward = generator.quaternion_matrix_xyzw(camera_quaternion)[:, 2]
        self.assertGreater(abs(camera_forward[1]), 0.05)

        samples = generator.pilot_samples((camera_position, camera_quaternion))
        first = samples[0]
        forward_distance, lateral_distance = generator.validate_camera_relative_target(
            camera_position,
            camera_quaternion,
            (first.x_m, first.y_m, first.z_m),
            first.camera_forward_m,
        )
        self.assertAlmostEqual(2.4, forward_distance, places=6)
        self.assertLessEqual(lateral_distance, 0.25)
        heading_deg = np.rad2deg(np.arctan2(camera_forward[1], camera_forward[0]))
        self.assertAlmostEqual(
            (heading_deg + first.relative_yaw_deg) % 360.0,
            first.yaw_deg,
            places=6,
        )
        obstacles = generator.validate_pilot_clearance(
            generator.ROOT / generator.WORLD_REL, samples,
        )
        self.assertEqual(88, len(obstacles))

    def test_camera_relative_guard_rejects_short_or_lateral_target(self):
        camera_position, camera_quaternion = generator.nominal_camera_pose()
        forward = generator.quaternion_matrix_xyzw(camera_quaternion)[:, 2]
        right = generator.quaternion_matrix_xyzw(camera_quaternion)[:, 0]
        too_close = np.asarray(camera_position) + 1.0 * forward
        outside_lateral_clearance = np.asarray(camera_position) + 2.4 * forward + 0.30 * right
        with self.assertRaisesRegex(ValueError, "forward distance"):
            generator.validate_camera_relative_target(
                camera_position, camera_quaternion, tuple(too_close), 1.0,
            )
        with self.assertRaisesRegex(ValueError, "lateral clearance"):
            generator.validate_camera_relative_target(
                camera_position, camera_quaternion, tuple(outside_lateral_clearance), 2.4,
            )


class CaptureSafeguardTests(unittest.TestCase):
    def test_set_entity_state_request_uses_only_declared_request_fields(self):
        class SetRequest:
            __slots__ = ("state",)

            def __init__(self):
                self.state = SimpleNamespace(
                    name="",
                    reference_frame="",
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )

        sample = generator.PilotSample(
            "sample", "train", "scenario", 2.7, 0.0, 0.05, 90.0, 2.7, 90,
        )
        request = generator.populate_set_entity_state_request(SetRequest(), "suitcase_target", sample)
        self.assertEqual("suitcase_target", request.state.name)
        self.assertEqual("world", request.state.reference_frame)
        self.assertEqual((2.7, 0.0, 0.05), (
            request.state.pose.position.x,
            request.state.pose.position.y,
            request.state.pose.position.z,
        ))
        self.assertAlmostEqual(np.sqrt(0.5), request.state.pose.orientation.z)
        self.assertAlmostEqual(np.sqrt(0.5), request.state.pose.orientation.w)

    def test_rgb_and_depth_must_both_follow_the_current_pose_timestamp(self):
        self.assertTrue(generator.rgbd_timestamps_are_fresh(10.02, 10.03, 10.01, 0.05))
        self.assertFalse(generator.rgbd_timestamps_are_fresh(10.00, 10.03, 10.01, 0.05))
        self.assertFalse(generator.rgbd_timestamps_are_fresh(10.03, 10.00, 10.01, 0.05))
        self.assertFalse(generator.rgbd_timestamps_are_fresh(10.02, 10.10, 10.01, 0.05))

    def test_capture_pose_check_compares_full_translation_and_orientation(self):
        position_delta, orientation_delta = generator.capture_pose_delta(
            (2.7, 0.0, 0.05),
            (0.0, 0.0, 0.0, 1.0),
            (2.7, 0.0, 0.07),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(0.02, position_delta)
        self.assertAlmostEqual(0.0, orientation_delta)
        _, yaw_delta = generator.capture_pose_delta(
            (2.7, 0.0, 0.05),
            (0.0, 0.0, 0.0, 1.0),
            (2.7, 0.0, 0.05),
            (0.0, 0.0, np.sin(np.deg2rad(6.0)), np.cos(np.deg2rad(6.0))),
        )
        self.assertAlmostEqual(np.deg2rad(12.0), yaw_delta)


class CameraPoseSourceTests(unittest.TestCase):
    def test_fixed_transform_chain_comes_from_the_canonical_xacro(self):
        xacro = generator.ROOT / generator.ROBOT_DESCRIPTION_XACRO_REL
        sensor_transform, sensor_frames = generator.fixed_transform_between_links(
            xacro, "base_link", "sim_camera_link"
        )
        optical_transform, optical_frames = generator.fixed_transform_between_links(
            xacro, "base_link", "sim_camera_optical_frame"
        )
        self.assertEqual(("base_link", "chassis", "sim_camera_link"), sensor_frames)
        self.assertEqual(("base_link", "chassis", "sim_camera_link", "sim_camera_optical_frame"), optical_frames)
        np.testing.assert_allclose((0.18, 0.0, 0.12), sensor_transform[:3, 3], atol=1e-9)
        np.testing.assert_allclose(np.eye(3), sensor_transform[:3, :3], atol=1e-9)
        np.testing.assert_allclose((0.18, 0.0, 0.12), optical_transform[:3, 3], atol=1e-9)
        np.testing.assert_allclose(
            [[3.267948965e-7, 3.267948965e-7, 1.0],
             [-1.0, 1.067949044e-13, 3.267948965e-7],
             [0.0, -1.0, 3.267948965e-7]],
            optical_transform[:3, :3], atol=1e-9,
        )

    def test_camera_pose_uses_exact_base_link_and_composes_rotation_and_translation(self):
        pose = SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=np.sqrt(0.5), w=np.sqrt(0.5)),
        )
        states = SimpleNamespace(name=["other::base_link", "piracer::base_link"], pose=[pose, pose])
        xacro = generator.ROOT / generator.ROBOT_DESCRIPTION_XACRO_REL
        base_to_optical, _ = generator.fixed_transform_between_links(
            xacro, "base_link", "sim_camera_optical_frame"
        )
        position, quaternion = generator.camera_pose_from_link_states(
            states, "piracer::base_link", base_to_optical,
            received_at_monotonic=10.0, now_monotonic=10.1, max_age_s=0.5,
        )
        np.testing.assert_allclose((1.0, 2.18, 3.12), position, atol=1e-9)
        self.assertAlmostEqual(1.0, np.linalg.norm(quaternion))

    def test_missing_stale_duplicate_and_invalid_base_link_states_fail_safe(self):
        pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        transform = np.eye(4)
        fresh = SimpleNamespace(name=["piracer::base_link"], pose=[pose])
        arguments = dict(
            base_to_optical=transform, received_at_monotonic=1.0,
            now_monotonic=1.1, max_age_s=0.5,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            generator.camera_pose_from_link_states(
                SimpleNamespace(name=["other::base_link"], pose=[pose]),
                "piracer::base_link", **arguments,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            generator.camera_pose_from_link_states(
                SimpleNamespace(name=["piracer::base_link", "piracer::base_link"], pose=[pose, pose]),
                "piracer::base_link", **arguments,
            )
        with self.assertRaisesRegex(ValueError, "stale"):
            generator.camera_pose_from_link_states(
                fresh, "piracer::base_link", base_to_optical=transform,
                received_at_monotonic=1.0, now_monotonic=1.6, max_age_s=0.5,
            )
        invalid = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )
        with self.assertRaisesRegex(ValueError, "Quaternion"):
            generator.camera_pose_from_link_states(
                SimpleNamespace(name=["piracer::base_link"], pose=[invalid]),
                "piracer::base_link", **arguments,
            )


class GeometryTests(unittest.TestCase):
    def test_approved_dae_geometry_is_read_from_the_real_mesh(self):
        mesh = generator.load_collada_geometry(generator.ROOT / generator.MESH_REL)
        self.assertEqual(5, mesh.geometry_count)
        self.assertEqual(83104, mesh.vertices.shape[0])
        self.assertEqual(124944, mesh.triangle_count)
        self.assertAlmostEqual(1.0065, mesh.local_bounds_max[2], places=3)

    def test_pilot_footprints_clear_approved_static_world_collisions(self):
        world = generator.ROOT / generator.WORLD_REL
        obstacles = generator.validate_pilot_clearance(world)
        self.assertEqual(88, len(obstacles))

    def test_identity_camera_projection_and_yolo_normalization(self):
        camera_points = np.array([
            [-0.5, -0.25, 2.0],
            [0.5, 0.25, 2.0],
            [0.0, 0.0, 2.0],
        ])
        bbox = generator.projected_bbox(camera_points, (100.0, 100.0, 100.0, 50.0), 200, 100)
        self.assertEqual((75.0, 37.5, 125.0, 62.5), bbox)
        normalized = generator.yolo_bbox(bbox, 200, 100)
        np.testing.assert_allclose((0.5, 0.5, 0.25, 0.25), normalized)

    def test_pose_transform_respects_ros_xyzw_quaternion(self):
        points_camera = generator.pose_points_to_camera(
            np.array([[1.0, 0.0, 0.0]]),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose([[1.0, 0.0, 0.0]], points_camera)

    def test_distorted_projection_uses_camera_calibration(self):
        points = np.array([[1.0, 0.0, 2.0]])
        u, v, _ = generator.project_camera_points(
            points, (100.0, 100.0, 100.0, 50.0), (0.2, 0.0, 0.0, 0.0, 0.0)
        )
        self.assertGreater(u[0], 150.0)
        self.assertAlmostEqual(50.0, v[0])

    def test_edge_clipped_geometry_is_rejected(self):
        camera_points = np.array([
            [-2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.5, 2.0],
        ])
        with self.assertRaisesRegex(generator.CaptureRejected, "CLIPPED"):
            generator.projected_bbox(camera_points, (100.0, 100.0, 100.0, 50.0), 200, 100)

    def test_aligned_depth_confirms_visible_geometry_and_detects_occlusion(self):
        intrinsics = (100.0, 100.0, 20.0, 20.0)
        pixels = np.array([(x, y) for y in range(5, 15) for x in range(5, 15)], dtype=np.float64)
        z = np.full((pixels.shape[0], 1), 2.0)
        points = np.column_stack(((pixels[:, 0] - 20) * z[:, 0] / 100,
                                  (pixels[:, 1] - 20) * z[:, 0] / 100,
                                  z[:, 0]))
        visible_depth = np.full((40, 40), 2.0, dtype=np.float32)
        visible = generator.depth_visibility(points, intrinsics, visible_depth)
        self.assertEqual(1.0, visible["visible_fraction"])

        occluding_depth = visible_depth.copy()
        occluding_depth[5:15, 5:15] = 1.0
        occluded = generator.depth_visibility(points, intrinsics, occluding_depth)
        self.assertEqual(0.0, occluded["visible_fraction"])


if __name__ == "__main__":
    unittest.main()
