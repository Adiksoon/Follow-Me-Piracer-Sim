#!/usr/bin/env python3
"""Prepare a geometry-labelled suitcase pilot with an independent dataset camera.

The capture path is explicit: this script never launches Gazebo. Run it
without ``--capture`` to inspect the camera-orbit plan. With ``--capture``, it
connects to a dataset-camera launch, moves only ``dataset_camera``, and leaves
the approved suitcase in place. The workflow is informed by the
KulunuOS/gazebo_dataset_generation render and camera-sweep concept; labels use
the production DAE, CameraInfo and aligned depth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def _source_workspace_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "src/robot_simulation/worlds").is_dir()
            and (parent / "src/simulation_assets/assets").is_dir()
        ):
            return parent
    return None


SOURCE_ROOT = _source_workspace_root()
ROOT = SOURCE_ROOT or Path.cwd().resolve()
WORLD_REL = Path("src/robot_simulation/worlds/sala_laboratoryjna_production.world")
EMPTY_WORLD_REL = Path("src/robot_simulation/worlds/empty_sky.world")
MESH_REL = Path("src/simulation_assets/assets/hard_trolley_red/meshes/suitcase.dae")


def _package_share(package_name: str) -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(package_name))
    except (ImportError, LookupError):
        return None


SIMULATION_SHARE = (
    SOURCE_ROOT / "src/robot_simulation" if SOURCE_ROOT is not None
    else _package_share("robot_simulation") or (ROOT / "src/robot_simulation")
)
ASSET_SHARE = (
    SOURCE_ROOT / "src/simulation_assets" if SOURCE_ROOT is not None
    else _package_share("simulation_assets") or (ROOT / "src/simulation_assets")
)
DEFAULT_WORLD = SIMULATION_SHARE / "worlds/sala_laboratoryjna_production.world"
DEFAULT_EMPTY_WORLD = SIMULATION_SHARE / "worlds/empty_sky.world"
DEFAULT_MESH = ASSET_SHARE / "assets/hard_trolley_red/meshes/suitcase.dae"
WORLD_SHA256 = "71111ed9cfd933117c3ac7013bed41c85d342eb5cc6b8e4e0eddbeeb5ae39c7e"
EMPTY_WORLD_SHA256 = "00738c78deca99ea6d770451a1664237ac335894c8297f91a9426084c526c321"
MESH_SHA256 = "e4d75b678e566289f15d58fdc7216c93f2199a34bb82377bd785d1e94a14f1ca"
FURNITURE_SHA256 = {
    "lab_table_beige/model.sdf": "f72bd4d9b5f0b50f6375589080842762b38dc8fccef6cca5c54cf7bb5e5a44b9",
    "wooden_chair_blue_gray/model.sdf": "6c0d4d2219af49b9b5737e0a25b36ebe5b69349640d4509eff1f21fe3027b28f",
}
TARGET_ENTITY = "suitcase_target"
CAMERA_ENTITY = "dataset_camera"
OPTICAL_FRAME_NAME = "dataset_camera_optical_frame"
IMAGE_TOPIC = "/dataset_camera/color/image_raw"
DEPTH_TOPIC = "/dataset_camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/dataset_camera/color/camera_info"
DEPTH_CAMERA_INFO_TOPIC = "/dataset_camera/depth/camera_info"
DEFAULT_SAMPLE_COUNT = 20
DEFAULT_RANDOM_SEED = 20261012
MAX_AUTHORIZED_PILOT_SAMPLES = 20
AUTHORIZED_FULL_DATASET_SAMPLES = 400
AUTHORIZED_DATASET_COMPONENT_SIZES = (100, 300)
CAPTURE_CANDIDATE_MULTIPLIER = 2
MAX_FRAME_ACQUISITION_RETRIES = 1
CAMERA_DISTANCE_RANGE_M = (2.4, 4.0)
CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M = (0.35, 1.10)
TRAIN_AZIMUTH_INTERVALS_DEG = (
    (0.0, 44.0), (100.0, 140.0), (220.0, 230.0),
    (240.0, 270.0), (280.0, 360.0),
)
PILOT_VALIDATION_AZIMUTH_INTERVALS_DEG = ((54.0, 90.0), (234.0, 270.0))
VALIDATION_AZIMUTH_INTERVALS_DEG = ((54.0, 90.0),)
TEST_AZIMUTH_INTERVALS_DEG = ((150.0, 210.0),)
IMAGE_TARGET_X_FRACTION_RANGE = (0.30, 0.70)
IMAGE_TARGET_Y_FRACTION_RANGE = (0.40, 0.60)
CAMERA_CLEARANCE_RADIUS_M = 0.15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_UPDATE_RATE_HZ = 5.0
CAMERA_HORIZONTAL_FOV_RAD = 1.14075
CAMERA_LINK_TO_OPTICAL_ROTATION = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)
YOLO_CLASS_ID = 0
YOLO_CLASS_NAME = "suitcase"
MIN_DEPTH_MATCHES = 100
MIN_VISIBLE_FRACTION = 0.80
CAPTURE_POSE_MAX_POSITION_DELTA_M = 0.02
CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD = math.radians(2.0)
DEFAULT_DELIVERY_LATENCY_WARNING_S = 0.5
DEFAULT_MAX_CAMERA_STAMP_DELTA_S = 0.1

COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": COLLADA_NS}


@dataclass(frozen=True)
class PilotSample:
    sample_id: str
    split: str
    scenario_id: str
    camera_x_m: float
    camera_y_m: float
    camera_z_m: float
    camera_yaw_deg: float
    camera_pitch_deg: float
    azimuth_deg: float
    camera_distance_m: float
    camera_height_above_target_m: float
    random_seed: int
    image_target_x_fraction: float = 0.5
    image_target_y_fraction: float = 0.5
    camera_link_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    clearance_resampled: bool = False


@dataclass(frozen=True)
class MeshGeometry:
    vertices: np.ndarray
    triangles: np.ndarray
    local_bounds_min: tuple[float, float, float]
    local_bounds_max: tuple[float, float, float]
    geometry_count: int
    triangle_count: int


class CaptureRejected(RuntimeError):
    """A planned sample failed a predeclared geometric or depth check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def append_bounded_diagnostic_event(events: list[dict], row: dict, limit: int) -> bool:
    if limit < 0:
        raise ValueError("Event limit must be nonnegative")
    if len(events) >= limit:
        return False
    events.append(row)
    return True


def save_diagnostic_cdr(
    output_dir: Path, topic_key: str, callback_index: int, header_stamp_ns: int, payload: bytes,
) -> str:
    allowed_topics = {"rgb", "depth", "rgb_camera_info", "depth_camera_info"}
    if topic_key not in allowed_topics or callback_index < 0:
        raise ValueError("Invalid diagnostic message identity")
    relative_path = (
        Path("raw") / topic_key
        / f"callback_{callback_index:06d}_{header_stamp_ns}.cdr"
    )
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        written = temporary_path.write_bytes(payload)
        if written != len(payload):
            raise OSError(f"Short CDR write for {topic_key} callback {callback_index}")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return relative_path.as_posix()


def diagnostic_raw_history_is_complete(
    callback_counts: dict[str, int],
    persisted_counts: dict[str, int],
    overflow_counts: dict[str, int],
) -> bool:
    return (
        set(persisted_counts).issubset(callback_counts)
        and all(
            persisted_counts.get(key, 0) == count and overflow_counts.get(key, 0) == 0
            for key, count in callback_counts.items()
        )
    )


def camera_info_parameters(info) -> dict:
    roi = info.roi
    return {
        "width": int(info.width),
        "height": int(info.height),
        "distortion_model": info.distortion_model,
        "d": list(info.d),
        "k": list(info.k),
        "r": list(info.r),
        "p": list(info.p),
        "binning_x": int(info.binning_x),
        "binning_y": int(info.binning_y),
        "roi": {
            "x_offset": int(roi.x_offset),
            "y_offset": int(roi.y_offset),
            "width": int(roi.width),
            "height": int(roi.height),
            "do_rectify": bool(roi.do_rectify),
        },
    }


def camera_infos_share_calibration(color_info, depth_info) -> bool:
    if (int(color_info.width) != int(depth_info.width)
            or int(color_info.height) != int(depth_info.height)
            or color_info.distortion_model != depth_info.distortion_model
            or int(color_info.binning_x) != int(depth_info.binning_x)
            or int(color_info.binning_y) != int(depth_info.binning_y)
            or camera_info_parameters(color_info)["roi"] != camera_info_parameters(depth_info)["roi"]):
        return False
    for field in ("d", "k", "r", "p"):
        color_values = np.asarray(getattr(color_info, field), dtype=np.float64)
        depth_values = np.asarray(getattr(depth_info, field), dtype=np.float64)
        if (color_values.shape != depth_values.shape
                or not np.allclose(color_values, depth_values, atol=1e-6, rtol=0.0)):
            return False
    return True


def camera_info_configuration_sha256(color_info, depth_info) -> str:
    return canonical_sha256({
        "rgb": camera_info_parameters(color_info),
        "depth": camera_info_parameters(depth_info),
    })


def write_capture_progress(
    progress_path: Path,
    run_token: str,
    manifest: dict,
    accepted: list[dict],
    rejected: list[dict],
    current_sample_id: str | None,
) -> None:
    write_json_atomic(progress_path, {
        "schema": "data01-capture-progress-v2",
        "run_token": run_token,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": manifest.get("status", "RUNNING"),
        "current_sample_id": current_sample_id,
        "manifest": manifest,
        "accepted_samples": accepted,
        "rejected_samples": rejected,
    })


def scene_variant_for_world(world_path: Path) -> str:
    """Identify one of the two approved DATA-01 backgrounds by its pinned hash."""
    world_hash = sha256_file(Path(world_path))
    if world_hash == WORLD_SHA256:
        return "production_room"
    if world_hash == EMPTY_WORLD_SHA256:
        return "empty"
    raise ValueError(f"World is not an approved DATA-01 scene: SHA-256 {world_hash}")


def target_model_pose(world_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read the target pose; the empty runtime world uses the production model pose."""
    source_path = Path(world_path)
    if scene_variant_for_world(source_path) == "empty":
        source_path = DEFAULT_WORLD
    root = ET.parse(source_path).getroot()
    model = root.find(f".//world/model[@name='{TARGET_ENTITY}']")
    if model is None:
        raise ValueError("Production-world target model is missing")
    xyzrpy = _pose6(model.find("pose"))
    position = tuple(float(value) for value in xyzrpy[:3])
    rotation = _matrix_from_xyz_rpy(position, xyzrpy[3:])[:3, :3]
    return position, _quaternion_xyzw_from_matrix(rotation)


def suitcase_visual_center(
    target_position: tuple[float, float, float],
    target_quaternion_xyzw: tuple[float, float, float, float],
    mesh: MeshGeometry,
) -> tuple[float, float, float]:
    local_center = (
        np.asarray(mesh.local_bounds_min, dtype=np.float64)
        + np.asarray(mesh.local_bounds_max, dtype=np.float64)
    ) * 0.5
    local_center[2] -= 0.05  # suitcase_visual pose in the approved production SDF
    world_center = (
        quaternion_matrix_xyzw(target_quaternion_xyzw) @ local_center
        + np.asarray(target_position, dtype=np.float64)
    )
    return tuple(float(value) for value in world_center)


def camera_link_orientation(
    camera_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Orient the Gazebo camera link's +X axis toward the target."""
    direction = np.asarray(target_position, dtype=np.float64) - np.asarray(camera_position, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("Camera and target positions must be finite XYZ values")
    horizontal = math.hypot(float(direction[0]), float(direction[1]))
    if horizontal < 1e-6 or np.linalg.norm(direction) < 1e-6:
        raise ValueError("Camera cannot look at a target at the same horizontal position")
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    pitch = math.atan2(-float(direction[2]), horizontal)
    rotation = _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, pitch, yaw))[:3, :3]
    return _quaternion_xyzw_from_matrix(rotation)


def camera_link_orientation_for_target_pixel(
    camera_position: tuple[float, float, float],
    target_center: tuple[float, float, float],
    target_x_fraction: float,
    target_y_fraction: float,
) -> tuple[float, float, float, float]:
    """Aim so the suitcase center falls at a requested normalized image point."""
    if not all(math.isfinite(value) and 0.0 < value < 1.0
               for value in (target_x_fraction, target_y_fraction)):
        raise ValueError("Target image fractions must be finite and inside (0, 1)")

    centered_link_q = camera_link_orientation(camera_position, target_center)
    centered_optical_rotation = (
        quaternion_matrix_xyzw(centered_link_q) @ CAMERA_LINK_TO_OPTICAL_ROTATION
    )
    focal = CAMERA_WIDTH / (2.0 * math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))
    ray = np.asarray((
        (target_x_fraction * CAMERA_WIDTH - CAMERA_WIDTH / 2.0) / focal,
        (target_y_fraction * CAMERA_HEIGHT - CAMERA_HEIGHT / 2.0) / focal,
        1.0,
    ), dtype=np.float64)
    ray /= np.linalg.norm(ray)
    cross = np.cross(ray, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    sine = float(np.linalg.norm(cross))
    cosine = float(ray[2])
    if sine <= 1e-12:
        ray_to_forward = np.eye(3, dtype=np.float64)
    else:
        skew = np.asarray((
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ), dtype=np.float64)
        ray_to_forward = (
            np.eye(3, dtype=np.float64) + skew
            + skew @ skew * ((1.0 - cosine) / (sine * sine))
        )
    optical_rotation = centered_optical_rotation @ ray_to_forward
    link_rotation = optical_rotation @ CAMERA_LINK_TO_OPTICAL_ROTATION.T
    return _quaternion_xyzw_from_matrix(link_rotation)


def _stratified_unit_values(count: int, rng: np.random.Generator) -> np.ndarray:
    """Return deterministic seeded values with one sample in every equal stratum."""
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    return (rng.permutation(count) + rng.random(count)) / count


def _sample_azimuth_groups(
    count: int,
    intervals: tuple[tuple[float, float], ...],
    split: str,
    rng: np.random.Generator,
) -> list[tuple[float, str]]:
    if count <= 0:
        return []
    base, remainder = divmod(count, len(intervals))
    group_order = list(rng.permutation(len(intervals)))
    allocations = [base] * len(intervals)
    for index in group_order[:remainder]:
        allocations[int(index)] += 1
    samples: list[tuple[float, str]] = []
    for group_index, ((low, high), group_count) in enumerate(zip(intervals, allocations)):
        values = _stratified_unit_values(group_count, rng)
        scenario_id = f"{split}_azimuth_band_{group_index:02d}_{low:.0f}_{high:.0f}"
        samples.extend((float(low + (high - low) * value), scenario_id) for value in values)
    if samples:
        order = rng.permutation(len(samples))
        samples = [samples[int(index)] for index in order]
    return samples


def dataset_split_counts(sample_count: int) -> dict[str, int]:
    """Return the existing pilot split or the approved 80/10/10 full split."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least 2")
    if sample_count <= MAX_AUTHORIZED_PILOT_SAMPLES:
        val_count = min(sample_count - 1, max(1, int(round(sample_count * 0.20))))
        return {"train": sample_count - val_count, "val": val_count}
    train_count = int(round(sample_count * 0.80))
    val_count = int(round(sample_count * 0.10))
    return {"train": train_count, "val": val_count, "test": sample_count - train_count - val_count}


def capture_candidate_count(sample_count: int) -> int:
    """Reserve one bounded retry candidate per target sample for full data runs."""
    return (
        sample_count * CAPTURE_CANDIDATE_MULTIPLIER
        if sample_count > MAX_AUTHORIZED_PILOT_SAMPLES else sample_count
    )


def split_azimuth_intervals(sample_count: int) -> dict[str, tuple[tuple[float, float], ...]]:
    """Keep pilot coverage and full-dataset split sectors separate."""
    if sample_count <= MAX_AUTHORIZED_PILOT_SAMPLES:
        return {
            "train": TRAIN_AZIMUTH_INTERVALS_DEG,
            "val": PILOT_VALIDATION_AZIMUTH_INTERVALS_DEG,
        }
    return {
        "train": TRAIN_AZIMUTH_INTERVALS_DEG,
        "val": VALIDATION_AZIMUTH_INTERVALS_DEG,
        "test": TEST_AZIMUTH_INTERVALS_DEG,
    }


def pilot_coverage_summary(samples: list[PilotSample]) -> dict:
    """Summarize angular groups, split counts, range, and image-position coverage."""
    distance_edges = (2.4, 2.8, 3.2, 3.6, 4.0)
    height_edges = (0.35, 0.5375, 0.725, 0.9125, 1.10)

    def counts_for(values: list[float], edges: tuple[float, ...]) -> list[dict]:
        return [
            {
                "range": [low, high],
                "count": sum(
                    low <= value <= high if index == len(edges) - 2
                    else low <= value < high
                    for value in values
                ),
            }
            for index, (low, high) in enumerate(zip(edges, edges[1:]))
        ]

    intervals_by_split = split_azimuth_intervals(len(samples))
    azimuth_groups = {
        split: [
            {
                "range_deg": [low, high],
                "count": sum(
                    sample.split == split and low <= sample.azimuth_deg < high
                    for sample in samples
                ),
            }
            for low, high in intervals
        ]
        for split, intervals in intervals_by_split.items()
    }
    distances = [sample.camera_distance_m for sample in samples]
    heights = [sample.camera_height_above_target_m for sample in samples]
    return {
        "sample_count": len(samples),
        "clearance_resampled_count": sum(sample.clearance_resampled for sample in samples),
        "split_counts": {
            split: sum(sample.split == split for sample in samples)
            for split in intervals_by_split
        },
        "azimuth_groups": azimuth_groups,
        "distance_bins_m": counts_for(distances, distance_edges),
        "height_bins_m": counts_for(heights, height_edges),
        "image_target_x_bins_fraction": counts_for(
            [sample.image_target_x_fraction for sample in samples], (0.30, 0.40, 0.50, 0.60, 0.70),
        ),
        "image_target_y_bins_fraction": counts_for(
            [sample.image_target_y_fraction for sample in samples], (0.40, 0.45, 0.50, 0.55, 0.60),
        ),
        "distance_min_max_m": [min(distances), max(distances)] if distances else None,
        "height_min_max_m": [min(heights), max(heights)] if heights else None,
        "split_scenario_overlap": {
            f"{left}_{right}": sorted(
                {sample.scenario_id for sample in samples if sample.split == left}
                & {sample.scenario_id for sample in samples if sample.split == right}
            )
            for index, left in enumerate(intervals_by_split)
            for right in list(intervals_by_split)[index + 1:]
        },
    }


def pilot_samples(
    target_center: tuple[float, float, float],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
    world_path: Path | None = None,
) -> list[PilotSample]:
    """Plan a seeded camera orbit with distinct split sectors and varied framing."""
    center = np.asarray(target_center, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Suitcase visual center must contain three finite values")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")

    rng = np.random.default_rng(seed)
    split_counts = dataset_split_counts(sample_count)
    intervals_by_split = split_azimuth_intervals(sample_count)
    planned: list[PilotSample] = []
    for split, split_count in split_counts.items():
        intervals = intervals_by_split[split]
        azimuth_groups = _sample_azimuth_groups(split_count, intervals, split, rng)
        distances = CAMERA_DISTANCE_RANGE_M[0] + (
            CAMERA_DISTANCE_RANGE_M[1] - CAMERA_DISTANCE_RANGE_M[0]
        ) * _stratified_unit_values(split_count, rng)
        heights = CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[0] + (
            CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[1] - CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[0]
        ) * _stratified_unit_values(split_count, rng)
        target_x = IMAGE_TARGET_X_FRACTION_RANGE[0] + (
            IMAGE_TARGET_X_FRACTION_RANGE[1] - IMAGE_TARGET_X_FRACTION_RANGE[0]
        ) * _stratified_unit_values(split_count, rng)
        target_y = IMAGE_TARGET_Y_FRACTION_RANGE[0] + (
            IMAGE_TARGET_Y_FRACTION_RANGE[1] - IMAGE_TARGET_Y_FRACTION_RANGE[0]
        ) * _stratified_unit_values(split_count, rng)

        for (azimuth, scenario_id), distance, height, x_fraction, y_fraction in zip(
            azimuth_groups, distances, heights, target_x, target_y,
        ):
            angle = math.radians(azimuth)
            camera_position = (
                float(center[0] + distance * math.cos(angle)),
                float(center[1] + distance * math.sin(angle)),
                float(center[2] + height),
            )
            orientation = camera_link_orientation_for_target_pixel(
                camera_position, tuple(center), float(x_fraction), float(y_fraction),
            )
            rotation = quaternion_matrix_xyzw(orientation)
            yaw = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))) % 360.0
            pitch = math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0))))
            planned.append(PilotSample(
                sample_id="",
                split=split,
                scenario_id=scenario_id,
                camera_x_m=camera_position[0],
                camera_y_m=camera_position[1],
                camera_z_m=camera_position[2],
                camera_yaw_deg=yaw,
                camera_pitch_deg=pitch,
                azimuth_deg=float(azimuth),
                camera_distance_m=float(distance),
                camera_height_above_target_m=float(height),
                random_seed=seed,
                image_target_x_fraction=float(x_fraction),
                image_target_y_fraction=float(y_fraction),
                camera_link_quaternion_xyzw=orientation,
            ))

    if world_path is not None:
        obstacles = production_collision_boxes(Path(world_path))

        def has_clearance(sample: PilotSample) -> bool:
            point = np.asarray(
                (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m), dtype=np.float64,
            )
            for obstacle in obstacles:
                obstacle_center = np.asarray(
                    (obstacle["center_x"], obstacle["center_y"], obstacle["center_z"]),
                    dtype=np.float64,
                )
                obstacle_half = np.asarray(
                    (obstacle["half_x"], obstacle["half_y"], obstacle["half_z"]),
                    dtype=np.float64,
                )
                outside = np.maximum(np.abs(point - obstacle_center) - obstacle_half, 0.0)
                if float(np.linalg.norm(outside)) <= CAMERA_CLEARANCE_RADIUS_M:
                    return False
            return True

        for sample_index, sample in enumerate(planned):
            if has_clearance(sample):
                continue
            intervals = intervals_by_split[sample.split]
            group_index = next(
                (
                    index for index, (low, high) in enumerate(intervals)
                    if sample.scenario_id == (
                        f"{sample.split}_azimuth_band_{index:02d}_{low:.0f}_{high:.0f}"
                    )
                ),
                None,
            )
            if group_index is None:
                raise ValueError(f"Cannot recover azimuth group for {sample.sample_id or sample.scenario_id}")
            low, high = intervals[group_index]
            for _ in range(1000):
                azimuth = float(rng.uniform(low, high))
                distance = float(rng.uniform(*CAMERA_DISTANCE_RANGE_M))
                height = float(rng.uniform(*CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M))
                x_fraction = float(rng.uniform(*IMAGE_TARGET_X_FRACTION_RANGE))
                y_fraction = float(rng.uniform(*IMAGE_TARGET_Y_FRACTION_RANGE))
                angle = math.radians(azimuth)
                camera_position = (
                    float(center[0] + distance * math.cos(angle)),
                    float(center[1] + distance * math.sin(angle)),
                    float(center[2] + height),
                )
                orientation = camera_link_orientation_for_target_pixel(
                    camera_position, tuple(center), x_fraction, y_fraction,
                )
                rotation = quaternion_matrix_xyzw(orientation)
                yaw = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))) % 360.0
                pitch = math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0))))
                candidate = replace(
                    sample,
                    camera_x_m=camera_position[0],
                    camera_y_m=camera_position[1],
                    camera_z_m=camera_position[2],
                    camera_yaw_deg=yaw,
                    camera_pitch_deg=pitch,
                    azimuth_deg=azimuth,
                    camera_distance_m=distance,
                    camera_height_above_target_m=height,
                    image_target_x_fraction=x_fraction,
                    image_target_y_fraction=y_fraction,
                    camera_link_quaternion_xyzw=orientation,
                    clearance_resampled=True,
                )
                if has_clearance(candidate):
                    planned[sample_index] = candidate
                    break
            else:
                raise ValueError(
                    f"No collision-free dataset-camera pose found in azimuth group "
                    f"{sample.scenario_id} after 1000 deterministic candidates"
                )

    order = rng.permutation(len(planned))
    return [
        replace(planned[int(index)], sample_id=f"seed{seed}_view_{position:04d}")
        for position, index in enumerate(order)
    ]


def load_collada_geometry(mesh_path: Path) -> MeshGeometry:
    """Read the current model's untransformed Z-up COLLADA triangle geometry."""
    root = ET.parse(mesh_path).getroot()
    asset = root.find("c:asset", NS)
    if asset is None:
        raise ValueError("COLLADA asset metadata is missing")
    up_axis = asset.findtext("c:up_axis", default="", namespaces=NS).strip()
    if up_axis != "Z_UP":
        raise ValueError(f"Expected the approved Z_UP suitcase mesh, got {up_axis!r}")
    unit_element = asset.find("c:unit", NS)
    meter_scale = float(unit_element.get("meter", "1")) if unit_element is not None else 1.0

    # The approved DAE has identity-only scene nodes. Refuse an asset change
    # that would silently invalidate the mesh-to-link coordinate transform.
    for node in root.findall(".//c:library_visual_scenes/c:visual_scene//c:node", NS):
        for transform_tag in ("translate", "rotate", "scale", "matrix"):
            if node.find(f"c:{transform_tag}", NS) is not None:
                raise ValueError(
                    f"COLLADA node {node.get('id')!r} has a transform; update the parser first"
                )

    scene_geometries = {
        instance.get("url", "").lstrip("#")
        for instance in root.findall(
            ".//c:library_visual_scenes/c:visual_scene//c:instance_geometry", NS
        )
    }
    vertices_out: list[np.ndarray] = []
    triangles_out: list[np.ndarray] = []
    geometry_count = 0
    triangle_count = 0
    global_vertex_offset = 0

    for geometry in root.findall(".//c:library_geometries/c:geometry", NS):
        geometry_id = geometry.get("id", "")
        if geometry_id not in scene_geometries:
            continue
        mesh = geometry.find("c:mesh", NS)
        if mesh is None:
            continue
        sources: dict[str, np.ndarray] = {}
        for source in mesh.findall("c:source", NS):
            array = source.find("c:float_array", NS)
            if array is not None and array.text:
                accessor = source.find("c:technique_common/c:accessor", NS)
                stride = int(accessor.get("stride", "3")) if accessor is not None else 3
                values = np.fromstring(array.text, sep=" ", dtype=np.float64)
                if stride < 1 or values.size % stride:
                    raise ValueError(f"Unsupported COLLADA source layout in {geometry_id}")
                sources[source.get("id", "")] = values.reshape(-1, stride)

        vertices_element = mesh.find("c:vertices", NS)
        position_input = (
            vertices_element.find("c:input[@semantic='POSITION']", NS)
            if vertices_element is not None else None
        )
        if position_input is None:
            raise ValueError(f"COLLADA geometry {geometry_id} has no POSITION source")
        position_source_id = position_input.get("source", "").lstrip("#")
        if position_source_id not in sources:
            raise ValueError(f"COLLADA POSITION source is missing in {geometry_id}")
        vertices = sources[position_source_id] * meter_scale
        vertices_out.append(vertices)
        geometry_count += 1

        for triangles in mesh.findall("c:triangles", NS):
            inputs = triangles.findall("c:input", NS)
            vertex_input = next((item for item in inputs if item.get("semantic") == "VERTEX"), None)
            if vertex_input is None:
                raise ValueError(f"Triangles in {geometry_id} do not reference VERTEX")
            vertex_index_offset = int(vertex_input.get("offset", "0"))
            stride = 1 + max(int(item.get("offset", "0")) for item in inputs)
            index_text = triangles.findtext("c:p", default="", namespaces=NS)
            indices = np.fromstring(index_text, sep=" ", dtype=np.int64)
            if indices.size % (3 * stride):
                raise ValueError(f"Malformed triangle indices in {geometry_id}")
            position_indices = indices.reshape(-1, stride)[:, vertex_index_offset]
            triangle_indices = position_indices.reshape(-1, 3)
            triangles_out.append(triangle_indices + global_vertex_offset)
            triangle_count += triangle_indices.shape[0]

        if mesh.find("c:polylist", NS) is not None or mesh.find("c:polygons", NS) is not None:
            raise ValueError("The approved suitcase mesh uses triangles; polygon lists are unsupported")
        global_vertex_offset += vertices.shape[0]

    if not vertices_out or not triangles_out:
        raise ValueError("No rendered suitcase geometry was found in the COLLADA scene")
    vertices = np.concatenate(vertices_out, axis=0)
    triangles = np.concatenate(triangles_out, axis=0)
    bounds_min = tuple(float(value) for value in vertices.min(axis=0))
    bounds_max = tuple(float(value) for value in vertices.max(axis=0))
    return MeshGeometry(
        vertices=vertices,
        triangles=triangles,
        local_bounds_min=bounds_min,
        local_bounds_max=bounds_max,
        geometry_count=geometry_count,
        triangle_count=triangle_count,
    )


def quaternion_matrix_xyzw(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Return a 3x3 rotation matrix from a ROS-order x,y,z,w quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion is invalid or has zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _matrix_from_xyz_rpy(
    xyz: tuple[float, float, float], rpy: tuple[float, float, float],
) -> np.ndarray:
    values = np.asarray((*xyz, *rpy), dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Pose must contain six finite values")
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = (x, y, z)
    return transform


def _matrix_from_pose(
    position: tuple[float, float, float], quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    values = np.asarray(position, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Pose position must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_matrix_xyzw(quaternion_xyzw)
    transform[:3, 3] = values
    return transform


def _quaternion_xyzw_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Rotation matrix must be finite and 3x3")
    if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6)):
        raise ValueError("Rotation matrix is not a valid rigid transform")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Rotation matrix produced an invalid quaternion")
    return tuple(float(value) for value in quaternion / norm)










def pose_points_to_camera(
    local_points: np.ndarray,
    object_position: tuple[float, float, float],
    object_quaternion_xyzw: tuple[float, float, float, float],
    visual_offset: tuple[float, float, float],
    camera_position: tuple[float, float, float],
    camera_quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """Transform model-link points through world coordinates into optical frame."""
    object_rotation = quaternion_matrix_xyzw(object_quaternion_xyzw)
    camera_rotation = quaternion_matrix_xyzw(camera_quaternion_xyzw)
    points_link = local_points + np.asarray(visual_offset, dtype=np.float64)
    points_world = points_link @ object_rotation.T + np.asarray(object_position)
    return (points_world - np.asarray(camera_position)) @ camera_rotation


def project_camera_points(
    camera_points: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    distortion: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project optical-frame XYZ points with pinhole CameraInfo intrinsics."""
    fx, fy, cx, cy = intrinsics
    z = camera_points[:, 2]
    if np.any(z <= 0.0):
        raise CaptureRejected("MESH_BEHIND_CAMERA")
    if distortion and any(abs(value) > 1e-12 for value in distortion):
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        image_points, _ = cv2.projectPoints(
            camera_points.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            np.asarray(distortion, dtype=np.float64),
        )
        image_points = image_points.reshape(-1, 2)
        return image_points[:, 0], image_points[:, 1], z
    return fx * camera_points[:, 0] / z + cx, fy * camera_points[:, 1] / z + cy, z


def projected_bbox(
    mesh_vertices_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
    edge_margin_px: int = 2,
    distortion: tuple[float, ...] | None = None,
) -> tuple[float, float, float, float]:
    """Return an unclipped geometry bbox; reject edge-clipped suitcase poses."""
    u, v, _ = project_camera_points(mesh_vertices_camera, intrinsics, distortion)
    x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    if x1 < edge_margin_px or y1 < edge_margin_px or x2 >= width - edge_margin_px or y2 >= height - edge_margin_px:
        raise CaptureRejected("SUITCASE_CLIPPED_BY_IMAGE_EDGE")
    if x2 <= x1 or y2 <= y1:
        raise CaptureRejected("DEGENERATE_GEOMETRY_BBOX")
    return x1, y1, x2, y2


def yolo_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    encoding = encoding.upper()
    if encoding in {"32FC1", "32FC"}:
        return depth.astype(np.float32, copy=False)
    if encoding in {"16UC1", "MONO16", "16SC1"}:
        return depth.astype(np.float32) * 0.001
    raise ValueError(f"Unsupported depth image encoding: {encoding}")


def rasterize_front_surface_depth(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
    distortion: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize mesh triangles into a nearest-surface perspective depth map."""
    vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices_camera.ndim != 2 or vertices_camera.shape[1] != 3:
        raise ValueError("Camera-space mesh vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.size == 0:
        raise ValueError("Mesh triangle indices must have shape (N, 3)")
    if (triangles.min() < 0 or triangles.max() >= vertices_camera.shape[0]
            or width <= 0 or height <= 0):
        raise ValueError("Mesh topology or image dimensions are invalid")
    if not np.all(np.isfinite(vertices_camera)) or np.any(vertices_camera[:, 2] <= 0.0):
        raise CaptureRejected("MESH_BEHIND_CAMERA")
    if distortion and any(abs(value) > 1e-12 for value in distortion):
        raise CaptureRejected("UNSUPPORTED_NONZERO_CAMERA_DISTORTION")

    fx, fy, cx, cy = intrinsics
    z = vertices_camera[:, 2]
    u = fx * vertices_camera[:, 0] / z + cx
    v = fy * vertices_camera[:, 1] / z + cy
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise CaptureRejected("INVALID_PROJECTED_MESH_GEOMETRY")
    inverse_z = 1.0 / z
    expected = np.full((height, width), np.inf, dtype=np.float32)
    triangle_u = u[triangles]
    triangle_v = v[triangles]
    denominators = (
        (triangle_v[:, 1] - triangle_v[:, 2]) * (triangle_u[:, 0] - triangle_u[:, 2])
        + (triangle_u[:, 2] - triangle_u[:, 1]) * (triangle_v[:, 0] - triangle_v[:, 2])
    )
    degenerate = np.abs(denominators) < 1e-10
    xmin = np.maximum(0, np.ceil(triangle_u.min(axis=1)))
    xmax = np.minimum(width - 1, np.floor(triangle_u.max(axis=1)))
    ymin = np.maximum(0, np.ceil(triangle_v.min(axis=1)))
    ymax = np.minimum(height - 1, np.floor(triangle_v.max(axis=1)))
    candidates = np.flatnonzero(
        ~degenerate & (xmin <= xmax) & (ymin <= ymax)
    )
    rasterized_triangles = 0
    degenerate_triangles = int(np.count_nonzero(degenerate))

    for triangle_index in candidates:
        triangle = triangles[triangle_index]
        x0, x1, x2 = triangle_u[triangle_index]
        y0, y1, y2 = triangle_v[triangle_index]
        z0i, z1i, z2i = inverse_z[triangle]
        denominator = denominators[triangle_index]
        x_min = int(xmin[triangle_index])
        x_max = int(xmax[triangle_index])
        y_min = int(ymin[triangle_index])
        y_max = int(ymax[triangle_index])

        grid_x = np.arange(x_min, x_max + 1, dtype=np.float64)[None, :]
        grid_y = np.arange(y_min, y_max + 1, dtype=np.float64)[:, None]
        a_x, a_y = (y1 - y2) / denominator, (x2 - x1) / denominator
        b_x, b_y = (y2 - y0) / denominator, (x0 - x2) / denominator
        a = a_x * grid_x + a_y * grid_y - (a_x * x2 + a_y * y2)
        b = b_x * grid_x + b_y * grid_y - (b_x * x2 + b_y * y2)
        c = 1.0 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        if not np.any(inside):
            continue

        reciprocal_depth = a * z0i + b * z1i + c * z2i
        triangle_depth = np.full(reciprocal_depth.shape, np.inf, dtype=np.float32)
        triangle_depth[inside] = (1.0 / reciprocal_depth[inside]).astype(np.float32)
        image_region = expected[y_min:y_max + 1, x_min:x_max + 1]
        np.minimum(image_region, triangle_depth, out=image_region)
        rasterized_triangles += 1

    if not np.isfinite(expected).any():
        raise CaptureRejected("NO_RASTERIZED_MESH_PIXELS")
    return expected, {
        "rasterized_triangles": rasterized_triangles,
        "degenerate_triangles": degenerate_triangles,
    }


def depth_visibility_products(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_m: np.ndarray,
    distortion: tuple[float, ...] | None = None,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Compare the rasterized nearest mesh surface with aligned measured depth."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError("Measured depth must be a two-dimensional image")
    height, width = depth_m.shape
    expected, raster_stats = rasterize_front_surface_depth(
        vertices_camera, triangles, intrinsics, width, height, distortion,
    )
    projected = np.isfinite(expected)
    valid = projected & np.isfinite(depth_m) & (depth_m > 0.05)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < MIN_DEPTH_MATCHES:
        raise CaptureRejected(f"INSUFFICIENT_DEPTH_EVIDENCE_{valid_count}")
    tolerance = np.maximum(0.06, expected * 0.03)
    matched = valid & (np.abs(depth_m - expected) <= tolerance)
    match_count = int(np.count_nonzero(matched))
    fraction = match_count / valid_count
    metrics = {
        "projected_mesh_pixels": int(np.count_nonzero(projected)),
        **raster_stats,
        "valid_depth_pixels": valid_count,
        "matching_depth_pixels": match_count,
        "visible_fraction": float(fraction),
    }
    return metrics, matched, expected


def depth_visibility(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_m: np.ndarray,
    distortion: tuple[float, ...] | None = None,
) -> dict[str, float | int]:
    """Return visibility metrics while preserving the original offline API."""
    return depth_visibility_products(
        vertices_camera, triangles, intrinsics, depth_m, distortion,
    )[0]


def visible_bbox_from_mask(
    visible_mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Bound depth-confirmed visible suitcase pixels using XYXY edge coordinates."""
    visible_mask = np.asarray(visible_mask, dtype=bool)
    if visible_mask.shape != (height, width):
        raise ValueError("Visible-object mask dimensions do not match the RGB image")
    ys, xs = np.nonzero(visible_mask)
    if xs.size < MIN_DEPTH_MATCHES:
        raise CaptureRejected(f"INSUFFICIENT_VISIBLE_LABEL_PIXELS_{xs.size}")
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise CaptureRejected("INVALID_VISIBLE_OBJECT_BBOX")
    return float(x1), float(y1), float(x2), float(y2)


def depth_diagnostic_preview(
    depth_m: np.ndarray,
    expected_depth_m: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    """Create a fixed-range depth view with visible target pixels and mesh outline."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    expected_depth_m = np.asarray(expected_depth_m, dtype=np.float32)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    if depth_m.ndim != 2 or expected_depth_m.shape != depth_m.shape or visible_mask.shape != depth_m.shape:
        raise ValueError("Depth diagnostic arrays must share the same two-dimensional shape")
    valid = np.isfinite(depth_m) & (depth_m > 0.05)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    clipped = np.clip(depth_m[valid], 0.05, 20.0)
    normalized[valid] = np.rint(255.0 * (1.0 - (clipped - 0.05) / 19.95)).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    green = np.zeros_like(preview)
    green[:, :] = (0, 255, 0)
    preview[visible_mask] = cv2.addWeighted(preview, 0.35, green, 0.65, 0.0)[visible_mask]
    projected = np.isfinite(expected_depth_m).astype(np.uint8) * 255
    contours, _ = cv2.findContours(projected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 165, 255), 1)
    return preview


def check_scene_files(world_path: Path, mesh_path: Path) -> tuple[str, str]:
    world_hash, mesh_hash = sha256_file(world_path), sha256_file(mesh_path)
    scene_variant_for_world(world_path)
    if mesh_hash != MESH_SHA256:
        raise ValueError(f"Suitcase mesh is not the approved hard_trolley_red asset: SHA-256 {mesh_hash}")
    for relative_path, expected_hash in FURNITURE_SHA256.items():
        furniture_path = ASSET_SHARE / "assets" / relative_path
        actual_hash = sha256_file(furniture_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Approved production furniture changed: {relative_path} SHA-256 {actual_hash}")
    # Both backgrounds use the same hash-pinned target model block from the
    # approved production world. In the empty runtime wrapper it is static.
    world_xml = ET.parse(DEFAULT_WORLD).getroot()
    target = world_xml.find(".//model[@name='suitcase_target']")
    if target is None:
        raise ValueError("Approved world does not contain suitcase_target")
    mesh_uri = target.findtext(".//visual/geometry/mesh/uri", default="")
    if "hard_trolley_red/meshes/suitcase.dae" not in mesh_uri:
        raise ValueError(f"Unexpected production suitcase mesh reference: {mesh_uri!r}")
    return world_hash, mesh_hash


def _pose6(element: ET.Element | None) -> tuple[float, float, float, float, float, float]:
    if element is None or not element.text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = tuple(float(value) for value in element.text.split())
    if len(values) != 6:
        raise ValueError(f"Expected an SDF 6-DOF pose, got {element.text!r}")
    return values


def _compose_pose2d(parent: tuple[float, float, float, float, float, float],
                    child: tuple[float, float, float, float, float, float]):
    px, py, pz, pr, pp, pyaw = parent
    cx, cy, cz, cr, cp, cyaw = child
    cosine, sine = math.cos(pyaw), math.sin(pyaw)
    return (
        px + cosine * cx - sine * cy,
        py + sine * cx + cosine * cy,
        pz + cz,
        pr + cr,
        pp + cp,
        pyaw + cyaw,
    )


def _collision_boxes(model_element: ET.Element, base_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> list[dict]:
    boxes: list[dict] = []
    model_pose = _compose_pose2d(base_pose, _pose6(model_element.find("pose")))
    for link in model_element.findall("link"):
        link_pose = _compose_pose2d(model_pose, _pose6(link.find("pose")))
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None:
                raise ValueError(f"Collision {collision.get('name')} has no geometry")
            collision_pose = _compose_pose2d(link_pose, _pose6(collision.find("pose")))
            shape = next(iter(geometry), None)
            if shape is None:
                raise ValueError(f"Collision {collision.get('name')} has no shape")
            if shape.tag == "plane":
                # The camera orbit remains above this support surface; it is
                # not a finite obstacle for the point-clearance check.
                continue
            if shape.tag == "box":
                size = tuple(float(value) for value in shape.findtext("size", "").split())
                if len(size) != 3:
                    raise ValueError(f"Unsupported box size on {collision.get('name')}")
                local_hx, local_hy, half_z = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
            elif shape.tag == "cylinder":
                radius = float(shape.findtext("radius", "0"))
                length = float(shape.findtext("length", "0"))
                local_hx = local_hy = radius
                half_z = length / 2.0
            else:
                raise ValueError(f"Unsupported collision shape {shape.tag!r} in approved world")
            cx, cy, cz, _, _, yaw = collision_pose
            half_x = abs(math.cos(yaw)) * local_hx + abs(math.sin(yaw)) * local_hy
            half_y = abs(math.sin(yaw)) * local_hx + abs(math.cos(yaw)) * local_hy
            # Ground plane contact is intentional; the suitcase collision body
            # must be clear of all other collidable geometry above floor level.
            if cz + half_z <= 0.01:
                continue
            boxes.append({
                "name": collision.get("name", "unnamed_collision"),
                "center_x": cx,
                "center_y": cy,
                "center_z": cz,
                "half_x": half_x,
                "half_y": half_y,
                "half_z": half_z,
            })
    return boxes


def _asset_model_sdf(model_name: str) -> Path:
    candidates = (
        ASSET_SHARE / "assets" / model_name / "model.sdf",
        SIMULATION_SHARE / "models" / model_name / "model.sdf",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"Cannot resolve approved-world model {model_name!r} to a local SDF")


def production_collision_boxes(world_path: Path) -> list[dict]:
    """Resolve finite collision bounds used for the selected scene's camera path."""
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("SDF has no world element")
    obstacles: list[dict] = []
    for model in world.findall("model"):
        if model.get("name") != TARGET_ENTITY:
            obstacles.extend(_collision_boxes(model))
    for include in world.findall("include"):
        uri = include.findtext("uri", default="")
        if not uri.startswith("model://"):
            raise ValueError(f"Unexpected production-world include URI: {uri!r}")
        model_name = uri.removeprefix("model://")
        model_root = ET.parse(_asset_model_sdf(model_name)).getroot()
        model = model_root.find("model")
        if model is None:
            raise ValueError(f"Included asset {model_name!r} has no SDF model")
        include_name = include.findtext("name", default=model_name)
        include_pose = _pose6(include.find("pose"))
        obstacles.extend(
            {**box, "model": include_name}
            for box in _collision_boxes(model, include_pose)
        )
    if not obstacles and scene_variant_for_world(world_path) != "empty":
        raise ValueError("No production-world collision geometry was checked")
    return obstacles


def validate_camera_clearance(
    world_path: Path,
    samples: list[PilotSample],
    clearance_radius_m: float = CAMERA_CLEARANCE_RADIUS_M,
) -> list[dict]:
    """Reject camera poses that overlap a production-world collision AABB."""
    if not math.isfinite(clearance_radius_m) or clearance_radius_m <= 0.0:
        raise ValueError("Camera clearance radius must be positive and finite")
    obstacles = production_collision_boxes(world_path)
    for sample in samples:
        point = np.asarray(
            (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m), dtype=np.float64
        )
        for obstacle in obstacles:
            center = np.asarray(
                (obstacle["center_x"], obstacle["center_y"], obstacle["center_z"]),
                dtype=np.float64,
            )
            half = np.asarray(
                (obstacle["half_x"], obstacle["half_y"], obstacle["half_z"]),
                dtype=np.float64,
            )
            outside = np.maximum(np.abs(point - center) - half, 0.0)
            if float(np.linalg.norm(outside)) <= clearance_radius_m:
                raise ValueError(
                    f"Camera pose {sample.sample_id} is within {clearance_radius_m:.2f} m of "
                    f"{obstacle.get('model', 'world')}::{obstacle['name']}"
                )
    return obstacles




def _unit_quaternion_difference(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    norm_a, norm_b = float(np.linalg.norm(qa)), float(np.linalg.norm(qb))
    if (not math.isfinite(norm_a) or not math.isfinite(norm_b)
            or norm_a < 1e-12 or norm_b < 1e-12):
        raise ValueError("Cannot compare invalid or zero-length quaternions")
    qa /= norm_a
    qb /= norm_b
    dot = min(1.0, abs(float(np.dot(qa, qb))))
    return 2.0 * math.acos(dot)


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def entity_pose(response) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pose = response.state.pose
    position = (pose.position.x, pose.position.y, pose.position.z)
    orientation = (
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
    )
    _matrix_from_pose(position, orientation)
    return position, orientation


def camera_optical_pose(
    camera_link_position: tuple[float, float, float],
    camera_link_quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert the Gazebo camera link pose to its ROS optical frame pose."""
    link_rotation = quaternion_matrix_xyzw(camera_link_quaternion_xyzw)
    optical_rotation = link_rotation @ CAMERA_LINK_TO_OPTICAL_ROTATION
    optical_quaternion = _quaternion_xyzw_from_matrix(optical_rotation)
    return camera_link_position, optical_quaternion


def validate_camera_sample_pose(
    sample: PilotSample,
    camera_position: tuple[float, float, float],
    camera_quaternion_xyzw: tuple[float, float, float, float],
    target_center: tuple[float, float, float],
) -> tuple[float, float]:
    """Check actual camera range and look direction against its requested orbit sample."""
    expected_position = np.asarray(
        (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m), dtype=np.float64
    )
    actual_position = np.asarray(camera_position, dtype=np.float64)
    if actual_position.shape != (3,) or not np.all(np.isfinite(actual_position)):
        raise ValueError("Gazebo returned an invalid dataset-camera position")
    position_delta = float(np.linalg.norm(actual_position - expected_position))
    target = np.asarray(target_center, dtype=np.float64)
    horizontal_distance = float(np.linalg.norm((actual_position - target)[:2]))
    if abs(horizontal_distance - sample.camera_distance_m) > 0.05:
        raise ValueError(
            f"Dataset camera range differs from the planned {sample.camera_distance_m:.2f} m"
        )
    expected_orientation = sample.camera_link_quaternion_xyzw
    orientation_delta = _unit_quaternion_difference(
        camera_quaternion_xyzw, expected_orientation,
    )
    if position_delta > 0.03 or orientation_delta > math.radians(2.0):
        raise ValueError(
            f"Dataset camera did not reach the planned pose "
            f"(position delta {position_delta:.3f} m, orientation delta "
            f"{math.degrees(orientation_delta):.2f} deg)"
        )
    _, optical_quaternion = camera_optical_pose(
        tuple(actual_position), camera_quaternion_xyzw,
    )
    target_camera = (target - actual_position) @ quaternion_matrix_xyzw(optical_quaternion)
    if not np.all(np.isfinite(target_camera)) or target_camera[2] <= 0.0:
        raise ValueError("Suitcase center is behind the dataset camera")
    focal = CAMERA_WIDTH / (2.0 * math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))
    projected_x = focal * target_camera[0] / target_camera[2] + CAMERA_WIDTH / 2.0
    projected_y = focal * target_camera[1] / target_camera[2] + CAMERA_HEIGHT / 2.0
    expected_x = sample.image_target_x_fraction * CAMERA_WIDTH
    expected_y = sample.image_target_y_fraction * CAMERA_HEIGHT
    if (abs(projected_x - expected_x) > 2.0 or abs(projected_y - expected_y) > 2.0):
        raise ValueError(
            f"Suitcase center projects to ({projected_x:.1f}, {projected_y:.1f}) px, "
            f"outside planned point ({expected_x:.1f}, {expected_y:.1f}) px"
        )
    return horizontal_distance, position_delta


def projected_pilot_boxes(
    samples: list[PilotSample],
    target_position: tuple[float, float, float],
    target_quaternion_xyzw: tuple[float, float, float, float],
    target_center: tuple[float, float, float],
    mesh: MeshGeometry,
) -> dict[str, tuple[float, float, float, float]]:
    """Check planned views fit the 640x360 camera using its approved FOV."""
    focal = CAMERA_WIDTH / (2.0 * math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))
    intrinsics = (focal, focal, CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0)
    boxes: dict[str, tuple[float, float, float, float]] = {}
    vertices = mesh.vertices.copy()
    vertices[:, 2] -= 0.05
    for sample in samples:
        camera_position = (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m)
        link_quaternion = sample.camera_link_quaternion_xyzw
        optical_position, optical_quaternion = camera_optical_pose(
            camera_position, link_quaternion,
        )
        points_camera = pose_points_to_camera(
            vertices,
            target_position,
            target_quaternion_xyzw,
            (0.0, 0.0, 0.0),
            optical_position,
            optical_quaternion,
        )
        boxes[sample.sample_id] = projected_bbox(
            points_camera, intrinsics, CAMERA_WIDTH, CAMERA_HEIGHT,
        )
    return boxes


def populate_set_entity_state_request(request, entity_name: str, sample: PilotSample):
    """Move the independent dataset camera in the world frame."""
    request.state.name = entity_name
    request.state.reference_frame = "world"
    request.state.pose.position.x = sample.camera_x_m
    request.state.pose.position.y = sample.camera_y_m
    request.state.pose.position.z = sample.camera_z_m
    quaternion = sample.camera_link_quaternion_xyzw
    (request.state.pose.orientation.x, request.state.pose.orientation.y,
     request.state.pose.orientation.z, request.state.pose.orientation.w) = quaternion
    for vector in (request.state.twist.linear, request.state.twist.angular):
        vector.x = vector.y = vector.z = 0.0
    return request


def rgbd_timestamps_are_fresh(
    color_stamp_s: float,
    depth_stamp_s: float,
    pose_stamp_s: float,
    max_sync_delta_s: float,
) -> bool:
    return (
        color_stamp_s > pose_stamp_s
        and depth_stamp_s > pose_stamp_s
        and abs(color_stamp_s - depth_stamp_s) <= max_sync_delta_s
    )


def capture_pose_delta(
    before_position: tuple[float, float, float],
    before_orientation: tuple[float, float, float, float],
    after_position: tuple[float, float, float],
    after_orientation: tuple[float, float, float, float],
) -> tuple[float, float]:
    position_delta = float(np.linalg.norm(np.asarray(after_position) - np.asarray(before_position)))
    orientation_delta = _unit_quaternion_difference(before_orientation, after_orientation)
    return position_delta, orientation_delta


def run_capture(
    args: argparse.Namespace,
    mesh: MeshGeometry,
    world_hash: str,
    mesh_hash: str,
) -> Path:
    """Capture the authorized accepted target using only the independent dataset camera."""
    if (not math.isfinite(args.max_camera_stamp_delta_s)
            or args.max_camera_stamp_delta_s <= 0.0):
        raise ValueError("--max-camera-stamp-delta-s must be positive and finite")
    if (not math.isfinite(args.delivery_latency_warning_s)
            or args.delivery_latency_warning_s < 0.0):
        raise ValueError("--delivery-latency-warning-s must be nonnegative and finite")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    desired_split_counts = dataset_split_counts(args.sample_count)
    candidate_pool_count = capture_candidate_count(args.sample_count)
    candidate_split_counts = dataset_split_counts(candidate_pool_count)
    for relative in (
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test",
        "depth/train", "depth/val", "depth/test", "preview/samples", "preview/depth",
        "metadata/diagnostics",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    run_token = getattr(args, "run_token", uuid.uuid4().hex)
    progress_path = output_dir / "metadata" / "capture_progress.json"
    initial_progress_manifest = {
        "schema": "suitcase-pilot-v3-independent-camera",
        "status": "RUNNING",
        "run_token": run_token,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "world_path": str(Path(args.world).resolve()),
        "scene_variant": scene_variant_for_world(Path(args.world).resolve()),
        "mesh_path": str(Path(args.mesh).resolve()),
        "requested_sample_count": args.sample_count,
        "requested_split_counts": desired_split_counts,
        "candidate_pool_count": candidate_pool_count,
        "candidate_pool_split_counts": candidate_split_counts,
        "max_frame_acquisition_retries_per_candidate": MAX_FRAME_ACQUISITION_RETRIES,
        "random_seed": args.seed,
        "pilot_plan": [],
    }
    write_capture_progress(progress_path, run_token, initial_progress_manifest, [], [], None)

    camera_model_path = SIMULATION_SHARE / "models/dataset_camera/model.sdf"
    static_target_position, static_target_q = target_model_pose(Path(args.world).resolve())
    static_target_center = suitcase_visual_center(static_target_position, static_target_q, mesh)
    planned_samples = pilot_samples(
        static_target_center, candidate_pool_count, args.seed, Path(args.world).resolve(),
    )
    planned_obstacles = validate_camera_clearance(Path(args.world).resolve(), planned_samples)
    planned_boxes = projected_pilot_boxes(
        planned_samples, static_target_position, static_target_q, static_target_center, mesh,
    )
    camera_model_hash = sha256_file(camera_model_path)
    camera_configuration = {
        "model_path": str(camera_model_path),
        "model_sha256": camera_model_hash,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "horizontal_fov_rad": CAMERA_HORIZONTAL_FOV_RAD,
        "update_rate_hz": CAMERA_UPDATE_RATE_HZ,
        "expected_rgb_encoding": "rgb8",
        "expected_depth_encoding": "32FC1",
        "optical_frame": OPTICAL_FRAME_NAME,
        "horizontal_distance_range_m": CAMERA_DISTANCE_RANGE_M,
        "height_above_target_center_range_m": CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M,
    }
    configuration_sha = canonical_sha256({
        "scene_variant": scene_variant_for_world(Path(args.world).resolve()),
        "capture_world_sha256": world_hash,
        "production_world_sha256": WORLD_SHA256,
        "target_model_source_world_sha256": WORLD_SHA256,
        "suitcase_mesh_sha256": mesh_hash,
        "camera_configuration": camera_configuration,
        "sample_count": args.sample_count,
        "random_seed": args.seed,
        "train_azimuth_intervals_deg": TRAIN_AZIMUTH_INTERVALS_DEG,
        "validation_azimuth_intervals_deg": VALIDATION_AZIMUTH_INTERVALS_DEG,
        "max_sync_delta_s": args.max_sync_delta_s,
        "max_camera_stamp_delta_s": args.max_camera_stamp_delta_s,
        "candidate_pool_count": candidate_pool_count,
        "requested_split_counts": desired_split_counts,
        "candidate_pool_split_counts": candidate_split_counts,
        "capture_candidate_multiplier": (
            CAPTURE_CANDIDATE_MULTIPLIER
            if args.sample_count > MAX_AUTHORIZED_PILOT_SAMPLES else 1
        ),
        "test_azimuth_intervals_deg": TEST_AZIMUTH_INTERVALS_DEG,
        "image_target_x_fraction_range": IMAGE_TARGET_X_FRACTION_RANGE,
        "image_target_y_fraction_range": IMAGE_TARGET_Y_FRACTION_RANGE,
    })
    run_token = getattr(args, "run_token", uuid.uuid4().hex)
    progress_path = output_dir / "metadata" / "capture_progress.json"
    progress_manifest = {
        "schema": "suitcase-pilot-v3-independent-camera",
        "status": "RUNNING",
        "run_token": run_token,
        "world_path": str(Path(args.world).resolve()),
        "world_sha256": world_hash,
        "mesh_path": str(Path(args.mesh).resolve()),
        "mesh_sha256": mesh_hash,
        "camera_model_path": str(camera_model_path),
        "camera_model_sha256": camera_model_hash,
        "configuration_sha256": configuration_sha,
        "random_seed": args.seed,
        "requested_sample_count": args.sample_count,
        "requested_split_counts": desired_split_counts,
        "candidate_pool_count": candidate_pool_count,
        "candidate_pool_split_counts": candidate_split_counts,
        "max_frame_acquisition_retries_per_candidate": MAX_FRAME_ACQUISITION_RETRIES,
        "planned_sample_count": len(planned_samples),
        "pilot_plan": [asdict(sample) for sample in planned_samples],
        "planned_full_geometry_boxes_xyxy_px": planned_boxes,
        "camera_configuration": camera_configuration,
        "coverage_summary": pilot_coverage_summary(planned_samples),
        "static_world_collision_boxes_checked": len(planned_obstacles),
        "target_initial_position_world_m": static_target_position,
        "target_initial_orientation_world_xyzw": static_target_q,
        "target_visual_center_world_m": static_target_center,
        "topics": {
            "rgb": args.image_topic,
            "depth": args.depth_topic,
            "camera_info": args.camera_info_topic,
            "depth_camera_info": args.depth_camera_info_topic,
        },
    }
    write_capture_progress(progress_path, run_token, progress_manifest, [], [], None)

    try:
        import rclpy
        from cv_bridge import CvBridge
        from gazebo_msgs.srv import GetEntityState, SetEntityState
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Humble, gazebo_msgs, cv_bridge, and sensor_msgs must be sourced"
        ) from error

    class DatasetCaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("suitcase_dataset_pilot")
            self.bridge = CvBridge()
            self.latest_color = None
            self.latest_depth = None
            self.latest_color_info = None
            self.latest_depth_info = None
            self.latest_color_received_at = None
            self.latest_depth_received_at = None
            self.latest_color_info_received_at = None
            self.latest_depth_info_received_at = None
            self.callback_counts = {"rgb": 0, "depth": 0, "rgb_camera_info": 0, "depth_camera_info": 0}
            self.create_subscription(Image, args.image_topic, self._on_color, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, args.camera_info_topic, self._on_color_info, qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo, args.depth_camera_info_topic, self._on_depth_info, qos_profile_sensor_data,
            )
            self.set_state = self.create_client(SetEntityState, args.set_state_service)
            self.get_state = self.create_client(GetEntityState, args.get_state_service)

        def _on_color(self, message: Image) -> None:
            self.latest_color = message
            self.latest_color_received_at = time.monotonic()
            self.callback_counts["rgb"] += 1

        def _on_depth(self, message: Image) -> None:
            self.latest_depth = message
            self.latest_depth_received_at = time.monotonic()
            self.callback_counts["depth"] += 1

        def _on_color_info(self, message: CameraInfo) -> None:
            self.latest_color_info = message
            self.latest_color_info_received_at = time.monotonic()
            self.callback_counts["rgb_camera_info"] += 1

        def _on_depth_info(self, message: CameraInfo) -> None:
            self.latest_depth_info = message
            self.latest_depth_info_received_at = time.monotonic()
            self.callback_counts["depth_camera_info"] += 1

        def spin_future(self, future, timeout_s: float = 10.0):
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
            if not future.done():
                raise TimeoutError("Gazebo service call timed out")
            result = future.result()
            if result is None:
                raise RuntimeError("Gazebo service returned no result")
            return result

        def entity_state_response(self, entity_name: str):
            request = GetEntityState.Request()
            request.name = entity_name
            request.reference_frame = "world"
            response = self.spin_future(self.get_state.call_async(request))
            if not response.success:
                raise RuntimeError(f"GetEntityState failed for {entity_name}")
            return response

        def set_camera(self, sample: PilotSample) -> None:
            request = SetEntityState.Request()
            populate_set_entity_state_request(request, CAMERA_ENTITY, sample)
            response = self.spin_future(self.set_state.call_async(request))
            if not response.success:
                raise RuntimeError(f"SetEntityState failed for {CAMERA_ENTITY}")

        @staticmethod
        def _frame_record(message, received_at: float | None) -> dict | None:
            if message is None:
                return None
            return {
                "stamp_s": stamp_seconds(message.header.stamp),
                "frame_id": message.header.frame_id,
                "encoding": getattr(message, "encoding", None),
                "width": getattr(message, "width", None),
                "height": getattr(message, "height", None),
                "received_monotonic_s": received_at,
            }

        def snapshot(
            self,
            timeout_s: float,
            camera_pose_stamp_s: float,
            camera_state_received_at: float,
            target_state_received_at: float,
            sample: PilotSample,
            attempt_index: int = 0,
        ):
            deadline = time.monotonic() + timeout_s
            rejection_counts: dict[str, int] = {}
            latest_metrics: dict = {}
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
                color, depth = self.latest_color, self.latest_depth
                color_info, depth_info = self.latest_color_info, self.latest_depth_info
                if color is None or depth is None or color_info is None or depth_info is None:
                    rejection_counts["INPUT_NOT_YET_RECEIVED"] = rejection_counts.get("INPUT_NOT_YET_RECEIVED", 0) + 1
                    continue
                color_received_at = self.latest_color_received_at
                depth_received_at = self.latest_depth_received_at
                if (color_received_at is None or depth_received_at is None
                        or color_received_at <= camera_state_received_at
                        or depth_received_at <= camera_state_received_at):
                    rejection_counts["FRAME_CALLBACK_PRECEDES_CAMERA_STATE_QUERY"] = rejection_counts.get(
                        "FRAME_CALLBACK_PRECEDES_CAMERA_STATE_QUERY", 0,
                    ) + 1
                    continue
                if any(frame != color.header.frame_id for frame in (
                    depth.header.frame_id, color_info.header.frame_id, depth_info.header.frame_id,
                )):
                    rejection_counts["FRAME_ID_MISMATCH"] = rejection_counts.get("FRAME_ID_MISMATCH", 0) + 1
                    continue

                color_stamp = stamp_seconds(color.header.stamp)
                depth_stamp = stamp_seconds(depth.header.stamp)
                latest_metrics = {
                    "sample_id": sample.sample_id,
                    "camera_pose_stamp_s": camera_pose_stamp_s,
                    "camera_state_received_monotonic_s": camera_state_received_at,
                    "target_state_received_monotonic_s": target_state_received_at,
                    "rgb": self._frame_record(color, color_received_at),
                    "depth": self._frame_record(depth, depth_received_at),
                    "rgb_camera_info": self._frame_record(
                        color_info, self.latest_color_info_received_at,
                    ),
                    "depth_camera_info": self._frame_record(
                        depth_info, self.latest_depth_info_received_at,
                    ),
                    "callback_counts": dict(self.callback_counts),
                }
                if not rgbd_timestamps_are_fresh(
                    color_stamp, depth_stamp, camera_pose_stamp_s, args.max_sync_delta_s,
                ):
                    rejection_counts["RGB_DEPTH_OR_POSE_HEADER_STAMP_MISMATCH"] = rejection_counts.get(
                        "RGB_DEPTH_OR_POSE_HEADER_STAMP_MISMATCH", 0,
                    ) + 1
                    continue
                pose_stamp_delta_s = max(color_stamp, depth_stamp) - camera_pose_stamp_s
                if pose_stamp_delta_s > args.max_camera_stamp_delta_s:
                    rejection_counts["CAMERA_POSE_TO_FRAME_SIM_STAMP_DELTA_EXCEEDED"] = rejection_counts.get(
                        "CAMERA_POSE_TO_FRAME_SIM_STAMP_DELTA_EXCEEDED", 0,
                    ) + 1
                    continue

                delivery_latency_s = max(color_received_at, depth_received_at) - camera_state_received_at
                snapshot_metrics = {
                    **latest_metrics,
                    "target_state_to_rgb_callback_s": color_received_at - target_state_received_at,
                    "target_state_to_depth_callback_s": depth_received_at - target_state_received_at,
                    "camera_state_to_rgb_callback_s": color_received_at - camera_state_received_at,
                    "camera_state_to_depth_callback_s": depth_received_at - camera_state_received_at,
                    "rgb_depth_callback_delta_s": abs(color_received_at - depth_received_at),
                    "rgb_depth_header_stamp_delta_s": abs(color_stamp - depth_stamp),
                    "camera_pose_to_frame_header_stamp_delta_s": pose_stamp_delta_s,
                    "rgb_camera_info_stamp_delta_s": abs(
                        color_stamp - stamp_seconds(color_info.header.stamp)
                    ),
                    "depth_camera_info_stamp_delta_s": abs(
                        depth_stamp - stamp_seconds(depth_info.header.stamp)
                    ),
                    "delivery_latency_warning_s": args.delivery_latency_warning_s,
                    "delivery_latency_warning": delivery_latency_s > args.delivery_latency_warning_s,
                    "camera_pose_to_frame_delivery_latency_s": delivery_latency_s,
                    "frame_pair_checks": {
                        "callbacks_after_camera_state_query": True,
                        "same_frame_id": True,
                        "rgb_depth_stamp_delta_within_limit": True,
                        "frame_stamp_after_camera_pose": True,
                        "camera_pose_sim_stamp_delta_within_limit": True,
                    },
                    "rejection_counts_while_waiting": rejection_counts,
                }
                color_bgr = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
                depth_raw = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                return (
                    color_bgr.copy(), depth_raw.copy(), color, depth, color_info, depth_info,
                    snapshot_metrics,
                )

            timeout_record = {
                **latest_metrics,
                "schema": "data01-rgbd-wait-diagnostic-v2",
                "sample_id": sample.sample_id,
                "timeout_s": timeout_s,
                "camera_pose_stamp_s": camera_pose_stamp_s,
                "camera_state_received_monotonic_s": camera_state_received_at,
                "target_state_received_monotonic_s": target_state_received_at,
                "callback_counts": dict(self.callback_counts),
                "rejection_counts": rejection_counts,
                "timed_out": True,
            }
            diagnostic_path = (
                output_dir / "metadata" / "diagnostics"
                / f"frame_wait_{sample.sample_id}_attempt{attempt_index + 1}.json"
            )
            write_json_atomic(diagnostic_path, timeout_record)
            raise TimeoutError(
                "No fresh synchronized RGB-D frame and CameraInfo arrived before frame timeout; "
                f"details: {diagnostic_path}"
            )

    manifest_samples: list[dict] = []
    rejected: list[dict] = []
    retry_records: list[dict] = []
    node = None
    current_sample_id: str | None = None
    base_metadata = progress_manifest
    last_progress_persistence_s = 0.0

    def persist_progress() -> None:
        nonlocal last_progress_persistence_s
        started_at = time.perf_counter()
        base_metadata["frame_acquisition_retry_count"] = len(retry_records)
        base_metadata["frame_acquisition_retries"] = list(retry_records)
        base_metadata["accepted_split_counts"] = {
            split: sum(row["split"] == split for row in manifest_samples)
            for split in desired_split_counts
        }
        write_capture_progress(
            progress_path, run_token, base_metadata, manifest_samples, rejected,
            current_sample_id,
        )
        last_progress_persistence_s = time.perf_counter() - started_at

    def add_candidate_timings(row: dict, stage_timings: dict[str, float], started_at: float) -> None:
        stage_timings["candidate_total_s"] = time.perf_counter() - started_at
        row["stage_timings_s"] = {
            name: round(duration, 6) for name, duration in stage_timings.items()
        }

    try:
        rclpy.init(args=None)
        node = DatasetCaptureNode()
        for client, service_name in (
            (node.set_state, args.set_state_service),
            (node.get_state, args.get_state_service),
        ):
            if not client.wait_for_service(timeout_sec=args.service_timeout_s):
                raise TimeoutError(f"Required Gazebo service is unavailable: {service_name}")

        target_initial_response = node.entity_state_response(TARGET_ENTITY)
        initial_target_position, initial_target_q = entity_pose(target_initial_response)
        target_center_initial = suitcase_visual_center(
            initial_target_position, initial_target_q, mesh,
        )
        samples = pilot_samples(
            target_center_initial, candidate_pool_count, args.seed, Path(args.world).resolve(),
        )
        capture_obstacles = validate_camera_clearance(Path(args.world).resolve(), samples)
        pilot_boxes = projected_pilot_boxes(
            samples, initial_target_position, initial_target_q, target_center_initial, mesh,
        )

        base_metadata = {
            "schema": "suitcase-pilot-v3-independent-camera",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_token": run_token,
            "status": "RUNNING",
            "random_seed": args.seed,
            "requested_sample_count": args.sample_count,
            "requested_split_counts": desired_split_counts,
            "candidate_pool_count": candidate_pool_count,
            "candidate_pool_split_counts": candidate_split_counts,
            "max_frame_acquisition_retries_per_candidate": MAX_FRAME_ACQUISITION_RETRIES,
            "planned_sample_count": len(samples),
            "configuration_sha256": configuration_sha,
            "world_path": str(Path(args.world).resolve()),
            "world_sha256": world_hash,
            "scene_variant": scene_variant_for_world(Path(args.world).resolve()),
            "production_world_sha256": WORLD_SHA256,
            "target_model_source_world_sha256": WORLD_SHA256,
            "mesh_path": str(Path(args.mesh).resolve()),
            "mesh_sha256": mesh_hash,
            "camera_model_path": str(camera_model_path),
            "camera_model_sha256": camera_model_hash,
            "furniture_sha256": FURNITURE_SHA256,
            "static_world_collision_boxes_checked": len(capture_obstacles),
            "camera_clearance_radius_m": CAMERA_CLEARANCE_RADIUS_M,
            "target_entity": args.target_entity,
            "target_pose_source": f"GetEntityState({args.target_entity})",
            "camera_pose_source": {
                "service": args.get_state_service,
                "entity": CAMERA_ENTITY,
                "link_frame": "dataset_camera_link",
                "optical_frame": OPTICAL_FRAME_NAME,
                "link_to_optical_rotation_xyzw": CAMERA_LINK_TO_OPTICAL_ROTATION.tolist(),
                "max_camera_pose_to_frame_stamp_delta_s": args.max_camera_stamp_delta_s,
                "delivery_latency_warning_s": args.delivery_latency_warning_s,
            },
            "topics": {
                "rgb": args.image_topic,
                "depth": args.depth_topic,
                "camera_info": args.camera_info_topic,
                "depth_camera_info": args.depth_camera_info_topic,
            },
            "camera": {
                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT,
                "update_rate_hz": CAMERA_UPDATE_RATE_HZ,
                "horizontal_fov_rad": CAMERA_HORIZONTAL_FOV_RAD,
                "horizontal_distance_range_m": CAMERA_DISTANCE_RANGE_M,
                "height_above_target_center_range_m": CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M,
                "random_seed": args.seed,
                "sample_count": args.sample_count,
                "candidate_pool_count": len(samples),
                "requested_split_counts": desired_split_counts,
                "image_target_x_fraction_range": IMAGE_TARGET_X_FRACTION_RANGE,
                "image_target_y_fraction_range": IMAGE_TARGET_Y_FRACTION_RANGE,
            },
            "geometry": {
                "source": "hard_trolley_red DAE vertices and triangle topology; visibility uses rasterized nearest front-surface depth",
                "vertices": int(mesh.vertices.shape[0]),
                "triangles": mesh.triangle_count,
                "local_bounds_before_visual_offset_m": {
                    "min": mesh.local_bounds_min,
                    "max": mesh.local_bounds_max,
                },
                "visual_offset_m": [0.0, 0.0, -0.05],
            },
            "label": {"format": "YOLO", "class_id": YOLO_CLASS_ID, "class_name": YOLO_CLASS_NAME},
            "target_initial_position_world_m": initial_target_position,
            "target_initial_orientation_world_xyzw": initial_target_q,
            "target_visual_center_world_m": target_center_initial,
            "pilot_plan": [asdict(sample) for sample in samples],
            "coverage_summary": pilot_coverage_summary(samples),
            "planned_full_geometry_boxes_xyxy_px": pilot_boxes,
            "camera_orbit": {
                "target_is_moved": False,
                "horizontal_distance_range_m": CAMERA_DISTANCE_RANGE_M,
                "height_above_target_center_range_m": CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M,
                "train_azimuth_intervals_deg": TRAIN_AZIMUTH_INTERVALS_DEG,
                "validation_azimuth_intervals_deg": VALIDATION_AZIMUTH_INTERVALS_DEG,
                "test_azimuth_intervals_deg": TEST_AZIMUTH_INTERVALS_DEG,
                "azimuth_guard_intervals_deg": (
                    (44.0, 54.0), (90.0, 100.0), (140.0, 150.0),
                    (210.0, 220.0), (230.0, 240.0), (270.0, 280.0),
                ),
                "same_azimuth_split_blocks_across_scenes": True,
                "orientation": "camera link +X is rotated so the DAE visual center lands at a seeded image point",
                "random_seed": args.seed,
                "planned_sample_count": len(samples),
                "split_counts": {
                    split: sum(sample.split == split for sample in samples)
                    for split in candidate_split_counts
                },
            },
            "split_policy": (
                "Pilot uses 80/20. Each mixed-scene component uses 80/10/10. "
                "Full-data azimuth blocks are disjoint with guard gaps and assigned "
                "to the same split for both scene backgrounds; neighboring views are not randomly split."
            ),
            "kulunuos_reference": {
                "repository": "https://github.com/KulunuOS/gazebo_dataset_generation",
                "revision": "928c0be3d1366adf3ce3a82e4c432fc1b0171b46",
                "reused_concept": "independent camera orbit and RGB/depth/mask-aligned capture",
                "code_copied": False,
            },
            "asset_attribution": (
                'Based on “Hard Trolley Bag With Adjustable Handle” by Adrian Saputra, '
                'Sketchfab UID 1758ad5207d943d597db8ff2b21bf827, CC BY 4.0; '
                'red shell variant. Source credit retained in simulation_assets.'
            ),
        }
        persist_progress()

        accepted_split_counts = {split: 0 for split in desired_split_counts}
        unattempted_candidate_ids: list[str] = []

        def read_capture_pose(sample: PilotSample) -> dict:
            camera_response = node.entity_state_response(CAMERA_ENTITY)
            camera_state_received_at = time.monotonic()
            camera_link_position, camera_link_q = entity_pose(camera_response)
            target_response = node.entity_state_response(args.target_entity)
            target_state_received_at = time.monotonic()
            target_position, target_q = entity_pose(target_response)
            target_position_delta, target_orientation_delta = capture_pose_delta(
                initial_target_position, initial_target_q, target_position, target_q,
            )
            if (target_position_delta > CAPTURE_POSE_MAX_POSITION_DELTA_M
                    or target_orientation_delta > CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD):
                raise RuntimeError(
                    "Approved suitcase moved during camera orbit; refusing geometry labels"
                )
            target_center = suitcase_visual_center(target_position, target_q, mesh)
            camera_distance, _ = validate_camera_sample_pose(
                sample, camera_link_position, camera_link_q, target_center,
            )
            optical_position, optical_q = camera_optical_pose(
                camera_link_position, camera_link_q,
            )
            return {
                "camera_response": camera_response,
                "camera_state_received_at": camera_state_received_at,
                "camera_link_position": camera_link_position,
                "camera_link_q": camera_link_q,
                "target_state_received_at": target_state_received_at,
                "target_position": target_position,
                "target_q": target_q,
                "target_center": target_center,
                "camera_distance": camera_distance,
                "optical_position": optical_position,
                "optical_q": optical_q,
                "camera_state_stamp_s": stamp_seconds(camera_response.header.stamp),
            }

        for candidate_index, sample in enumerate(samples):
            if all(
                accepted_split_counts[split] >= target_count
                for split, target_count in desired_split_counts.items()
            ):
                unattempted_candidate_ids.extend(
                    candidate.sample_id for candidate in samples[candidate_index:]
                )
                break
            if accepted_split_counts[sample.split] >= desired_split_counts[sample.split]:
                unattempted_candidate_ids.append(sample.sample_id)
                continue
            current_sample_id = sample.sample_id
            candidate_started_at = time.perf_counter()
            stage_timings: dict[str, float] = {}
            persist_progress()
            stage_timings["pre_candidate_progress_persistence_s"] = last_progress_persistence_s
            stage_started_at = time.perf_counter()
            node.set_camera(sample)
            stage_timings["set_camera_service_s"] = time.perf_counter() - stage_started_at
            stage_started_at = time.perf_counter()
            settle_deadline = time.monotonic() + args.settle_s
            while rclpy.ok() and time.monotonic() < settle_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            stage_timings["initial_settle_s"] = time.perf_counter() - stage_started_at

            stage_started_at = time.perf_counter()
            pose = read_capture_pose(sample)
            stage_timings["initial_pose_read_s"] = time.perf_counter() - stage_started_at
            snapshot_result = None
            acquisition_failures: list[dict] = []
            stage_started_at = time.perf_counter()
            for attempt_index in range(MAX_FRAME_ACQUISITION_RETRIES + 1):
                if attempt_index > 0:
                    retry_started_at = time.perf_counter()
                    settle_deadline = time.monotonic() + args.settle_s
                    while rclpy.ok() and time.monotonic() < settle_deadline:
                        rclpy.spin_once(node, timeout_sec=0.05)
                    pose = read_capture_pose(sample)
                    stage_timings["retry_settle_and_pose_read_s"] = (
                        stage_timings.get("retry_settle_and_pose_read_s", 0.0)
                        + time.perf_counter() - retry_started_at
                    )
                try:
                    snapshot_result = node.snapshot(
                        args.frame_timeout_s, pose["camera_state_stamp_s"],
                        pose["camera_state_received_at"], pose["target_state_received_at"],
                        sample, attempt_index,
                    )
                    if attempt_index:
                        retry_records.append({
                            "sample_id": sample.sample_id,
                            "retry_count": attempt_index,
                            "initial_failure": acquisition_failures[0],
                            "outcome": "RECOVERED_WITH_FRESH_RGBD",
                        })
                    break
                except TimeoutError as error:
                    acquisition_failures.append({
                        "attempt": attempt_index + 1,
                        "error": str(error),
                    })
                    if attempt_index == MAX_FRAME_ACQUISITION_RETRIES:
                        rejected.append({
                            "sample_id": sample.sample_id,
                            "split": sample.split,
                            "reason": "FRESH_SYNCHRONIZED_RGBD_TIMEOUT_AFTER_LIMITED_RETRY",
                            "attempts": acquisition_failures,
                        })
                        retry_records.append({
                            "sample_id": sample.sample_id,
                            "retry_count": attempt_index,
                            "initial_failure": acquisition_failures[0],
                            "final_failure": acquisition_failures[-1],
                            "outcome": "CANDIDATE_REJECTED",
                        })
            stage_timings["frame_acquisition_s"] = time.perf_counter() - stage_started_at
            if snapshot_result is None:
                current_sample_id = None
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(rejected[-1], stage_timings, candidate_started_at)
                continue
            (
                color, depth_raw, color_message, depth_message, info, depth_info,
                snapshot_metrics,
            ) = snapshot_result
            snapshot_metrics["frame_acquisition_retry_count"] = len(acquisition_failures)
            camera_response = pose["camera_response"]
            camera_state_received_at = pose["camera_state_received_at"]
            camera_link_position = pose["camera_link_position"]
            camera_link_q = pose["camera_link_q"]
            target_state_received_at = pose["target_state_received_at"]
            target_position = pose["target_position"]
            target_q = pose["target_q"]
            target_center = pose["target_center"]
            camera_distance = pose["camera_distance"]
            optical_position = pose["optical_position"]
            optical_q = pose["optical_q"]

            stage_started_at = time.perf_counter()
            snapshot_completed_at = time.monotonic()
            camera_after_response = node.entity_state_response(CAMERA_ENTITY)
            camera_after_received_at = time.monotonic()
            camera_after_position, camera_after_q = entity_pose(camera_after_response)
            camera_position_delta, camera_orientation_delta = capture_pose_delta(
                camera_link_position, camera_link_q, camera_after_position, camera_after_q,
            )
            if (camera_position_delta > CAPTURE_POSE_MAX_POSITION_DELTA_M
                    or camera_orientation_delta > CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD):
                raise RuntimeError(
                    "Dataset camera moved during RGB-D capture; refusing labels from a stale pose"
                )

            target_after_response = node.entity_state_response(args.target_entity)
            target_after_position, target_after_q = entity_pose(target_after_response)
            target_position_delta, target_orientation_delta = capture_pose_delta(
                target_position, target_q, target_after_position, target_after_q,
            )
            if (target_position_delta > CAPTURE_POSE_MAX_POSITION_DELTA_M
                    or target_orientation_delta > CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD):
                raise RuntimeError(
                    "Approved suitcase moved during RGB-D capture; refusing geometry labels"
                )
            stage_timings["post_capture_pose_checks_s"] = time.perf_counter() - stage_started_at

            quality_started_at = time.perf_counter()
            height, width = color.shape[:2]
            if (
                (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT)
                or depth_raw.shape[:2] != (height, width)
                or (info.height, info.width) != (height, width)
                or (depth_info.height, depth_info.width) != (height, width)
            ):
                row = {"sample_id": sample.sample_id, "reason": "RGB_DEPTH_CAMERAINFO_SIZE_MISMATCH"}
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(row, stage_timings, candidate_started_at)
                continue
            if color_message.encoding != "rgb8" or depth_message.encoding != "32FC1":
                row = {"sample_id": sample.sample_id, "reason": "RGB_DEPTH_ENCODING_MISMATCH"}
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(row, stage_timings, candidate_started_at)
                continue
            if not camera_infos_share_calibration(info, depth_info):
                row = {"sample_id": sample.sample_id, "reason": "RGB_DEPTH_CALIBRATION_MISMATCH"}
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(row, stage_timings, candidate_started_at)
                continue
            fx, fy, cx, cy = (
                float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]),
            )
            if (
                fx <= 0 or fy <= 0
                or info.binning_x != 0 or info.binning_y != 0
                or info.roi.width != 0 or info.roi.height != 0
            ):
                row = {"sample_id": sample.sample_id, "reason": "INVALID_CAMERA_INTRINSICS"}
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(row, stage_timings, candidate_started_at)
                continue
            expected_focal_px = CAMERA_WIDTH / (2.0 * math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))
            if (abs(fx - expected_focal_px) > expected_focal_px * 0.01
                    or abs(fy - expected_focal_px) > expected_focal_px * 0.01
                    or abs(cx - CAMERA_WIDTH / 2.0) > 1.5
                    or abs(cy - CAMERA_HEIGHT / 2.0) > 1.5):
                row = {"sample_id": sample.sample_id, "reason": "CAMERAINFO_MISMATCHES_APPROVED_FOV"}
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(row, stage_timings, candidate_started_at)
                continue
            intrinsics = (fx, fy, cx, cy)
            distortion = tuple(float(value) for value in info.d)

            vertices_local = mesh.vertices.copy()
            vertices_local[:, 2] -= 0.05
            points_camera = pose_points_to_camera(
                vertices_local, target_position, target_q, (0.0, 0.0, 0.0),
                optical_position, optical_q,
            )
            visibility = None
            try:
                geometry_bbox = projected_bbox(
                    points_camera, intrinsics, width, height, distortion=distortion,
                )
                depth_m = depth_to_meters(depth_raw, depth_message.encoding)
                visibility, visible_mask, expected_depth_m = depth_visibility_products(
                    points_camera, mesh.triangles, intrinsics, depth_m, distortion,
                )
                if visibility["visible_fraction"] < MIN_VISIBLE_FRACTION:
                    raise CaptureRejected(
                        f"OCCLUDED_OR_DEPTH_MISMATCH_{visibility['visible_fraction']:.3f}"
                    )
                bbox = visible_bbox_from_mask(visible_mask, width, height)
            except CaptureRejected as error:
                rejected_row = {"sample_id": sample.sample_id, "reason": str(error)}
                if visibility is not None:
                    rejected_row["depth_visibility"] = visibility
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(rejected_row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(rejected_row, stage_timings, candidate_started_at)
                continue

            normalized = yolo_bbox(bbox, width, height)
            if (not np.all(np.isfinite(normalized))
                    or normalized[2] <= 0.0 or normalized[3] <= 0.0
                    or normalized[0] - normalized[2] / 2.0 < 0.0
                    or normalized[1] - normalized[3] / 2.0 < 0.0
                    or normalized[0] + normalized[2] / 2.0 > 1.0
                    or normalized[1] + normalized[3] / 2.0 > 1.0):
                row = {"sample_id": sample.sample_id, "reason": "INVALID_NORMALIZED_VISIBLE_OBJECT_BBOX"}
                stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
                rejected.append(row)
                persist_progress()
                stage_timings["post_candidate_progress_persistence_s"] = last_progress_persistence_s
                add_candidate_timings(row, stage_timings, candidate_started_at)
                continue
            stage_timings["quality_and_geometry_s"] = time.perf_counter() - quality_started_at
            artifact_write_started_at = time.perf_counter()
            image_path = output_dir / "images" / sample.split / f"{sample.sample_id}.png"
            if not cv2.imwrite(str(image_path), color):
                raise IOError(f"Could not write RGB frame: {image_path}")
            depth_path = output_dir / "depth" / sample.split / f"{sample.sample_id}.npy"
            np.save(depth_path, depth_m.astype(np.float32, copy=False))
            label_path = output_dir / "labels" / sample.split / f"{sample.sample_id}.txt"
            label_path.write_text(
                f"{YOLO_CLASS_ID} " + " ".join(f"{value:.8f}" for value in normalized) + "\n",
                encoding="utf-8",
            )

            x1, y1, x2, y2 = bbox
            geometry_x1, geometry_y1, geometry_x2, geometry_y2 = geometry_bbox
            preview = color.copy()
            cv2.rectangle(
                preview,
                (round(geometry_x1), round(geometry_y1)),
                (round(geometry_x2), round(geometry_y2)),
                (0, 165, 255), 1,
            )
            cv2.rectangle(preview, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"visible GT (green); full geometry (orange); {sample.scenario_id}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"azimuth={sample.azimuth_deg:.1f} deg distance={sample.camera_distance_m:.2f}m height={sample.camera_height_above_target_m:.2f}m",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
            preview_path = output_dir / "preview" / "samples" / f"{sample.sample_id}.jpg"
            if not cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise IOError(f"Could not write RGB label overlay: {preview_path}")
            depth_preview_path = output_dir / "preview" / "depth" / f"{sample.sample_id}.png"
            depth_preview = depth_diagnostic_preview(depth_m, expected_depth_m, visible_mask)
            if not cv2.imwrite(str(depth_preview_path), depth_preview):
                raise IOError(f"Could not write depth diagnostic: {depth_preview_path}")
            stage_timings["artifact_write_s"] = time.perf_counter() - artifact_write_started_at
            add_candidate_timings({}, stage_timings, candidate_started_at)

            rgb_stamp = stamp_seconds(color_message.header.stamp)
            depth_stamp = stamp_seconds(depth_message.header.stamp)
            manifest_samples.append({
                **asdict(sample),
                "accepted": True,
                "frame_acquisition_retry_count": len(acquisition_failures),
                "rgb_path": str(image_path.relative_to(output_dir)),
                "depth_path": str(depth_path.relative_to(output_dir)),
                "label_path": str(label_path.relative_to(output_dir)),
                "preview_path": str(preview_path.relative_to(output_dir)),
                "depth_diagnostic_preview_path": str(depth_preview_path.relative_to(output_dir)),
                "rgb_stamp_s": rgb_stamp,
                "depth_stamp_s": depth_stamp,
                "rgb_depth_delta_s": abs(rgb_stamp - depth_stamp),
                "camera_pose_source": f"GetEntityState({CAMERA_ENTITY})",
                "camera_pose_to_frame_delivery_latency_s": snapshot_metrics["camera_pose_to_frame_delivery_latency_s"],
                "camera_pose_to_frame_header_stamp_delta_s": snapshot_metrics["camera_pose_to_frame_header_stamp_delta_s"],
                "delivery_latency_warning": snapshot_metrics["delivery_latency_warning"],
                "snapshot_diagnostics": snapshot_metrics,
                "stage_timings_s": dict(stage_timings),
                "frame_to_camera_state_response_s": camera_after_received_at - snapshot_completed_at,
                "camera_link_position_world_m": camera_link_position,
                "camera_link_orientation_world_xyzw": camera_link_q,
                "camera_optical_position_world_m": optical_position,
                "camera_optical_orientation_world_xyzw": optical_q,
                "camera_post_frame_delta_m": camera_position_delta,
                "camera_post_frame_delta_rad": camera_orientation_delta,
                "camera_horizontal_distance_m": camera_distance,
                "camera_height_above_target_center_m": sample.camera_height_above_target_m,
                "configuration_sha256": base_metadata["configuration_sha256"],
                "scene_variant": scene_variant_for_world(Path(args.world).resolve()),
                "capture_world_sha256": world_hash,
                "production_world_sha256": WORLD_SHA256,
                "suitcase_mesh_sha256": mesh_hash,
                "image_encoding": color_message.encoding,
                "depth_encoding": depth_message.encoding,
                "frame_id": color_message.header.frame_id,
                "image_width": width,
                "image_height": height,
                "camera_info_parameters": {
                    "rgb": camera_info_parameters(info),
                    "depth": camera_info_parameters(depth_info),
                },
                "target_position_world_m": target_position,
                "target_orientation_world_xyzw": target_q,
                "camera_configuration_sha256": camera_info_configuration_sha256(info, depth_info),
                "bbox_visible_xyxy_px": [x1, y1, x2, y2],
                "bbox_full_geometry_xyxy_px": list(geometry_bbox),
                "bbox_yolo_xywh_normalized": normalized,
                "depth_visibility": visibility,
                "quality_checks": {
                    "rgb_depth_header_stamps_paired": True,
                    "rgb_depth_frame_ids_match": True,
                    "rgb_depth_camera_info_calibration_matches": True,
                    "camera_info_dimensions_valid": True,
                    "camera_info_matches_approved_fov": True,
                    "camera_static_during_capture": True,
                    "target_static_during_capture": True,
                    "visible_bbox_from_depth_confirmed_surface": True,
                    "normalized_bbox_valid": True,
                    "label_uses_yolo_prediction": False,
                },
                "label_origin": (
                    "bounding box of DAE front-surface pixels confirmed by aligned depth; "
                    "full-geometry box retained separately; never a YOLO prediction"
                ),
            })
            accepted_split_counts[sample.split] += 1
            persist_progress()
            manifest_samples[-1]["stage_timings_s"]["post_candidate_progress_persistence_s"] = round(
                last_progress_persistence_s, 6,
            )
            manifest_samples[-1]["stage_timings_s"]["candidate_total_s"] = round(
                time.perf_counter() - candidate_started_at, 6,
            )

        current_sample_id = None
        accepted_split_counts = {
            split: sum(row["split"] == split for row in manifest_samples)
            for split in desired_split_counts
        }
        split_counts_match_target = accepted_split_counts == desired_split_counts
        status = (
            "COMPLETE"
            if len(manifest_samples) == args.sample_count and split_counts_match_target
            else "FAILED"
        )
        base_metadata.update({
            "status": status,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "requested_sample_count": args.sample_count,
            "planned_sample_count": len(samples),
            "candidate_pool_count": len(samples),
            "attempted_sample_count": len(manifest_samples) + len(rejected),
            "frame_acquisition_attempt_count": (
                len(manifest_samples) + len(rejected) + len(retry_records)
            ),
            "accepted_sample_count": len(manifest_samples),
            "accepted_split_counts": accepted_split_counts,
            "requested_split_counts": desired_split_counts,
            "rejected_sample_count": len(rejected),
            "unattempted_candidate_count": len(unattempted_candidate_ids),
            "unattempted_candidate_ids": unattempted_candidate_ids,
            "frame_acquisition_retry_count": len(retry_records),
            "frame_acquisition_retries": retry_records,
            "rejection_reason_counts": {
                reason: sum(row.get("reason") == reason for row in rejected)
                for reason in sorted({row.get("reason", "UNKNOWN") for row in rejected})
            },
            "dataset_ready_for_training": False,
            "visual_review_status": "AWAITING_AUTHOR" if status == "COMPLETE" else "NOT_READY",
            "failure": None if status == "COMPLETE" else {
                "type": "QUALITY_GATE_FAILURE",
                "message": (
                    "The bounded candidate pool did not produce the requested accepted split counts."
                ),
                "accepted_split_counts": accepted_split_counts,
                "requested_split_counts": desired_split_counts,
            },
            "samples": manifest_samples,
            "rejected_samples": rejected,
            "capture_settings": {
                "settle_s": args.settle_s,
                "frame_timeout_s": args.frame_timeout_s,
                "max_sync_delta_s": args.max_sync_delta_s,
                "max_camera_stamp_delta_s": args.max_camera_stamp_delta_s,
                "max_frame_acquisition_retries_per_candidate": MAX_FRAME_ACQUISITION_RETRIES,
                "delivery_latency_warning_s": args.delivery_latency_warning_s,
                "minimum_visible_fraction": MIN_VISIBLE_FRACTION,
                "depth_tolerance_m": "max(0.06, 0.03 * expected_z)",
                "label_bbox_policy": "depth-confirmed visible front-surface pixels; full geometry bbox is diagnostic only",
                "edge_clipping_policy": "reject if projected full geometry crosses a 2 px image margin",
                "target_moved": False,
            },
        })
        write_json_atomic(output_dir / "metadata" / "capture_manifest.json", base_metadata)
        write_json_atomic(output_dir / "metadata" / "rejected_samples.json", rejected)
        if status == "COMPLETE":
            yaml_lines = ["path: .", "train: images/train", "val: images/val"]
            if "test" in desired_split_counts:
                yaml_lines.append("test: images/test")
            yaml_lines.extend(("names:", "  0: suitcase"))
            (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        attribution = (
            'Based on “Hard Trolley Bag With Adjustable Handle” by Adrian Saputra, '
            'Sketchfab UID 1758ad5207d943d597db8ff2b21bf827; CC BY 4.0. '
            'Red shell variant. Original credit: https://sketchfab.com/3d-models/'
            'hard-trolley-bag-with-adjustable-handle-1758ad5207d943d597db8ff2b21bf827\n'
            'Dataset workflow reference: KulunuOS/gazebo_dataset_generation, '
            'revision 928c0be3d1366adf3ce3a82e4c432fc1b0171b46, render.py; '
            'camera-sweep concept only, no source code copied.\n'
        )
        (output_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
        contact_samples = representative_contact_samples(manifest_samples)
        create_contact_sheet(output_dir, contact_samples)
        write_json_atomic(
            output_dir / "preview" / "contact_sheet_samples.json",
            {
                "selected_count": len(contact_samples),
                "accepted_count": len(manifest_samples),
                "selection": "evenly spaced by azimuth within each split; at most 16 per split",
                "sample_ids": [sample["sample_id"] for sample in contact_samples],
            },
        )
        persist_progress()
        print(
            f"Dataset status: {status}; accepted={len(manifest_samples)}; "
            f"rejected={len(rejected)}"
        )
        print(f"Output: {output_dir}")
        if status != "COMPLETE":
            raise RuntimeError(
                f"Dataset FAILED split-count gates: accepted {len(manifest_samples)}/{args.sample_count}; "
                f"split_counts={accepted_split_counts}; rejected_candidates={len(rejected)}. "
                "See metadata/capture_manifest.json"
            )
        return output_dir
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def diagnostic_sample_from_manifest(
    source_manifest_path: Path,
    sample_id: str | None,
    expected_world_sha256: str,
    expected_mesh_sha256: str,
) -> tuple[PilotSample, dict]:
    """Load one exact planned test pose for a raw diagnostic, never for training."""
    if not sample_id:
        raise ValueError("--diagnostic-source-manifest requires --diagnostic-sample-id")
    path = Path(source_manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("world_sha256") != expected_world_sha256:
        raise ValueError("Diagnostic source manifest uses a different approved world")
    if manifest.get("mesh_sha256") != expected_mesh_sha256:
        raise ValueError("Diagnostic source manifest uses a different suitcase mesh")
    matches = [
        row for row in manifest.get("pilot_plan", [])
        if row.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one planned sample named {sample_id!r}")
    record = matches[0]
    if record.get("split") != "test" or not str(record.get("scenario_id", "")).startswith(
        "test_azimuth_band_"
    ):
        raise ValueError("A diagnostic source sample must belong to a held-out test band")
    required = set(PilotSample.__dataclass_fields__)
    if not required.issubset(record):
        missing = sorted(required - set(record))
        raise ValueError(f"Diagnostic source sample is missing fields: {missing}")
    sample_record = {key: record[key] for key in required}
    sample_record["camera_link_quaternion_xyzw"] = tuple(
        sample_record["camera_link_quaternion_xyzw"]
    )
    sample = PilotSample(**sample_record)
    scenario_parts = sample.scenario_id.rsplit("_", 2)
    try:
        low, high = float(scenario_parts[-2]), float(scenario_parts[-1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid test-band scenario id: {sample.scenario_id}") from error
    if not low <= sample.azimuth_deg < high:
        raise ValueError("Diagnostic sample azimuth is outside its recorded test band")
    provenance = {
        "source_manifest_path": str(path),
        "source_manifest_sha256": sha256_file(path),
        "source_run_token": manifest.get("run_token"),
        "sample_id": sample.sample_id,
        "split": sample.split,
        "scenario_id": sample.scenario_id,
    }
    return sample, provenance


def run_diagnostic_single_shot(
    args: argparse.Namespace, mesh: MeshGeometry, world_hash: str, mesh_hash: str,
) -> Path:
    """Capture one isolated diagnostic set, never writing dataset labels."""
    from collections import deque

    if args.max_camera_stamp_delta_s != DEFAULT_MAX_CAMERA_STAMP_DELTA_S:
        raise ValueError(
            "--diagnostic-single-shot keeps the existing 0.1 s stamp delta as a fixed metric"
        )
    if (not math.isfinite(args.service_timeout_s) or args.service_timeout_s <= 0.0
            or not math.isfinite(args.frame_timeout_s) or args.frame_timeout_s <= 0.0
            or not math.isfinite(args.settle_s) or args.settle_s < 0.0):
        raise ValueError("Diagnostic service/frame timeouts must be positive and settle time nonnegative")

    source_sample = None
    source_sample_provenance = None
    if args.diagnostic_source_manifest is not None:
        source_sample, source_sample_provenance = diagnostic_sample_from_manifest(
            args.diagnostic_source_manifest, args.diagnostic_sample_id,
            world_hash, mesh_hash,
        )

    try:
        import rclpy
        from cv_bridge import CvBridge
        from gazebo_msgs.srv import GetEntityState, SetEntityState
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
        from rclpy.serialization import serialize_message
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as error:
        raise RuntimeError(
            "Diagnostic capture requires ROS 2 Humble, rosgraph_msgs, gazebo_msgs, "
            "cv_bridge, and sensor_msgs"
        ) from error

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    for name in ("raw", "decoded", "preview", "metadata"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "NOT_GROUND_TRUTH.txt").write_text(
        "DATA-01 single-shot diagnostic output. Not approved as ground truth or training data.\n",
        encoding="utf-8",
    )

    run_token, epoch_id = getattr(args, "run_token", uuid.uuid4().hex), f"pose_epoch_{uuid.uuid4().hex}"
    manifest_path = output_dir / "metadata" / "diagnostic_manifest.json"
    raw_paths: dict[str, str] = {}
    raw_index_paths: dict[str, dict[int, str]] = {}
    raw_history_paths: dict[str, list[str]] = {}
    selected_callback_indexes: dict[str, int] = {}
    manifest = {
        "schema": "data01-single-shot-diagnostic-v1", "mode": "diagnostic-single-shot",
        "status": "RUNNING", "run_token": run_token, "pose_epoch_id": epoch_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "world_path": str(Path(args.world).resolve()), "world_sha256": world_hash,
        "mesh_path": str(Path(args.mesh).resolve()), "mesh_sha256": mesh_hash,
        "camera_model_path": str(SIMULATION_SHARE / "models/dataset_camera/model.sdf"),
        "camera_model_sha256": None, "seed": args.seed,
        "topics": {"rgb": args.image_topic, "depth": args.depth_topic,
                   "rgb_camera_info": args.camera_info_topic,
                   "depth_camera_info": args.depth_camera_info_topic, "clock": "/clock"},
        "camera_configuration": {
            "width": CAMERA_WIDTH, "height": CAMERA_HEIGHT,
            "horizontal_fov_rad": CAMERA_HORIZONTAL_FOV_RAD,
            "update_rate_hz": CAMERA_UPDATE_RATE_HZ, "rgb_encoding": "rgb8",
            "depth_encoding": "32FC1", "optical_frame": OPTICAL_FRAME_NAME,
            "use_sim_time": True,
        },
        "raw_data_label": "NOT_GROUND_TRUTH", "dataset_training_paths_created": False,
        "source_sample_plan": source_sample_provenance,
        "raw_message_paths": raw_paths, "raw_message_history_paths": {},
        "raw_message_history_complete": False,
        "set_attempt_count": 0, "capture_attempt_count": 0,
        "validation": {}, "rejection_reasons": [],
    }
    write_json_atomic(manifest_path, manifest)

    node, initialized = None, False
    set_attempt_count = capture_attempt_count = 0

    def persist_raw_row(key: str, row: dict) -> str:
        indexed_paths = raw_index_paths.setdefault(key, {})
        callback_index = int(row["index"])
        if callback_index not in indexed_paths:
            payload = serialize_message(row["message"])
            indexed_paths[callback_index] = save_diagnostic_cdr(
                output_dir, key, callback_index, row["header_stamp_ns"], payload,
            )
        return indexed_paths[callback_index]

    raw_persistence_errors: list[dict] = []

    def persist_all_available_raw() -> bool:
        if node is None:
            return False
        raw_persistence_errors.clear()
        for key, history in node.histories.items():
            for row in history:
                try:
                    persist_raw_row(key, row)
                except Exception as error:
                    raw_persistence_errors.append({
                        "topic_key": key, "callback_index": row["index"],
                        "error_type": type(error).__name__, "message": str(error),
                    })
        for key in node.histories:
            indexed_paths = raw_index_paths.get(key, {})
            ordered_indexes = sorted(indexed_paths)
            raw_history_paths[key] = [indexed_paths[index] for index in ordered_indexes]
            selected_index = selected_callback_indexes.get(key)
            if selected_index in indexed_paths:
                raw_paths[key] = indexed_paths[selected_index]
            elif ordered_indexes:
                raw_paths[key] = indexed_paths[ordered_indexes[-1]]
        persisted_counts = {
            key: len(raw_index_paths.get(key, {})) for key in node.callback_counts
        }
        complete = diagnostic_raw_history_is_complete(
            node.callback_counts, persisted_counts, node.history_overflow,
        )
        manifest["raw_message_history_persistence_errors"] = list(raw_persistence_errors)
        return complete and not raw_persistence_errors

    class DiagnosticNode(Node):
        HISTORY_LIMIT, CLOCK_LIMIT, EVENT_LIMIT = 128, 20000, 4096

        def __init__(self) -> None:
            super().__init__(
                "suitcase_dataset_single_shot_diagnostic",
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            self.bridge = CvBridge()
            self.phase, self.pose_epoch_id = "PRE_SET", epoch_id
            self.histories = {key: deque(maxlen=self.HISTORY_LIMIT) for key in (
                "rgb", "depth", "rgb_camera_info", "depth_camera_info",
            )}
            self.history_overflow = {key: 0 for key in self.histories}
            self.callback_counts = {key: 0 for key in self.histories}
            self.clock_history = deque(maxlen=self.CLOCK_LIMIT)
            self.clock_overflow, self.events, self.event_overflow = 0, [], 0
            self.create_subscription(Image, args.image_topic, self._rgb, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self._depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info_topic, self._rgb_info, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.depth_camera_info_topic, self._depth_info, qos_profile_sensor_data)
            self.create_subscription(Clock, "/clock", self._clock_callback, qos_profile_system_default)
            self.set_state = self.create_client(SetEntityState, args.set_state_service)
            self.get_state = self.create_client(GetEntityState, args.get_state_service)

        @staticmethod
        def stamp_ns(stamp) -> int:
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def node_clock_ns(self) -> int:
            stage = "get_clock_lookup"
            details = {"instance_get_clock_shadow": "get_clock" in self.__dict__}
            try:
                getter = self.get_clock
                details.update({
                    "getter_type": type(getter).__name__,
                    "getter_callable": callable(getter),
                    "class_get_clock_type": type(getattr(type(self), "get_clock", None)).__name__,
                })
                stage = "get_clock_call"
                node_clock = getter()
                details["clock_type"] = type(node_clock).__name__
                stage = "clock_now_lookup"
                now_method = node_clock.now
                details.update({
                    "now_type": type(now_method).__name__,
                    "now_callable": callable(now_method),
                })
                stage = "clock_now_call"
                node_time = now_method()
                stage = "clock_nanoseconds_read"
                return int(node_time.nanoseconds)
            except Exception as error:
                details.update({
                    "failure_stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                raise RuntimeError(
                    "ROS clock snapshot failed: " + json.dumps(details, sort_keys=True)
                ) from error

        def time_snapshot(self) -> dict:
            clock = self.clock_history[-1] if self.clock_history else None
            return {
                "monotonic_ns": time.monotonic_ns(),
                "node_clock_ns": self.node_clock_ns(),
                "latest_clock_stamp_ns": clock["stamp_ns"] if clock else None,
                "latest_clock_received_monotonic_ns": clock["received_monotonic_ns"] if clock else None,
            }

        def event(self, name: str, **fields) -> dict:
            row = {"event": name, "phase": self.phase, **self.time_snapshot(), **fields}
            if not append_bounded_diagnostic_event(self.events, row, self.EVENT_LIMIT):
                self.event_overflow += 1
            return row

        def _record(self, key: str, message) -> None:
            callback_entry_ns = time.monotonic_ns()
            history = self.histories[key]
            if len(history) == history.maxlen:
                self.history_overflow[key] += 1
            latest_clock = self.clock_history[-1] if self.clock_history else None
            history.append({
                "index": self.callback_counts[key], "pose_epoch_id": self.pose_epoch_id,
                "phase": self.phase, "callback_entry_monotonic_ns": callback_entry_ns,
                "node_clock_ns_at_callback": self.node_clock_ns(),
                "latest_clock_stamp_ns_at_callback": latest_clock["stamp_ns"] if latest_clock else None,
                "latest_clock_received_monotonic_ns_at_callback": (
                    latest_clock["received_monotonic_ns"] if latest_clock else None
                ),
                "header_stamp_ns": self.stamp_ns(message.header.stamp),
                "header_stamp_s": stamp_seconds(message.header.stamp),
                "frame_id": message.header.frame_id,
                "encoding": getattr(message, "encoding", None),
                "width": getattr(message, "width", None), "height": getattr(message, "height", None),
                "message": message,
            })
            self.callback_counts[key] += 1

        def _rgb(self, message: Image) -> None: self._record("rgb", message)
        def _depth(self, message: Image) -> None: self._record("depth", message)
        def _rgb_info(self, message: CameraInfo) -> None: self._record("rgb_camera_info", message)
        def _depth_info(self, message: CameraInfo) -> None: self._record("depth_camera_info", message)

        def _clock_callback(self, message: Clock) -> None:
            if len(self.clock_history) == self.clock_history.maxlen:
                self.clock_overflow += 1
            self.clock_history.append({
                "stamp_ns": self.stamp_ns(message.clock), "stamp_s": stamp_seconds(message.clock),
                "received_monotonic_ns": time.monotonic_ns(),
            })

        def spin_future(self, future):
            deadline = time.monotonic_ns() + int(args.service_timeout_s * 1e9)
            while rclpy.ok() and not future.done() and time.monotonic_ns() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, (deadline - time.monotonic_ns()) / 1e9)))
            if not future.done():
                raise TimeoutError("Gazebo service call timed out")
            result = future.result()
            if result is None:
                raise RuntimeError("Gazebo service returned no result")
            return result

        def get_entity(self, name: str):
            request = GetEntityState.Request()
            request.name, request.reference_frame = name, "world"
            return self.spin_future(self.get_state.call_async(request))

        def set_camera(self, sample: PilotSample):
            request = SetEntityState.Request()
            populate_set_entity_state_request(request, CAMERA_ENTITY, sample)
            return self.spin_future(self.set_state.call_async(request))

        def history_metadata(self) -> dict:
            return {key: [{k: v for k, v in row.items() if k != "message"} for row in rows]
                    for key, rows in self.histories.items()}

    try:
        camera_model_path = SIMULATION_SHARE / "models/dataset_camera/model.sdf"
        manifest["camera_model_sha256"] = sha256_file(camera_model_path)
        manifest["configuration_sha256"] = canonical_sha256({
            "world_sha256": world_hash, "mesh_sha256": mesh_hash,
            "camera_model_sha256": manifest["camera_model_sha256"],
            "camera_configuration": manifest["camera_configuration"], "seed": args.seed,
            "sample_plan": asdict(source_sample) if source_sample else (
                "one deterministic seeded view from existing azimuth intervals; no retry"
            ),
            "source_sample_manifest_sha256": (
                source_sample_provenance["source_manifest_sha256"]
                if source_sample_provenance else None
            ),
        })
        target_from_world = target_model_pose(Path(args.world).resolve())
        target_center = suitcase_visual_center(*target_from_world, mesh)
        if source_sample is not None:
            sample = source_sample
        else:
            rng = np.random.default_rng(args.seed)
            azimuth, scenario_id = _sample_azimuth_groups(
                1, TRAIN_AZIMUTH_INTERVALS_DEG + VALIDATION_AZIMUTH_INTERVALS_DEG,
                "diagnostic", rng,
            )[0]
            distance = float(rng.uniform(*CAMERA_DISTANCE_RANGE_M))
            camera_height = float(rng.uniform(*CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M))
            angle = math.radians(azimuth)
            camera_position = (
                float(target_center[0] + distance * math.cos(angle)),
                float(target_center[1] + distance * math.sin(angle)),
                float(target_center[2] + camera_height),
            )
            orientation = camera_link_orientation(camera_position, target_center)
            rotation = quaternion_matrix_xyzw(orientation)
            sample = PilotSample(
                sample_id=f"diagnostic_seed{args.seed}_single", split="diagnostic",
                scenario_id=scenario_id, camera_x_m=camera_position[0],
                camera_y_m=camera_position[1], camera_z_m=camera_position[2],
                camera_yaw_deg=math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))) % 360.0,
                camera_pitch_deg=math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))),
                camera_link_quaternion_xyzw=orientation,
                azimuth_deg=float(azimuth), camera_distance_m=distance,
                camera_height_above_target_m=camera_height, random_seed=args.seed,
            )
        validate_camera_clearance(Path(args.world).resolve(), [sample])
        manifest["requested_capture_count"], manifest["sample"] = 1, asdict(sample)
        rclpy.init(args=None)
        initialized = True
        node = DiagnosticNode()
        for client, service_name in ((node.set_state, args.set_state_service),
                                     (node.get_state, args.get_state_service)):
            if not client.wait_for_service(timeout_sec=args.service_timeout_s):
                raise TimeoutError(f"Required Gazebo service is unavailable: {service_name}")

        def wait_for_clock_boundary(label: str, target_stamp_ns=None, strict_greater=False) -> dict:
            started_ns = time.monotonic_ns()
            deadline_ns = started_ns + int(args.service_timeout_s * 1e9)
            while rclpy.ok():
                latest = node.clock_history[-1] if node.clock_history else None
                satisfied = bool(latest) and (
                    target_stamp_ns is None
                    or (latest["stamp_ns"] > target_stamp_ns if strict_greater
                        else latest["stamp_ns"] >= target_stamp_ns)
                )
                if satisfied or time.monotonic_ns() >= deadline_ns:
                    break
                remaining_s = max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9)
                rclpy.spin_once(node, timeout_sec=min(0.05, remaining_s))
            latest = node.clock_history[-1] if node.clock_history else None
            satisfied = bool(latest) and (
                target_stamp_ns is None
                or (latest["stamp_ns"] > target_stamp_ns if strict_greater
                    else latest["stamp_ns"] >= target_stamp_ns)
            )
            result = {
                "label": label,
                "target_stamp_ns": target_stamp_ns,
                "strict_greater": bool(strict_greater),
                "satisfied": bool(satisfied),
                "latest_clock_stamp_ns": latest["stamp_ns"] if latest else None,
                "latest_clock_received_monotonic_ns": latest["received_monotonic_ns"] if latest else None,
                "wait_started_monotonic_ns": started_ns,
                "wait_finished_monotonic_ns": time.monotonic_ns(),
                "timeout_s": args.service_timeout_s,
            }
            node.event("CLOCK_BOUNDARY_WAIT", **result)
            return result

        def timed_get(name: str, label: str):
            started = node.time_snapshot()
            response = node.get_entity(name)
            finished = node.time_snapshot()
            pose_error = None
            try:
                position, orientation = entity_pose(response)
            except (ValueError, TypeError) as error:
                position, orientation = None, None
                pose_error = str(error)
            during_call = [row for row in node.clock_history
                           if started["monotonic_ns"] <= row["received_monotonic_ns"]
                           <= finished["monotonic_ns"]]
            response_stamp_ns = node.stamp_ns(response.header.stamp)
            event = node.event(
                "GET_ENTITY_STATE", label=label, entity=name, call_started=started,
                call_finished=finished, service_success=bool(response.success),
                response_header_stamp_ns=response_stamp_ns,
                response_header_stamp_s=stamp_seconds(response.header.stamp),
                response_pose_position_m=position, response_pose_orientation_xyzw=orientation,
                response_pose_validation_error=pose_error,
                clock_stamps_received_during_call_ns=[row["stamp_ns"] for row in during_call],
                response_stamp_seen_on_clock_during_call=any(
                    row["stamp_ns"] == response_stamp_ns for row in during_call
                ),
            )
            if not response.success:
                raise RuntimeError(f"GetEntityState failed for {name} during {label}")
            if pose_error is not None:
                raise RuntimeError(f"GetEntityState returned invalid pose for {name}: {pose_error}")
            return event, response

        def wait_for_entity_ready(name: str) -> None:
            """Wait until Gazebo has spawned an entity before reading its pose."""
            deadline_ns = time.monotonic_ns() + int(args.service_timeout_s * 1e9)
            service_available = False
            attempts = 0
            while rclpy.ok() and time.monotonic_ns() < deadline_ns:
                remaining_s = max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9)
                service_available = node.get_state.wait_for_service(
                    timeout_sec=min(0.25, remaining_s),
                )
                if service_available:
                    response = node.get_entity(name)
                    attempts += 1
                    if response.success:
                        node.event(
                            "ENTITY_READY", entity=name, attempts=attempts,
                            wait_elapsed_s=(time.monotonic_ns() - (deadline_ns - int(args.service_timeout_s * 1e9))) / 1e9,
                        )
                        return
                    node.event("ENTITY_NOT_READY", entity=name, attempts=attempts)
                else:
                    node.event("ENTITY_STATE_SERVICE_NOT_READY", entity=name)
                rclpy.spin_once(node, timeout_sec=0.05)
            raise TimeoutError(
                f"Timed out waiting for Gazebo entity {name} to spawn "
                f"within {args.service_timeout_s:.1f} s"
            )

        clock_lower_wait = wait_for_clock_boundary("before_first_GET")
        wait_for_entity_ready(TARGET_ENTITY)
        wait_for_entity_ready(CAMERA_ENTITY)
        node.phase = "PRE_SET"
        target_initial_event, target_initial_response = timed_get(TARGET_ENTITY, "target_initial")
        camera_initial_event, camera_initial_response = timed_get(CAMERA_ENTITY, "camera_before_set")
        target_initial = entity_pose(target_initial_response)
        node.phase = "SETTING_CAMERA"
        set_started = node.time_snapshot()
        set_attempt_count += 1
        manifest["set_attempt_count"] = set_attempt_count
        if set_attempt_count != 1:
            raise RuntimeError("Diagnostic mode permits exactly one camera SET attempt")
        set_response = node.set_camera(sample)
        set_finished = node.time_snapshot()
        set_success = bool(set_response.success)
        set_event = node.event(
            "SET_CAMERA_RESULT", entity=CAMERA_ENTITY, pose_epoch_id=epoch_id,
            requested_pose_position_m=(sample.camera_x_m, sample.camera_y_m, sample.camera_z_m),
            requested_pose_orientation_xyzw=camera_link_orientation(
                (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m), target_center,
            ), call_started=set_started, call_finished=set_finished,
            service_success=set_success,
        )
        if not set_success:
            raise RuntimeError("SetEntityState returned success=false")
        node.phase = "SET_ACKNOWLEDGED_SETTLING"
        settle_end = time.monotonic_ns() + int(args.settle_s * 1e9)
        while rclpy.ok() and time.monotonic_ns() < settle_end:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, (settle_end - time.monotonic_ns()) / 1e9)))

        node.phase = "POSE_READBACK"
        camera_pre_event, camera_pre_response = timed_get(CAMERA_ENTITY, "camera_before_acquisition")
        target_pre_event, target_pre_response = timed_get(TARGET_ENTITY, "target_before_acquisition")
        camera_pre = entity_pose(camera_pre_response)
        target_pre = entity_pose(target_pre_response)
        camera_pre_stamp_ns = node.stamp_ns(camera_pre_response.header.stamp)
        target_pre_stamp_ns = node.stamp_ns(target_pre_response.header.stamp)
        try:
            validate_camera_sample_pose(sample, *camera_pre, suitcase_visual_center(*target_pre, mesh))
            pose_matches_plan, pose_error = True, None
        except (ValueError, RuntimeError) as error:
            pose_matches_plan, pose_error = False, str(error)
        target_initial_pre_delta = capture_pose_delta(*target_initial, *target_pre)
        manifest["events_before_capture"] = {
            "target_initial": target_initial_event, "camera_before_set": camera_initial_event,
            "set": set_event, "camera_before_acquisition": camera_pre_event,
            "target_before_acquisition": target_pre_event,
        }
        manifest["set_readback_validation"] = {
            "set_success": set_success, "planned_pose_confirmed": pose_matches_plan,
            "detail": pose_error,
        }

        capture_attempt_count += 1
        manifest["capture_attempt_count"] = capture_attempt_count
        if capture_attempt_count != 1:
            raise RuntimeError("Diagnostic mode permits exactly one RGB-D acquisition window")
        capture_start = node.time_snapshot()
        capture_start_ns = capture_start["monotonic_ns"]
        node.phase = "CAPTURING_AFTER_POSE_READBACK"

        def find_candidates() -> list[dict]:
            eligible = {
                key: [row for row in rows if row["phase"] == node.phase
                      and row["callback_entry_monotonic_ns"] >= capture_start_ns]
                for key, rows in node.histories.items()
            }
            tolerance_ns = int(args.max_sync_delta_s * 1e9)
            found = []
            for rgb in eligible["rgb"]:
                for depth in eligible["depth"]:
                    if abs(rgb["header_stamp_ns"] - depth["header_stamp_ns"]) > tolerance_ns:
                        continue
                    rgb_infos = [r for r in eligible["rgb_camera_info"]
                                 if abs(rgb["header_stamp_ns"] - r["header_stamp_ns"]) <= tolerance_ns]
                    depth_infos = [r for r in eligible["depth_camera_info"]
                                   if abs(depth["header_stamp_ns"] - r["header_stamp_ns"]) <= tolerance_ns]
                    if not rgb_infos or not depth_infos:
                        continue
                    rgb_info = min(rgb_infos, key=lambda r: abs(rgb["header_stamp_ns"] - r["header_stamp_ns"]))
                    depth_info = min(depth_infos, key=lambda r: abs(depth["header_stamp_ns"] - r["header_stamp_ns"]))
                    rows = {"rgb": rgb, "depth": depth,
                            "rgb_camera_info": rgb_info, "depth_camera_info": depth_info}
                    found.append({
                        "candidate_id": f"{sample.sample_id}_{rgb['header_stamp_ns']}_{rgb['index']}_{depth['index']}",
                        "pose_epoch_id": epoch_id, "rows": rows,
                        "selected_monotonic_ns": max(r["callback_entry_monotonic_ns"] for r in rows.values()),
                    })
            return sorted(found, key=lambda r: r["selected_monotonic_ns"])

        deadline_ns = capture_start_ns + int(args.frame_timeout_s * 1e9)
        candidates = find_candidates()
        while rclpy.ok() and not candidates and time.monotonic_ns() < deadline_ns:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9)))
            candidates = find_candidates()
        selected = candidates[0] if candidates else None
        candidate_set_history = [{
            "candidate_id": candidate["candidate_id"],
            "pose_epoch_id": candidate["pose_epoch_id"],
            "selected_monotonic_ns": candidate["selected_monotonic_ns"],
            "members": {key: {
                "callback_index": row["index"], "header_stamp_ns": row["header_stamp_ns"],
                "callback_entry_monotonic_ns": row["callback_entry_monotonic_ns"],
                "frame_id": row["frame_id"],
            } for key, row in candidate["rows"].items()},
        } for candidate in candidates]
        selected_event = node.event(
            "RGBD_SET_SELECTED" if selected else "RGBD_SET_NOT_FOUND",
            pose_epoch_id=epoch_id, candidate_id=selected["candidate_id"] if selected else None,
            candidate_count=len(candidates), selection_policy="earliest complete post-readback set",
        )
        if selected:
            selected_callback_indexes.update({
                key: int(row["index"]) for key, row in selected["rows"].items()
            })
            for key, row in selected["rows"].items():
                raw_paths[key] = persist_raw_row(key, row)
        node.phase = "POST_ACQUISITION_STATE"
        camera_post_event, camera_post_response = timed_get(CAMERA_ENTITY, "camera_after_acquisition")
        target_post_event, target_post_response = timed_get(TARGET_ENTITY, "target_after_acquisition")
        camera_post, target_post = entity_pose(camera_post_response), entity_pose(target_post_response)
        camera_static = capture_pose_delta(*camera_pre, *camera_post)
        target_static = capture_pose_delta(*target_pre, *target_post)
        target_total = capture_pose_delta(*target_initial, *target_post)
        camera_post_stamp_ns = node.stamp_ns(camera_post_response.header.stamp)
        target_post_stamp_ns = node.stamp_ns(target_post_response.header.stamp)
        state_header_stamps_ns = {
            "target_initial": target_initial_event["response_header_stamp_ns"],
            "camera_before_set": camera_initial_event["response_header_stamp_ns"],
            "camera_before_acquisition": camera_pre_stamp_ns,
            "target_before_acquisition": target_pre_stamp_ns,
            "camera_after_acquisition": camera_post_stamp_ns,
            "target_after_acquisition": target_post_stamp_ns,
        }
        frame_header_stamps_ns = ({
            key: int(row["header_stamp_ns"])
            for key, row in selected["rows"].items()
        } if selected else {})
        headers_to_bracket_ns = {**state_header_stamps_ns, **frame_header_stamps_ns}
        clock_upper_target_ns = max(headers_to_bracket_ns.values()) if headers_to_bracket_ns else None
        clock_upper_wait = wait_for_clock_boundary(
            "after_last_GET_and_frozen_headers", clock_upper_target_ns, strict_greater=True,
        ) if clock_upper_target_ns is not None else {
            "label": "after_last_GET_and_frozen_headers", "target_stamp_ns": None,
            "strict_greater": True, "satisfied": False, "latest_clock_stamp_ns": None,
            "latest_clock_received_monotonic_ns": None, "timeout_s": args.service_timeout_s,
        }

        raw_history_complete = persist_all_available_raw()
        manifest["raw_message_paths"] = dict(raw_paths)
        manifest["raw_message_history_paths"] = {
            key: list(paths) for key, paths in raw_history_paths.items()
        }
        manifest["raw_message_history_counts"] = {
            key: len(paths) for key, paths in raw_history_paths.items()
        }
        manifest["raw_message_history_complete"] = raw_history_complete

        clock_rows = list(node.clock_history)
        clock_stamps = [row["stamp_ns"] for row in clock_rows]
        clock_monotonic = all(a <= b for a, b in zip(clock_stamps, clock_stamps[1:]))
        clock_progress = any(a < b for a, b in zip(clock_stamps, clock_stamps[1:]))
        clock_witnesses = {}
        for header_name, header_stamp_ns in headers_to_bracket_ns.items():
            lower = [stamp for stamp in clock_stamps if stamp <= header_stamp_ns]
            upper = [stamp for stamp in clock_stamps if stamp >= header_stamp_ns]
            clock_witnesses[header_name] = {
                "header_stamp_ns": header_stamp_ns,
                "observed_lower_clock_stamp_ns": max(lower) if lower else None,
                "observed_upper_clock_stamp_ns": min(upper) if upper else None,
                "has_lower_witness": bool(lower), "has_upper_witness": bool(upper),
                "bracketed_without_interpolation": bool(lower and upper),
            }
        clock_headers_bracketed = bool(headers_to_bracket_ns) and all(
            witness["bracketed_without_interpolation"] for witness in clock_witnesses.values()
        )
        clock_bracket_pass = bool(
            clock_lower_wait.get("satisfied")
            and clock_upper_wait.get("satisfied")
            and clock_monotonic and clock_progress and clock_headers_bracketed
        )
        histories = node.history_metadata()
        manifest.update({
            "set_attempt_count": set_attempt_count,
            "capture_attempt_count": capture_attempt_count,
            "capture_started": capture_start, "selection_event": selected_event,
            "candidate_set_history": candidate_set_history,
            "candidate_histories": histories, "clock_history": clock_rows,
            "event_trace": node.events, "callback_counts": node.callback_counts,
            "history_overflow_counts": node.history_overflow,
            "clock_history_overflow_count": node.clock_overflow,
            "event_trace_overflow_count": node.event_overflow,
            "raw_message_paths": dict(raw_paths),
            "raw_message_history_paths": {
                key: list(paths) for key, paths in raw_history_paths.items()
            },
            "raw_message_history_counts": {
                key: len(paths) for key, paths in raw_history_paths.items()
            },
            "raw_message_history_complete": raw_history_complete,
            "camera_state_after_acquisition": camera_post_event,
            "target_state_after_acquisition": target_post_event,
            "time_sources": {
                "use_sim_time": bool(node.get_parameter("use_sim_time").value),
                "clock_topic": "/clock", "clock_samples": len(clock_rows),
                "clock_stamps_non_decreasing": clock_monotonic,
                "clock_progress_observed": clock_progress,
                "clock_reset_detected": not clock_monotonic,
                "lower_boundary_wait_before_first_get": clock_lower_wait,
                "upper_boundary_wait_after_frozen_headers": clock_upper_wait,
                "header_clock_witnesses": clock_witnesses,
                "header_clock_bracketing_uses_interpolation": False,
                "header_clock_bracketing_pass": clock_bracket_pass,
            },
            "camera_stability": {
                "position_delta_m": camera_static[0], "orientation_delta_rad": camera_static[1],
                "limits": [CAPTURE_POSE_MAX_POSITION_DELTA_M, CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD],
                "pass": camera_static[0] <= CAPTURE_POSE_MAX_POSITION_DELTA_M
                        and camera_static[1] <= CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD,
            },
            "target_stability": {
                "initial_to_pre_delta": target_initial_pre_delta,
                "pre_to_post_delta": target_static, "initial_to_post_delta": target_total,
                "limits": [CAPTURE_POSE_MAX_POSITION_DELTA_M, CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD],
                "pass": max(target_initial_pre_delta[0], target_static[0], target_total[0])
                        <= CAPTURE_POSE_MAX_POSITION_DELTA_M
                        and max(target_initial_pre_delta[1], target_static[1], target_total[1])
                        <= CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD,
            },
        })

        rows = selected["rows"] if selected else {}
        if selected:
            rgb, depth = rows["rgb"]["message"], rows["depth"]["message"]
            rgb_info, depth_info = rows["rgb_camera_info"]["message"], rows["depth_camera_info"]["message"]
        else:
            rgb = depth = rgb_info = depth_info = None
        manifest["raw_message_paths"] = dict(raw_paths)
        manifest["raw_message_history_paths"] = {
            key: list(paths) for key, paths in raw_history_paths.items()
        }

        expected_focal_px = CAMERA_WIDTH / (2.0 * math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))

        def info_matches_approved_optics(info) -> bool:
            return (
                float(info.k[0]) > 0.0 and float(info.k[4]) > 0.0
                and abs(float(info.k[0]) - expected_focal_px) <= expected_focal_px * 0.01
                and abs(float(info.k[4]) - expected_focal_px) <= expected_focal_px * 0.01
                and abs(float(info.k[2]) - CAMERA_WIDTH / 2.0) <= 1.5
                and abs(float(info.k[5]) - CAMERA_HEIGHT / 2.0) <= 1.5
                and info.binning_x == info.binning_y == 0
                and info.roi.width == info.roi.height == 0
            )

        checks = {
            "exactly_one_set_and_capture_attempt": set_attempt_count == 1 and capture_attempt_count == 1,
            "set_service_succeeded": set_success,
            "camera_readback_matches_planned_pose": pose_matches_plan,
            "selected_complete_rgbd_camera_info_set": selected is not None,
            "selected_messages_assigned_to_pose_epoch": bool(selected) and all(
                row["pose_epoch_id"] == epoch_id and row["phase"] == "CAPTURING_AFTER_POSE_READBACK"
                for row in rows.values()
            ),
            "clock_series_monotonic_with_progress": clock_monotonic and clock_progress,
            "camera_static_pre_to_post": manifest["camera_stability"]["pass"],
            "target_static_initial_pre_post": manifest["target_stability"]["pass"],
            "clock_history_not_overflowed": node.clock_overflow == 0,
            "message_histories_not_overflowed": all(v == 0 for v in node.history_overflow.values()),
            "event_trace_not_overflowed": node.event_overflow == 0,
            "raw_message_history_complete": raw_history_complete,
            "clock_tick_observed_before_first_get": bool(clock_lower_wait.get("satisfied")),
            "clock_tick_strictly_after_max_frozen_header": bool(clock_upper_wait.get("satisfied")),
            "all_pose_and_frame_stamps_bracketed_by_observed_clock_ticks": clock_bracket_pass,
        }
        if selected:
            stamps = {key: row["header_stamp_ns"] for key, row in rows.items()}
            state_stamps = [target_initial_event["response_header_stamp_ns"],
                            camera_initial_event["response_header_stamp_ns"], camera_pre_stamp_ns,
                            target_pre_stamp_ns, camera_post_stamp_ns, target_post_stamp_ns]
            checks.update({
                "rgb_depth_within_sync_limit": abs(stamps["rgb"] - stamps["depth"]) <= int(args.max_sync_delta_s * 1e9),
                "rgb_camera_info_within_sync_limit": abs(stamps["rgb"] - stamps["rgb_camera_info"]) <= int(args.max_sync_delta_s * 1e9),
                "depth_camera_info_within_sync_limit": abs(stamps["depth"] - stamps["depth_camera_info"]) <= int(args.max_sync_delta_s * 1e9),
                "pose_and_frame_stamps_nonzero": all(
                    stamp > 0 for stamp in (*state_stamps, *stamps.values())
                ),
                "frame_stamps_bracketed_by_camera_pose_reads": min(stamps.values()) > camera_pre_stamp_ns
                    and max(stamps.values()) <= camera_post_stamp_ns,
                "frame_ids_match": len({row["frame_id"] for row in rows.values()}) == 1,
                "expected_encodings": rgb.encoding == "rgb8" and depth.encoding == "32FC1",
                "dimensions_match_640x360": all(
                    (msg.width, msg.height) == (CAMERA_WIDTH, CAMERA_HEIGHT)
                    for msg in (rgb, depth, rgb_info, depth_info)
                ),
                "camera_info_calibration_matches": camera_infos_share_calibration(rgb_info, depth_info),
                "camera_info_matches_approved_optics": (
                    info_matches_approved_optics(rgb_info)
                    and info_matches_approved_optics(depth_info)
                ),
            })
            image_stamp_ns = max(stamps["rgb"], stamps["depth"])
            legacy_delta_s = (image_stamp_ns - camera_pre_stamp_ns) / 1e9
            manifest["legacy_0_1s_metric"] = {
                "limit_s": DEFAULT_MAX_CAMERA_STAMP_DELTA_S, "delta_s": legacy_delta_s,
                "within_limit": 0 <= legacy_delta_s <= DEFAULT_MAX_CAMERA_STAMP_DELTA_S,
                "role": "reported metric; not relaxed or removed; acceptance additionally requires exact clock correspondence, epoch and bracketing",
            }
            manifest["camera_info_parameters"] = {
                "rgb": camera_info_parameters(rgb_info), "depth": camera_info_parameters(depth_info),
                "configuration_sha256": camera_info_configuration_sha256(rgb_info, depth_info),
            }
            manifest["selected_candidate"] = {
                "candidate_id": selected["candidate_id"], "pose_epoch_id": epoch_id,
                "callback_receipt_metrics": {
                    key: {
                        "callback_entry_monotonic_ns": row["callback_entry_monotonic_ns"],
                        "seconds_after_post_readback_acquisition_start": (
                            row["callback_entry_monotonic_ns"] - capture_start_ns
                        ) / 1e9,
                        "above_delivery_latency_warning": (
                            row["callback_entry_monotonic_ns"] - capture_start_ns
                        ) / 1e9 > args.delivery_latency_warning_s,
                        "latest_clock_stamp_at_callback_ns": row["latest_clock_stamp_ns_at_callback"],
                        "callback_receipt_is_not_image_execution_time": True,
                    } for key, row in rows.items()
                },
                "legacy_delivery_latency_warning_s": args.delivery_latency_warning_s,
                "messages": {key: {k: v for k, v in row.items() if k != "message"}
                             for key, row in rows.items()},
            }
        else:
            checks.update({key: False for key in (
                "all_pose_and_frame_stamps_bracketed_by_observed_clock_ticks", "rgb_depth_within_sync_limit",
                "rgb_camera_info_within_sync_limit", "depth_camera_info_within_sync_limit",
                "pose_and_frame_stamps_nonzero", "frame_stamps_bracketed_by_camera_pose_reads", "frame_ids_match",
                "expected_encodings", "dimensions_match_640x360", "camera_info_calibration_matches",
                "camera_info_matches_approved_optics",
            )})
            manifest["rejection_reasons"].append("NO_COMPLETE_POST_READBACK_RGB_DEPTH_CAMERAINFO_SET")
        manifest["validation"] = checks
        manifest["rejection_reasons"].extend(key.upper() for key, value in checks.items() if not value)

        color_bgr = depth_m = None
        decoded_selected = False
        if selected:
            try:
                color_bgr = node.bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
                depth_m = depth_to_meters(
                    node.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough"), depth.encoding,
                )
                if not cv2.imwrite(str(output_dir / "decoded" / "rgb.png"), color_bgr):
                    raise IOError("Could not write decoded RGB diagnostic image")
                np.save(output_dir / "decoded" / "depth_m.npy", depth_m.astype(np.float32, copy=False))
                decoded_selected = True
            except (ValueError, RuntimeError, cv2.error, OSError) as error:
                manifest["rejection_reasons"].append(f"SELECTED_RGB_DEPTH_DECODE_FAILED: {error}")
        else:
            for key, history in node.histories.items():
                if not history:
                    continue
                message = history[-1]["message"]
                try:
                    if key == "rgb" and message.encoding == "rgb8":
                        latest_rgb = node.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                        cv2.imwrite(str(output_dir / "decoded" / "unpaired_latest_rgb.png"), latest_rgb)
                    elif key == "depth":
                        latest_depth = node.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
                        np.save(output_dir / "decoded" / "unpaired_latest_depth_raw.npy", latest_depth)
                except (ValueError, RuntimeError, cv2.error, OSError) as error:
                    manifest.setdefault("unpaired_decode_errors", {})[key] = str(error)
        checks["selected_rgb_depth_decoded"] = decoded_selected if selected else True
        if selected and not decoded_selected:
            manifest["validation"]["selected_rgb_depth_decoded"] = False

        preview_status = "NOT_ATTEMPTED"
        geometry_requirements = (
            "exactly_one_set_and_capture_attempt", "set_service_succeeded",
            "camera_readback_matches_planned_pose",
            "selected_complete_rgbd_camera_info_set", "selected_messages_assigned_to_pose_epoch",
            "clock_series_monotonic_with_progress", "raw_message_history_complete",
            "all_pose_and_frame_stamps_bracketed_by_observed_clock_ticks",
            "clock_tick_observed_before_first_get",
            "clock_tick_strictly_after_max_frozen_header",
            "frame_stamps_bracketed_by_camera_pose_reads", "camera_static_pre_to_post",
            "target_static_initial_pre_post", "rgb_depth_within_sync_limit",
            "rgb_camera_info_within_sync_limit", "depth_camera_info_within_sync_limit",
            "frame_ids_match", "expected_encodings", "dimensions_match_640x360",
            "camera_info_calibration_matches",
            "camera_info_matches_approved_optics",
            "selected_rgb_depth_decoded",
        )
        if selected and all(checks[key] for key in geometry_requirements):
            try:
                optical_position, optical_q = camera_optical_pose(*camera_pre)
                vertices = mesh.vertices.copy()
                vertices[:, 2] -= 0.05
                points_camera = pose_points_to_camera(
                    vertices, *target_pre, (0.0, 0.0, 0.0), optical_position, optical_q,
                )
                intrinsics = (float(rgb_info.k[0]), float(rgb_info.k[4]),
                              float(rgb_info.k[2]), float(rgb_info.k[5]))
                distortion = tuple(float(v) for v in rgb_info.d)
                full_bbox = projected_bbox(points_camera, intrinsics, CAMERA_WIDTH, CAMERA_HEIGHT, distortion=distortion)
                visibility, mask, expected_depth = depth_visibility_products(
                    points_camera, mesh.triangles, intrinsics, depth_m, distortion,
                )
                visible_bbox = visible_bbox_from_mask(mask, CAMERA_WIDTH, CAMERA_HEIGHT)
                normalized_visible_bbox = yolo_bbox(visible_bbox, CAMERA_WIDTH, CAMERA_HEIGHT)
                bbox_valid = (
                    np.all(np.isfinite(normalized_visible_bbox))
                    and normalized_visible_bbox[2] > 0.0 and normalized_visible_bbox[3] > 0.0
                    and normalized_visible_bbox[0] - normalized_visible_bbox[2] / 2.0 >= 0.0
                    and normalized_visible_bbox[1] - normalized_visible_bbox[3] / 2.0 >= 0.0
                    and normalized_visible_bbox[0] + normalized_visible_bbox[2] / 2.0 <= 1.0
                    and normalized_visible_bbox[1] + normalized_visible_bbox[3] / 2.0 <= 1.0
                )
                visibility_valid = visibility["visible_fraction"] >= MIN_VISIBLE_FRACTION and bbox_valid
                preview = color_bgr.copy()
                x1, y1, x2, y2 = full_bbox
                cv2.rectangle(preview, (round(x1), round(y1)), (round(x2), round(y2)), (0, 165, 255), 1)
                x1, y1, x2, y2 = visible_bbox
                cv2.rectangle(preview, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
                cv2.putText(preview, "DIAGNOSTIC ONLY — NOT GROUND TRUTH", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
                if not cv2.imwrite(str(output_dir / "preview" / "rgb_bbox_diagnostic.jpg"), preview):
                    raise IOError("Could not write diagnostic RGB overlay")
                if not cv2.imwrite(
                    str(output_dir / "preview" / "depth_alignment_diagnostic.png"),
                    depth_diagnostic_preview(depth_m, expected_depth, mask),
                ):
                    raise IOError("Could not write diagnostic depth preview")
                manifest["depth_visibility"] = visibility
                manifest["validation"]["depth_confirmed_visible_bbox_valid"] = bool(visibility_valid)
                manifest["diagnostic_bboxes_xyxy_px"] = {
                    "full_geometry": list(full_bbox), "visible_surface": list(visible_bbox),
                    "visible_bbox_yolo_xywh_normalized_diagnostic_only": normalized_visible_bbox,
                    "minimum_visible_fraction": MIN_VISIBLE_FRACTION,
                    "visible_fraction_gate_pass": visibility["visible_fraction"] >= MIN_VISIBLE_FRACTION,
                    "bbox_bounds_gate_pass": bool(bbox_valid), "yolo_label_written": False,
                }
                if not visibility_valid:
                    manifest["rejection_reasons"].append("DEPTH_VISIBLE_SURFACE_OR_BBOX_GATE_FAILED")
                preview_status = "CREATED_DIAGNOSTIC_ONLY"
            except (ValueError, RuntimeError, cv2.error, OSError) as error:
                preview_status = f"FAILED: {error}"
                manifest["rejection_reasons"].append("DIAGNOSTIC_GEOMETRY_OR_PREVIEW_FAILED")

        manifest["diagnostic_overlay_status"] = preview_status
        manifest["visual_review_status"] = "AWAITING_AUTHOR" if not manifest["rejection_reasons"] else "NOT_READY"
        manifest["ground_truth_status"] = "AWAITING_AUTHOR_VISUAL_REVIEW" if not manifest["rejection_reasons"] else "NOT_GROUND_TRUTH"
        manifest["dataset_ready_for_training"] = False
        manifest["decision_reason"] = (
            "All recorded technical gates passed; the image remains diagnostic until author visual review."
            if not manifest["rejection_reasons"] else
            "Ground truth rejected because one or more required proofs or quality gates failed: "
            + ", ".join(manifest["rejection_reasons"])
        )
        manifest["status"] = "COMPLETE_AWAITING_AUTHOR_VISUAL_REVIEW" if not manifest["rejection_reasons"] else "COMPLETE_REJECTED"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["operator_notes"] = "Diagnostic only: no YOLO labels or train/val files are created; author review is required before ground-truth use."
        write_json_atomic(manifest_path, manifest)
        print(f"Single-shot diagnostic status: {manifest['status']}")
        print(f"Manifest: {manifest_path}")
        return output_dir
    except Exception as error:
        manifest.update({
            "status": "FAILED", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "set_attempt_count": set_attempt_count,
            "capture_attempt_count": capture_attempt_count,
            "failure": {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "ground_truth_status": "NOT_GROUND_TRUTH", "dataset_ready_for_training": False,
        })
        if node is not None:
            try:
                manifest["candidate_histories"] = node.history_metadata()
                manifest["clock_history"] = list(node.clock_history)
                manifest["event_trace"] = node.events
                manifest["callback_counts"] = dict(node.callback_counts)
                manifest["history_overflow_counts"] = dict(node.history_overflow)
                manifest["clock_history_overflow_count"] = node.clock_overflow
                manifest["event_trace_overflow_count"] = node.event_overflow
                manifest["raw_message_history_complete"] = persist_all_available_raw()
                manifest["raw_message_paths"] = dict(raw_paths)
                manifest["raw_message_history_paths"] = {
                    key: list(paths) for key, paths in raw_history_paths.items()
                }
                manifest["raw_message_history_counts"] = {
                    key: len(paths) for key, paths in raw_history_paths.items()
                }
            except Exception as persist_error:
                manifest["raw_message_history_complete"] = False
                manifest["raw_persistence_failure"] = str(persist_error)
        else:
            manifest["raw_message_history_complete"] = False
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()


def create_contact_sheet(output_dir: Path, samples: list[dict]) -> None:
    if not samples:
        return
    columns = 4
    thumb_w, thumb_h, title_h = 320, 180, 24
    rows = math.ceil(len(samples) / columns)
    sheet = np.full((rows * (thumb_h + title_h), columns * thumb_w, 3), 245, dtype=np.uint8)
    for index, sample in enumerate(samples):
        preview_path = output_dir / sample["preview_path"]
        image = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        x, y = col * thumb_w, row * (thumb_h + title_h)
        sheet[y:y + thumb_h, x:x + thumb_w] = image
        cv2.putText(
            sheet, sample["sample_id"][:42], (x + 5, y + thumb_h + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / "preview" / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def representative_contact_samples(samples: list[dict], maximum_per_split: int = 16) -> list[dict]:
    """Choose a compact preview spanning each split's azimuth range."""
    selected: list[dict] = []
    split_order = ("train", "val", "test")
    for split in split_order:
        rows = sorted(
            (sample for sample in samples if sample.get("split") == split),
            key=lambda sample: (
                float(sample.get("azimuth_deg", 0.0)),
                float(sample.get("camera_distance_m", 0.0)),
                sample.get("sample_id", ""),
            ),
        )
        if not rows:
            continue
        indices = np.linspace(
            0, len(rows) - 1, num=min(maximum_per_split, len(rows)), dtype=np.int64,
        )
        selected.extend(rows[int(index)] for index in indices)
    return selected


def finalize_failed_capture(args: argparse.Namespace, error: Exception) -> None:
    """Commit a FAILED manifest from progress without overwriting another run."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    progress_path = output_dir / "metadata" / "capture_progress.json"
    expected_token = getattr(args, "run_token", None)
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if expected_token is not None and progress.get("run_token") != expected_token:
            return
        manifest = progress.get("manifest", {})
        accepted = progress.get("accepted_samples", [])
        rejected = progress.get("rejected_samples", [])
        current_sample_id = progress.get("current_sample_id")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "suitcase-pilot-v3-independent-camera",
            "run_token": expected_token,
            "random_seed": getattr(args, "seed", None),
            "requested_sample_count": getattr(args, "sample_count", None),
        }
        accepted, rejected, current_sample_id = [], [], None

    accepted_ids = {sample.get("sample_id") for sample in accepted}
    rejected_ids = {sample.get("sample_id") for sample in rejected}
    if (current_sample_id and current_sample_id not in accepted_ids
            and current_sample_id not in rejected_ids):
        rejected.append({
            "sample_id": current_sample_id,
            "reason": "CAPTURE_ABORTED",
            "detail": str(error),
        })
        rejected_ids.add(current_sample_id)
    planned = [sample.get("sample_id") for sample in manifest.get("pilot_plan", [])]
    handled = accepted_ids | rejected_ids
    if current_sample_id:
        handled.add(current_sample_id)
    manifest.update({
        "status": "FAILED",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "sample_id": current_sample_id,
        },
        "failed_sample_id": current_sample_id,
        "requested_sample_count": getattr(args, "sample_count", None),
        "planned_sample_count": len(planned),
        "attempted_sample_count": len(accepted_ids | rejected_ids),
        "accepted_sample_count": len(accepted),
        "rejected_sample_count": len(rejected),
        "unattempted_sample_ids": [sample_id for sample_id in planned if sample_id not in handled],
        "dataset_ready_for_training": False,
        "visual_review_status": "NOT_READY",
        "samples": accepted,
        "rejected_samples": rejected,
    })
    metadata_dir = output_dir / "metadata"
    write_json_atomic(metadata_dir / "capture_manifest.json", manifest)
    write_json_atomic(metadata_dir / "rejected_samples.json", rejected)
    write_json_atomic(progress_path, {
        "schema": "data01-capture-progress-v2",
        "run_token": manifest.get("run_token"),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAILED",
        "current_sample_id": current_sample_id,
        "manifest": manifest,
        "accepted_samples": accepted,
        "rejected_samples": rejected,
    })
    if accepted:
        try:
            create_contact_sheet(output_dir, accepted)
        except (OSError, cv2.error):
            pass


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "verification" / "datasets" / f"suitcase_pilot_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Capture from an already-running approved Gazebo session")
    parser.add_argument(
        "--diagnostic-single-shot", action="store_true",
        help="Capture one diagnostic RGB-D set into an isolated non-training output directory",
    )
    parser.add_argument(
        "--diagnostic-source-manifest", type=Path,
        help="Reuse one exact held-out pose from a prior capture manifest",
    )
    parser.add_argument(
        "--diagnostic-sample-id",
        help="Sample ID from --diagnostic-source-manifest; diagnostic-only, never training data",
    )
    parser.add_argument("--world", default=str(DEFAULT_WORLD))
    parser.add_argument("--mesh", default=str(DEFAULT_MESH))
    default_output_dir = str(_default_output())
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.set_defaults(default_output_dir=default_output_dir)
    parser.add_argument("--target-entity", default=TARGET_ENTITY)
    parser.add_argument("--image-topic", default=IMAGE_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--camera-info-topic", default=CAMERA_INFO_TOPIC)
    parser.add_argument("--depth-camera-info-topic", default=DEPTH_CAMERA_INFO_TOPIC)
    parser.add_argument("--set-state-service", default="/gazebo/set_entity_state")
    parser.add_argument("--get-state-service", default="/gazebo/get_entity_state")
    parser.add_argument("--service-timeout-s", type=float, default=10.0)
    parser.add_argument("--frame-timeout-s", type=float, default=5.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--max-sync-delta-s", type=float, default=0.05)
    parser.add_argument("--max-camera-stamp-delta-s", type=float, default=DEFAULT_MAX_CAMERA_STAMP_DELTA_S)
    parser.add_argument("--delivery-latency-warning-s", type=float, default=DEFAULT_DELIVERY_LATENCY_WARNING_S)
    parser.add_argument(
        "--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT,
        help="accepted target: 12–20 pilot, 300 production-room component, 100 empty-scene component, or 400 single-scene dataset",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    world_path, mesh_path = Path(args.world).resolve(), Path(args.mesh).resolve()
    if args.capture or args.diagnostic_single_shot:
        args.run_token = uuid.uuid4().hex
    try:
        if args.capture and args.diagnostic_single_shot:
            raise ValueError("--capture and --diagnostic-single-shot are separate modes; use only one")
        if args.diagnostic_single_shot and args.output_dir == args.default_output_dir:
            raise ValueError("--diagnostic-single-shot requires an explicit unique --output-dir")
        if (args.diagnostic_source_manifest is None) != (args.diagnostic_sample_id is None):
            raise ValueError(
                "--diagnostic-source-manifest and --diagnostic-sample-id must be supplied together"
            )
        if args.diagnostic_source_manifest is not None and not args.diagnostic_single_shot:
            raise ValueError("Reusing a planned pose is allowed only with --diagnostic-single-shot")
        if args.target_entity != TARGET_ENTITY:
            raise ValueError(f"Dataset generator only supports the approved {TARGET_ENTITY} model")
        if args.sample_count < 2 and not args.diagnostic_single_shot:
            raise ValueError("--sample-count must be at least 2")
        if isinstance(args.seed, bool) or not 0 <= args.seed < 2**64:
            raise ValueError("--seed must be an integer in [0, 2**64)")
        scene_variant = scene_variant_for_world(world_path)
        pilot_capture = 12 <= args.sample_count <= MAX_AUTHORIZED_PILOT_SAMPLES
        approved_component = (
            (scene_variant == "production_room" and args.sample_count == 300)
            or (scene_variant == "empty" and args.sample_count == 100)
        )
        approved_single_scene_dataset = (
            scene_variant == "production_room"
            and args.sample_count == AUTHORIZED_FULL_DATASET_SAMPLES
        )
        if args.capture and not (
            pilot_capture or approved_component or approved_single_scene_dataset
        ):
            raise ValueError(
                "Live capture accepts a 12–20-view pilot, 300 production-room samples, "
                "100 empty-scene samples, or the legacy 400-sample room target"
            )
        if (not math.isfinite(args.max_camera_stamp_delta_s)
                or args.max_camera_stamp_delta_s <= 0.0):
            raise ValueError("--max-camera-stamp-delta-s must be positive and finite")
        if (not math.isfinite(args.delivery_latency_warning_s)
                or args.delivery_latency_warning_s < 0.0):
            raise ValueError("--delivery-latency-warning-s must be nonnegative and finite")
        world_hash, mesh_hash = check_scene_files(world_path, mesh_path)
        mesh = load_collada_geometry(mesh_path)
        if args.diagnostic_single_shot:
            run_diagnostic_single_shot(args, mesh, world_hash, mesh_hash)
            return 0
        target_position, target_q = target_model_pose(world_path)
        target_center = suitcase_visual_center(target_position, target_q, mesh)
        samples = pilot_samples(target_center, args.sample_count, args.seed, world_path)
        obstacles = validate_camera_clearance(world_path, samples)
        projected_boxes = projected_pilot_boxes(
            samples, target_position, target_q, target_center, mesh,
        )
        if not args.capture and not args.diagnostic_single_shot:
            split_counts = {
                split: sum(sample.split == split for sample in samples)
                for split in dataset_split_counts(args.sample_count)
            }
            print("OFFLINE DATASET PLAN ONLY — Gazebo and ROS are not contacted.")
            print(f"Approved world SHA-256: {world_hash}")
            print(f"Approved suitcase mesh SHA-256: {mesh_hash}")
            print(f"Mesh: {mesh.geometry_count} geometries, {mesh.vertices.shape[0]} vertices, {mesh.triangle_count} triangles")
            print(f"Static world collision boxes checked: {len(obstacles)}; camera clearance radius: {CAMERA_CLEARANCE_RADIUS_M:.2f} m")
            print(f"Planned accepted target: {len(samples)} ({split_counts}); seed={args.seed}")
            print(f"Horizontal distance range: {CAMERA_DISTANCE_RANGE_M[0]:.2f}–{CAMERA_DISTANCE_RANGE_M[1]:.2f} m")
            print(f"Height above target center: {CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[0]:.2f}–{CAMERA_HEIGHT_ABOVE_TARGET_RANGE_M[1]:.2f} m")
            print(f"Train azimuth bands: {TRAIN_AZIMUTH_INTERVALS_DEG}")
            print(f"Validation azimuth bands: {VALIDATION_AZIMUTH_INTERVALS_DEG}")
            if "test" in split_counts:
                print(f"Test azimuth bands: {TEST_AZIMUTH_INTERVALS_DEG}")
            print("10-degree angular guard gaps keep full-dataset split sectors separate.")
            print(f"Image target position ranges: x={IMAGE_TARGET_X_FRACTION_RANGE}, y={IMAGE_TARGET_Y_FRACTION_RANGE}")
            print(f"Coverage summary: {json.dumps(pilot_coverage_summary(samples), ensure_ascii=False)}")
            print(f"Full-geometry boxes fit {CAMERA_WIDTH}x{CAMERA_HEIGHT}: {len(projected_boxes)}/{len(samples)}")
            print("The visible-object YOLO box will come from depth-confirmed DAE surface pixels.")
            return 0
        run_capture(args, mesh, world_hash, mesh_hash)
        return 0
    except Exception as error:
        if args.capture:
            try:
                finalize_failed_capture(args, error)
            except Exception as manifest_error:
                print(f"dataset manifest finalization failed: {manifest_error}", file=sys.stderr)
        print(f"dataset generator: {error}", file=sys.stderr)
        if args.diagnostic_single_shot:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
