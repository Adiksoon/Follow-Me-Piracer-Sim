#!/usr/bin/env python3
"""Prepare a small, geometry-labelled suitcase pilot for the approved world.

The capture path is deliberately explicit: this script never launches Gazebo.
Run it without ``--capture`` to inspect the deterministic pilot plan.  With
``--capture``, it connects to an already-running production simulation and
moves only ``suitcase_target`` along the clear centre aisle.

The workflow is informed by KulunuOS/gazebo_dataset_generation
(revision 928c0be3d1366adf3ce3a82e4c432fc1b0171b46, render.py).  Its ROS 1
camera-sweep/mask implementation was not copied verbatim; this focused ROS 2
implementation uses the production DAE, camera intrinsics, Gazebo entity state,
and the aligned depth stream to create suitcase bounding boxes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
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
DEFAULT_MESH = ASSET_SHARE / "assets/hard_trolley_red/meshes/suitcase.dae"
WORLD_SHA256 = "71111ed9cfd933117c3ac7013bed41c85d342eb5cc6b8e4e0eddbeeb5ae39c7e"
MESH_SHA256 = "e4d75b678e566289f15d58fdc7216c93f2199a34bb82377bd785d1e94a14f1ca"
FURNITURE_SHA256 = {
    "lab_table_beige/model.sdf": "f72bd4d9b5f0b50f6375589080842762b38dc8fccef6cca5c54cf7bb5e5a44b9",
    "wooden_chair_blue_gray/model.sdf": "6c0d4d2219af49b9b5737e0a25b36ebe5b69349640d4509eff1f21fe3027b28f",
}
TARGET_ENTITY = "suitcase_target"
CAMERA_ENTITY = "piracer::sim_camera_optical_frame"
IMAGE_TOPIC = "/piracer/sim_camera/color/image_raw"
DEPTH_TOPIC = "/piracer/sim_camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/piracer/sim_camera/color/camera_info"
DEPTH_CAMERA_INFO_TOPIC = "/piracer/sim_camera/depth/camera_info"
TARGET_Z = 0.05
SAFE_LANE_Y = 0.0
PILOT_TARGET_X_M = (2.7, 3.3)
PILOT_YAWS_DEG = (0, 45, 60, 90, 120, 180, 240, 270, 300, 315)
YOLO_CLASS_ID = 0
YOLO_CLASS_NAME = "suitcase"
MAX_OCCLUSION_SAMPLES = 12_000
MIN_DEPTH_MATCHES = 100
MIN_VISIBLE_FRACTION = 0.80
CAPTURE_POSE_MAX_POSITION_DELTA_M = 0.02
CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD = math.radians(2.0)

COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": COLLADA_NS}


@dataclass(frozen=True)
class PilotSample:
    sample_id: str
    split: str
    scenario_id: str
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: int


@dataclass(frozen=True)
class MeshGeometry:
    vertices: np.ndarray
    triangle_centers: np.ndarray
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


def pilot_samples() -> list[PilotSample]:
    """Return 20 deterministic poses, grouped by distance to prevent frame leakage."""
    samples: list[PilotSample] = []
    for distance_index, target_x in enumerate(PILOT_TARGET_X_M):
        split = "train" if distance_index == 0 else "val"
        scenario_id = f"target_x_{target_x:.1f}m"
        for yaw in PILOT_YAWS_DEG:
            samples.append(PilotSample(
                sample_id=f"{scenario_id}_yaw_{yaw:03d}",
                split=split,
                scenario_id=scenario_id,
                x_m=target_x,
                y_m=SAFE_LANE_Y,
                z_m=TARGET_Z,
                yaw_deg=yaw,
            ))
    return samples


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
    centers_out: list[np.ndarray] = []
    geometry_count = 0
    triangle_count = 0

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
            vertex_offset = int(vertex_input.get("offset", "0"))
            stride = 1 + max(int(item.get("offset", "0")) for item in inputs)
            index_text = triangles.findtext("c:p", default="", namespaces=NS)
            indices = np.fromstring(index_text, sep=" ", dtype=np.int64)
            if indices.size % (3 * stride):
                raise ValueError(f"Malformed triangle indices in {geometry_id}")
            position_indices = indices.reshape(-1, stride)[:, vertex_offset]
            triangle_vertices = vertices[position_indices].reshape(-1, 3, 3)
            centers_out.append(triangle_vertices.mean(axis=1))
            triangle_count += triangle_vertices.shape[0]

        if mesh.find("c:polylist", NS) is not None or mesh.find("c:polygons", NS) is not None:
            raise ValueError("The approved suitcase mesh uses triangles; polygon lists are unsupported")

    if not vertices_out or not centers_out:
        raise ValueError("No rendered suitcase geometry was found in the COLLADA scene")
    vertices = np.concatenate(vertices_out, axis=0)
    triangle_centers = np.concatenate(centers_out, axis=0)
    bounds_min = tuple(float(value) for value in vertices.min(axis=0))
    bounds_max = tuple(float(value) for value in vertices.max(axis=0))
    return MeshGeometry(
        vertices=vertices,
        triangle_centers=triangle_centers,
        local_bounds_min=bounds_min,
        local_bounds_max=bounds_max,
        geometry_count=geometry_count,
        triangle_count=triangle_count,
    )


def quaternion_matrix_xyzw(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Return a 3x3 rotation matrix from a ROS-order x,y,z,w quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Quaternion has zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


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


def depth_visibility(
    sample_points_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_m: np.ndarray,
    distortion: tuple[float, ...] | None = None,
) -> dict[str, float | int]:
    """Compare projected mesh surface samples with aligned measured depth."""
    height, width = depth_m.shape[:2]
    u, v, z = project_camera_points(sample_points_camera, intrinsics, distortion)
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    in_frame = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, z = px[in_frame], py[in_frame], z[in_frame]
    if px.size == 0:
        raise CaptureRejected("NO_PROJECTED_DEPTH_SAMPLES")

    # Point-sample z-buffer: where multiple mesh triangles project to the same
    # pixel, retain the nearest surface to avoid counting the suitcase's own
    # back faces as external occlusion.
    flat_indices = py * width + px
    nearest = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(nearest, flat_indices, z.astype(np.float32))
    expected = nearest.reshape(height, width)
    projected = np.isfinite(expected)
    observed = depth_m.astype(np.float32, copy=False)
    valid = projected & np.isfinite(observed) & (observed > 0.05)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < MIN_DEPTH_MATCHES:
        raise CaptureRejected(f"INSUFFICIENT_DEPTH_EVIDENCE_{valid_count}")
    tolerance = np.maximum(0.06, expected * 0.03)
    matched = valid & (np.abs(observed - expected) <= tolerance)
    match_count = int(np.count_nonzero(matched))
    fraction = match_count / valid_count
    return {
        "projected_mesh_pixels": int(np.count_nonzero(projected)),
        "valid_depth_pixels": valid_count,
        "matching_depth_pixels": match_count,
        "visible_fraction": float(fraction),
    }


def check_scene_files(world_path: Path, mesh_path: Path) -> tuple[str, str]:
    world_hash, mesh_hash = sha256_file(world_path), sha256_file(mesh_path)
    if world_hash != WORLD_SHA256:
        raise ValueError(f"World is not the approved production revision: SHA-256 {world_hash}")
    if mesh_hash != MESH_SHA256:
        raise ValueError(f"Suitcase mesh is not the approved hard_trolley_red asset: SHA-256 {mesh_hash}")
    for relative_path, expected_hash in FURNITURE_SHA256.items():
        furniture_path = ASSET_SHARE / "assets" / relative_path
        actual_hash = sha256_file(furniture_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Approved production furniture changed: {relative_path} SHA-256 {actual_hash}")
    world_xml = ET.parse(world_path).getroot()
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
                "half_x": half_x,
                "half_y": half_y,
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


def validate_pilot_clearance(world_path: Path) -> list[dict]:
    """Check every planned suitcase footprint against production-world collisions."""
    world_root = ET.parse(world_path).getroot()
    world = world_root.find("world")
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
        obstacles.extend({**box, "model": include_name} for box in _collision_boxes(model, include_pose))

    target_model = world.find(f"model[@name='{TARGET_ENTITY}']")
    if target_model is None:
        raise ValueError("Production-world target model is missing")
    # Collision radii are measured around the model origin, so remove the
    # target's initial world pose before reading its local collision geometry.
    target_world_pose = target_model.find("pose")
    if target_world_pose is not None:
        target_model.remove(target_world_pose)
    target_radius = max(
        math.hypot(box["half_x"], box["half_y"])
        + math.hypot(box["center_x"], box["center_y"])
        for box in _collision_boxes(target_model)
    )
    clearance_radius = target_radius + 0.05
    if not obstacles:
        raise ValueError("No static world obstacles were checked")

    for sample in pilot_samples():
        for obstacle in obstacles:
            nearest_x = min(max(sample.x_m, obstacle["center_x"] - obstacle["half_x"]),
                            obstacle["center_x"] + obstacle["half_x"])
            nearest_y = min(max(sample.y_m, obstacle["center_y"] - obstacle["half_y"]),
                            obstacle["center_y"] + obstacle["half_y"])
            distance = math.hypot(sample.x_m - nearest_x, sample.y_m - nearest_y)
            if distance <= clearance_radius:
                raise ValueError(
                    f"Pilot pose {sample.sample_id} intersects the conservative footprint "
                    f"of {obstacle.get('model', 'world')}::{obstacle['name']}"
                )
    return obstacles


def _unit_quaternion_difference(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    dot = min(1.0, abs(float(np.dot(qa, qb))))
    return 2.0 * math.acos(dot)


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def populate_set_entity_state_request(request, entity_name: str, sample: PilotSample):
    """Populate the fields actually declared by gazebo_msgs/SetEntityState."""
    request.state.name = entity_name
    request.state.reference_frame = "world"
    request.state.pose.position.x = sample.x_m
    request.state.pose.position.y = sample.y_m
    request.state.pose.position.z = sample.z_m
    half_yaw = math.radians(sample.yaw_deg) / 2.0
    request.state.pose.orientation.z = math.sin(half_yaw)
    request.state.pose.orientation.w = math.cos(half_yaw)
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
    obstacle_count: int,
) -> Path:
    """Capture the pilot from an already-running ROS 2 / Gazebo session."""
    try:
        import rclpy
        from cv_bridge import CvBridge
        from gazebo_msgs.srv import GetEntityState, SetEntityState
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as error:
        raise RuntimeError("ROS 2 Humble, gazebo_msgs, cv_bridge, and sensor_msgs must be sourced") from error

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    for relative in (
        "images/train", "images/val", "labels/train", "labels/val",
        "depth/train", "depth/val", "preview/samples", "metadata",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    class DatasetCaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("suitcase_dataset_pilot")
            self.bridge = CvBridge()
            self.latest_color = None
            self.latest_depth = None
            self.latest_color_info = None
            self.latest_depth_info = None
            self.create_subscription(Image, args.image_topic, self._on_color, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info_topic, self._on_color_info, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.depth_camera_info_topic, self._on_depth_info, qos_profile_sensor_data)
            self.set_state = self.create_client(SetEntityState, args.set_state_service)
            self.get_state = self.create_client(GetEntityState, args.get_state_service)

        def _on_color(self, message: Image) -> None:
            self.latest_color = message

        def _on_depth(self, message: Image) -> None:
            self.latest_depth = message

        def _on_color_info(self, message: CameraInfo) -> None:
            self.latest_color_info = message

        def _on_depth_info(self, message: CameraInfo) -> None:
            self.latest_depth_info = message

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

        def set_target(self, sample: PilotSample) -> None:
            request = SetEntityState.Request()
            populate_set_entity_state_request(request, args.target_entity, sample)
            response = self.spin_future(self.set_state.call_async(request))
            if not response.success:
                raise RuntimeError(f"SetEntityState failed for {args.target_entity}")

        def snapshot(self, timeout_s: float, pose_stamp_s: float):
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                color, depth = self.latest_color, self.latest_depth
                color_info, depth_info = self.latest_color_info, self.latest_depth_info
                if color is None or depth is None or color_info is None or depth_info is None:
                    continue
                if any(frame != color.header.frame_id for frame in (
                    depth.header.frame_id, color_info.header.frame_id, depth_info.header.frame_id
                )):
                    continue
                color_stamp = color.header.stamp.sec + color.header.stamp.nanosec * 1e-9
                depth_stamp = depth.header.stamp.sec + depth.header.stamp.nanosec * 1e-9
                if not rgbd_timestamps_are_fresh(
                    color_stamp, depth_stamp, pose_stamp_s, args.max_sync_delta_s
                ):
                    continue
                color_bgr = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
                depth_raw = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                return color_bgr.copy(), depth_raw.copy(), color, depth, color_info, depth_info
            raise TimeoutError("No synchronized RGB-D frame and CameraInfo received before timeout")

    rclpy.init(args=None)
    node = DatasetCaptureNode()
    manifest_samples: list[dict] = []
    rejected: list[dict] = []
    camera_reference = None
    base_metadata = {
        "schema": "suitcase-pilot-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "world_path": str(Path(args.world).resolve()),
        "world_sha256": world_hash,
        "mesh_path": str(Path(args.mesh).resolve()),
        "mesh_sha256": mesh_hash,
        "furniture_sha256": FURNITURE_SHA256,
        "static_world_collision_boxes_checked": obstacle_count,
        "static_clearance_margin_m": 0.05,
        "target_entity": args.target_entity,
        "camera_entity": args.camera_entity,
        "topics": {
            "rgb": args.image_topic,
            "depth": args.depth_topic,
            "camera_info": args.camera_info_topic,
            "depth_camera_info": args.depth_camera_info_topic,
        },
        "geometry": {
            "source": "hard_trolley_red DAE vertices and triangle centroids",
            "vertices": int(mesh.vertices.shape[0]),
            "triangles": mesh.triangle_count,
            "local_bounds_before_visual_offset_m": {
                "min": mesh.local_bounds_min,
                "max": mesh.local_bounds_max,
            },
            "visual_offset_m": [0.0, 0.0, -0.05],
        },
        "label": {"format": "YOLO", "class_id": YOLO_CLASS_ID, "class_name": YOLO_CLASS_NAME},
        "pilot_plan": [asdict(sample) for sample in pilot_samples()],
        "split_policy": "World-x scenarios are kept intact: target x=2.7 m=train, target x=3.3 m=val. Pilot split is for review only.",
        "kulunuos_reference": {
            "repository": "https://github.com/KulunuOS/gazebo_dataset_generation",
            "revision": "928c0be3d1366adf3ce3a82e4c432fc1b0171b46",
            "file": "render.py",
        },
        "asset_attribution": (
            'Based on “Hard Trolley Bag With Adjustable Handle” by Adrian Saputra, '
            'Sketchfab UID 1758ad5207d943d597db8ff2b21bf827, CC BY 4.0; '
            'red shell variant. Source credit retained in simulation_assets.'
        ),
    }

    try:
        for client, service_name in (
            (node.set_state, args.set_state_service),
            (node.get_state, args.get_state_service),
        ):
            if not client.wait_for_service(timeout_sec=args.service_timeout_s):
                raise TimeoutError(f"Required Gazebo service is unavailable: {service_name}")

        for sample in pilot_samples():
            node.set_target(sample)
            settle_deadline = time.monotonic() + args.settle_s
            while rclpy.ok() and time.monotonic() < settle_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)

            target_response = node.entity_state_response(args.target_entity)
            target_state = target_response.state
            target_pose_stamp_s = stamp_seconds(target_response.header.stamp)
            target_pose = target_state.pose
            actual_position = (
                target_pose.position.x, target_pose.position.y, target_pose.position.z
            )
            actual_q = (
                target_pose.orientation.x, target_pose.orientation.y,
                target_pose.orientation.z, target_pose.orientation.w,
            )
            yaw_error = _unit_quaternion_difference(
                actual_q,
                (0.0, 0.0, math.sin(math.radians(sample.yaw_deg) / 2.0), math.cos(math.radians(sample.yaw_deg) / 2.0)),
            )
            xy_error = math.hypot(actual_position[0] - sample.x_m, actual_position[1] - sample.y_m)
            z_error = abs(actual_position[2] - sample.z_m)
            if xy_error > 0.08 or z_error > 0.05 or yaw_error > math.radians(6.0):
                rejected.append({
                    "sample_id": sample.sample_id,
                    "reason": "TARGET_POSE_DRIFT_OR_GROUND_HEIGHT",
                    "xy_error_m": xy_error,
                    "z_error_m": z_error,
                    "yaw_error_rad": yaw_error,
                })
                continue

            camera_state = node.entity_state_response(args.camera_entity).state
            camera_pose = camera_state.pose
            camera_position = (
                camera_pose.position.x, camera_pose.position.y, camera_pose.position.z
            )
            camera_q = (
                camera_pose.orientation.x, camera_pose.orientation.y,
                camera_pose.orientation.z, camera_pose.orientation.w,
            )
            camera_forward = quaternion_matrix_xyzw(camera_q)[:, 2]
            camera_to_target = np.asarray((sample.x_m, sample.y_m, sample.z_m)) - np.asarray(camera_position)
            forward_distance = float(np.dot(camera_to_target, camera_forward))
            lateral_distance = float(np.linalg.norm(camera_to_target - forward_distance * camera_forward))
            if (
                camera_forward[0] < 0.98
                or abs(camera_forward[1]) > 0.05
                or forward_distance < 1.5
                or lateral_distance > 0.25
            ):
                raise RuntimeError(
                    "Fixed camera is not aimed down the approved clear aisle; "
                    f"forward={camera_forward.tolist()}, lateral_error={lateral_distance:.3f} m"
                )
            if camera_reference is None:
                camera_reference = (camera_position, camera_q)
            elif (
                np.linalg.norm(np.asarray(camera_position) - np.asarray(camera_reference[0])) > 0.01
                or _unit_quaternion_difference(camera_q, camera_reference[1]) > math.radians(1.0)
            ):
                raise RuntimeError("Camera moved during pilot; fixed-camera requirement was violated")

            color, depth_raw, color_message, depth_message, info, depth_info = node.snapshot(
                args.frame_timeout_s, target_pose_stamp_s
            )
            target_after_capture = node.entity_state_response(args.target_entity).state.pose
            target_after_position = (
                target_after_capture.position.x,
                target_after_capture.position.y,
                target_after_capture.position.z,
            )
            target_after_yaw = (
                target_after_capture.orientation.x,
                target_after_capture.orientation.y,
                target_after_capture.orientation.z,
                target_after_capture.orientation.w,
            )
            target_position_delta, target_orientation_delta = capture_pose_delta(
                actual_position, actual_q, target_after_position, target_after_yaw
            )
            if (
                target_position_delta > CAPTURE_POSE_MAX_POSITION_DELTA_M
                or target_orientation_delta > CAPTURE_POSE_MAX_ORIENTATION_DELTA_RAD
            ):
                rejected.append({
                    "sample_id": sample.sample_id,
                    "reason": "TARGET_MOVED_DURING_FRAME_CAPTURE",
                    "position_delta_m": target_position_delta,
                    "orientation_delta_rad": target_orientation_delta,
                })
                continue
            height, width = color.shape[:2]
            if (
                depth_raw.shape[:2] != (height, width)
                or (info.height, info.width) != (height, width)
                or (depth_info.height, depth_info.width) != (height, width)
            ):
                rejected.append({"sample_id": sample.sample_id, "reason": "RGB_DEPTH_CAMERAINFO_SIZE_MISMATCH"})
                continue
            if (
                not np.allclose(info.k, depth_info.k, atol=1e-6)
                or not np.allclose(info.d, depth_info.d, atol=1e-6)
            ):
                rejected.append({"sample_id": sample.sample_id, "reason": "RGB_DEPTH_CALIBRATION_MISMATCH"})
                continue
            fx, fy, cx, cy = (float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]))
            if (
                fx <= 0 or fy <= 0
                or info.binning_x != 0 or info.binning_y != 0
                or info.roi.width != 0 or info.roi.height != 0
            ):
                rejected.append({"sample_id": sample.sample_id, "reason": "INVALID_CAMERA_INTRINSICS"})
                continue
            intrinsics = (fx, fy, cx, cy)
            distortion = tuple(float(value) for value in info.d)

            vertices_local = mesh.vertices.copy()
            vertices_local[:, 2] -= 0.05  # suitcase_visual pose in the approved SDF
            points_camera = pose_points_to_camera(
                vertices_local,
                actual_position,
                actual_q,
                (0.0, 0.0, 0.0),
                camera_position,
                camera_q,
            )
            try:
                bbox = projected_bbox(points_camera, intrinsics, width, height, distortion=distortion)
                sample_indices = np.linspace(
                    0, mesh.triangle_centers.shape[0] - 1,
                    min(MAX_OCCLUSION_SAMPLES, mesh.triangle_centers.shape[0]),
                    dtype=np.int64,
                )
                centers_local = mesh.triangle_centers[sample_indices].copy()
                centers_local[:, 2] -= 0.05
                center_camera = pose_points_to_camera(
                    centers_local,
                    actual_position,
                    actual_q,
                    (0.0, 0.0, 0.0),
                    camera_position,
                    camera_q,
                )
                depth_m = depth_to_meters(depth_raw, depth_message.encoding)
                visibility = depth_visibility(center_camera, intrinsics, depth_m, distortion)
                if visibility["visible_fraction"] < MIN_VISIBLE_FRACTION:
                    raise CaptureRejected(
                        f"OCCLUDED_OR_DEPTH_MISMATCH_{visibility['visible_fraction']:.3f}"
                    )
            except CaptureRejected as error:
                rejected.append({"sample_id": sample.sample_id, "reason": str(error)})
                continue

            normalized = yolo_bbox(bbox, width, height)
            sample_dir = output_dir / "images" / sample.split
            image_path = sample_dir / f"{sample.sample_id}.png"
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
            preview = color.copy()
            cv2.rectangle(preview, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
            cv2.putText(
                preview, f"{sample.scenario_id} yaw={sample.yaw_deg}deg",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
            preview_path = output_dir / "preview" / "samples" / f"{sample.sample_id}.jpg"
            cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

            rgb_stamp = color_message.header.stamp.sec + color_message.header.stamp.nanosec * 1e-9
            depth_stamp = depth_message.header.stamp.sec + depth_message.header.stamp.nanosec * 1e-9
            manifest_samples.append({
                **asdict(sample),
                "accepted": True,
                "rgb_path": str(image_path.relative_to(output_dir)),
                "depth_path": str(depth_path.relative_to(output_dir)),
                "label_path": str(label_path.relative_to(output_dir)),
                "preview_path": str(preview_path.relative_to(output_dir)),
                "rgb_stamp_s": rgb_stamp,
                "depth_stamp_s": depth_stamp,
                "rgb_depth_delta_s": abs(rgb_stamp - depth_stamp),
                "image_encoding": color_message.encoding,
                "depth_encoding": depth_message.encoding,
                "frame_id": color_message.header.frame_id,
                "image_width": width,
                "image_height": height,
                "camera_intrinsics_K": list(info.k),
                "camera_distortion_D": list(info.d),
                "camera_projection_P": list(info.p),
                "depth_camera_intrinsics_K": list(depth_info.k),
                "depth_camera_distortion_D": list(depth_info.d),
                "depth_camera_projection_P": list(depth_info.p),
                "camera_position_world_m": camera_position,
                "camera_orientation_world_xyzw": camera_q,
                "target_position_world_m": actual_position,
                "target_orientation_world_xyzw": actual_q,
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "bbox_yolo_xywh_normalized": normalized,
                "depth_visibility": visibility,
                "label_origin": "projected vertices of the approved suitcase DAE; never a YOLO prediction",
            })

        accepted_train = sum(sample["split"] == "train" for sample in manifest_samples)
        accepted_val = sum(sample["split"] == "val" for sample in manifest_samples)
        status = "READY_FOR_AUTHOR_REVIEW" if len(manifest_samples) >= 12 and accepted_train and accepted_val else "INCOMPLETE"
        base_metadata.update({
            "status": status,
            "accepted_sample_count": len(manifest_samples),
            "rejected_sample_count": len(rejected),
            "samples": manifest_samples,
            "rejected_samples": rejected,
            "capture_settings": {
                "settle_s": args.settle_s,
                "max_sync_delta_s": args.max_sync_delta_s,
                "minimum_visible_fraction": MIN_VISIBLE_FRACTION,
                "depth_tolerance_m": "max(0.06, 0.03 * expected_z)",
                "edge_clipping_policy": "reject any projected geometry outside a 2 px image margin",
            },
        })
        (output_dir / "metadata" / "capture_manifest.json").write_text(
            json.dumps(base_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / "metadata" / "rejected_samples.json").write_text(
            json.dumps(rejected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: suitcase\n",
            encoding="utf-8",
        )
        attribution = (
            'Based on “Hard Trolley Bag With Adjustable Handle” by Adrian Saputra, '
            'Sketchfab UID 1758ad5207d943d597db8ff2b21bf827; CC BY 4.0. '
            'Red shell variant. Original credit: https://sketchfab.com/3d-models/'
            'hard-trolley-bag-with-adjustable-handle-1758ad5207d943d597db8ff2b21bf827\n'
            'Dataset workflow reference: KulunuOS/gazebo_dataset_generation, '
            'revision 928c0be3d1366adf3ce3a82e4c432fc1b0171b46, render.py.\n'
        )
        (output_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
        create_contact_sheet(output_dir, manifest_samples)
        print(f"Dataset status: {status}; accepted={len(manifest_samples)}; rejected={len(rejected)}")
        print(f"Output: {output_dir}")
        return output_dir
    finally:
        node.destroy_node()
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


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "verification" / "datasets" / f"suitcase_pilot_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Capture from an already-running approved Gazebo session")
    parser.add_argument("--world", default=str(DEFAULT_WORLD))
    parser.add_argument("--mesh", default=str(DEFAULT_MESH))
    parser.add_argument("--output-dir", default=str(_default_output()))
    parser.add_argument("--target-entity", default=TARGET_ENTITY)
    parser.add_argument("--camera-entity", default=CAMERA_ENTITY)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    world_path, mesh_path = Path(args.world).resolve(), Path(args.mesh).resolve()
    try:
        world_hash, mesh_hash = check_scene_files(world_path, mesh_path)
        mesh = load_collada_geometry(mesh_path)
        obstacles = validate_pilot_clearance(world_path)
        if not args.capture:
            print("OFFLINE PILOT PLAN ONLY — Gazebo and ROS are not contacted.")
            print(f"Approved world SHA-256: {world_hash}")
            print(f"Approved suitcase mesh SHA-256: {mesh_hash}")
            print(f"Mesh: {mesh.geometry_count} geometries, {mesh.vertices.shape[0]} vertices, {mesh.triangle_count} triangles")
            print(f"Static world collision boxes checked: {len(obstacles)}; suitcase clearance margin: 0.05 m")
            print(f"Planned samples: {len(pilot_samples())}")
            print("Target world-x positions: " + ", ".join(f"{x:.1f} m" for x in PILOT_TARGET_X_M))
            print("Yaw angles (deg): " + ", ".join(str(yaw) for yaw in PILOT_YAWS_DEG))
            print("Split: target x=2.7 m -> train; target x=3.3 m -> val (pilot review only).")
            return 0
        run_capture(args, mesh, world_hash, mesh_hash, len(obstacles))
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"dataset generator: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
