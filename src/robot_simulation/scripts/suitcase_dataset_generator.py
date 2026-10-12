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
CAMERA_ENTITY = "dataset_camera"
OPTICAL_FRAME_NAME = "dataset_camera_optical_frame"
IMAGE_TOPIC = "/dataset_camera/color/image_raw"
DEPTH_TOPIC = "/dataset_camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/dataset_camera/color/camera_info"
DEPTH_CAMERA_INFO_TOPIC = "/dataset_camera/depth/camera_info"
PILOT_CAMERA_DISTANCES_M = (2.4, 3.0)
PILOT_TRAIN_AZIMUTHS_DEG = tuple(range(0, 360, 30))
PILOT_VAL_AZIMUTHS_DEG = (15, 60, 105, 150, 195, 240, 285, 330)
PILOT_CAMERA_HEIGHT_ABOVE_TARGET_M = 0.75
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
DEFAULT_MAX_POSE_AGE_S = 0.5

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
    azimuth_deg: int
    camera_distance_m: float


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


def target_model_pose(world_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read the approved suitcase model pose from the selected production world."""
    root = ET.parse(world_path).getroot()
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


def pilot_samples(target_center: tuple[float, float, float]) -> list[PilotSample]:
    """Plan 12 near and 8 far views on a full azimuth orbit around the suitcase."""
    center = np.asarray(target_center, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Suitcase visual center must contain three finite values")
    samples: list[PilotSample] = []
    view_groups = (
        ("train", PILOT_CAMERA_DISTANCES_M[0], PILOT_TRAIN_AZIMUTHS_DEG),
        ("val", PILOT_CAMERA_DISTANCES_M[1], PILOT_VAL_AZIMUTHS_DEG),
    )
    for split, distance, azimuths in view_groups:
        scenario_id = f"camera_orbit_{distance:.1f}m"
        for azimuth in azimuths:
            angle = math.radians(azimuth)
            camera_position = (
                float(center[0] + distance * math.cos(angle)),
                float(center[1] + distance * math.sin(angle)),
                float(center[2] + PILOT_CAMERA_HEIGHT_ABOVE_TARGET_M),
            )
            orientation = camera_link_orientation(camera_position, tuple(center))
            rotation = quaternion_matrix_xyzw(orientation)
            yaw = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))) % 360.0
            pitch = math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0))))
            samples.append(PilotSample(
                sample_id=f"{scenario_id}_azimuth_{azimuth:03d}",
                split=split,
                scenario_id=scenario_id,
                camera_x_m=camera_position[0],
                camera_y_m=camera_position[1],
                camera_z_m=camera_position[2],
                camera_yaw_deg=yaw,
                camera_pitch_deg=pitch,
                azimuth_deg=azimuth,
                camera_distance_m=distance,
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


def depth_visibility(
    vertices_camera: np.ndarray,
    triangles: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_m: np.ndarray,
    distortion: tuple[float, ...] | None = None,
) -> dict[str, float | int]:
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
    return {
        "projected_mesh_pixels": int(np.count_nonzero(projected)),
        **raster_stats,
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
    """Resolve production-world collision bounds used for camera-position checks."""
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
    if not obstacles:
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
    expected_orientation = camera_link_orientation(tuple(actual_position), tuple(target))
    orientation_delta = _unit_quaternion_difference(
        camera_quaternion_xyzw, expected_orientation,
    )
    if position_delta > 0.03 or orientation_delta > math.radians(2.0):
        raise ValueError(
            f"Dataset camera did not reach the planned look-at pose "
            f"(position delta {position_delta:.3f} m, orientation delta "
            f"{math.degrees(orientation_delta):.2f} deg)"
        )
    forward = quaternion_matrix_xyzw(camera_quaternion_xyzw)[:, 0]
    target_direction = target - actual_position
    target_direction /= np.linalg.norm(target_direction)
    if float(np.dot(forward, target_direction)) < math.cos(math.radians(2.0)):
        raise ValueError("Dataset camera optical axis is not aimed at the suitcase")
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
        link_quaternion = camera_link_orientation(camera_position, target_center)
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
    quaternion = camera_link_orientation(
        (sample.camera_x_m, sample.camera_y_m, sample.camera_z_m),
        (
            sample.camera_x_m - sample.camera_distance_m * math.cos(math.radians(sample.azimuth_deg)),
            sample.camera_y_m - sample.camera_distance_m * math.sin(math.radians(sample.azimuth_deg)),
            sample.camera_z_m - PILOT_CAMERA_HEIGHT_ABOVE_TARGET_M,
        ),
    )
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
    """Capture a pilot by moving only the independent dataset camera."""
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

    if not math.isfinite(args.max_pose_age_s) or args.max_pose_age_s <= 0.0:
        raise ValueError("--max-pose-age-s must be a positive finite number")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    for relative in (
        "images/train", "images/val", "labels/train", "labels/val",
        "depth/train", "depth/val", "preview/samples", "metadata",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    camera_model_path = SIMULATION_SHARE / "models/dataset_camera/model.sdf"

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

        def _on_depth(self, message: Image) -> None:
            self.latest_depth = message
            self.latest_depth_received_at = time.monotonic()

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

        def set_camera(self, sample: PilotSample) -> None:
            request = SetEntityState.Request()
            populate_set_entity_state_request(request, CAMERA_ENTITY, sample)
            response = self.spin_future(self.set_state.call_async(request))
            if not response.success:
                raise RuntimeError(f"SetEntityState failed for {CAMERA_ENTITY}")

        def snapshot(
            self,
            timeout_s: float,
            camera_pose_stamp_s: float,
            camera_state_received_at: float,
        ):
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                color, depth = self.latest_color, self.latest_depth
                color_info, depth_info = self.latest_color_info, self.latest_depth_info
                if color is None or depth is None or color_info is None or depth_info is None:
                    continue
                color_received_at = self.latest_color_received_at
                depth_received_at = self.latest_depth_received_at
                if (color_received_at is None or depth_received_at is None
                        or color_received_at <= camera_state_received_at
                        or depth_received_at <= camera_state_received_at):
                    continue
                frame_pose_age_s = max(color_received_at, depth_received_at) - camera_state_received_at
                if frame_pose_age_s > args.max_pose_age_s:
                    raise RuntimeError("Dataset-camera pose became stale before its RGB-D frame arrived")
                if any(frame != color.header.frame_id for frame in (
                    depth.header.frame_id, color_info.header.frame_id, depth_info.header.frame_id,
                )):
                    continue
                color_stamp = stamp_seconds(color.header.stamp)
                depth_stamp = stamp_seconds(depth.header.stamp)
                if not rgbd_timestamps_are_fresh(
                    color_stamp, depth_stamp, camera_pose_stamp_s, args.max_sync_delta_s,
                ):
                    continue
                color_bgr = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
                depth_raw = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                return (
                    color_bgr.copy(), depth_raw.copy(), color, depth, color_info, depth_info,
                    frame_pose_age_s,
                )
            raise TimeoutError(
                "No synchronized dataset-camera RGB-D frame and CameraInfo arrived before timeout"
            )

    rclpy.init(args=None)
    node = DatasetCaptureNode()
    manifest_samples: list[dict] = []
    rejected: list[dict] = []
    try:
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
        samples = pilot_samples(target_center_initial)
        capture_obstacles = validate_camera_clearance(Path(args.world).resolve(), samples)
        pilot_boxes = projected_pilot_boxes(
            samples, initial_target_position, initial_target_q, target_center_initial, mesh,
        )

        base_metadata = {
            "schema": "suitcase-pilot-v2-independent-camera",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "world_path": str(Path(args.world).resolve()),
            "world_sha256": world_hash,
            "mesh_path": str(Path(args.mesh).resolve()),
            "mesh_sha256": mesh_hash,
            "camera_model_path": str(camera_model_path),
            "camera_model_sha256": sha256_file(camera_model_path),
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
                "max_state_age_s": args.max_pose_age_s,
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
                "height_above_target_center_m": PILOT_CAMERA_HEIGHT_ABOVE_TARGET_M,
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
            "target_initial_position_world_m": initial_target_position,
            "target_initial_orientation_world_xyzw": initial_target_q,
            "target_visual_center_world_m": target_center_initial,
            "pilot_plan": [asdict(sample) for sample in samples],
            "planned_projected_boxes_xyxy_px": pilot_boxes,
            "camera_orbit": {
                "target_is_moved": False,
                "camera_horizontal_distances_m": PILOT_CAMERA_DISTANCES_M,
                "train_azimuths_deg": PILOT_TRAIN_AZIMUTHS_DEG,
                "validation_azimuths_deg": PILOT_VAL_AZIMUTHS_DEG,
                "orientation": "camera link +X points toward the DAE visual center",
            },
            "split_policy": (
                "Pilot-only review split by camera range; final dataset must split "
                "whole capture scenarios, not adjacent frames."
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

        for sample in samples:
            node.set_camera(sample)
            settle_deadline = time.monotonic() + args.settle_s
            while rclpy.ok() and time.monotonic() < settle_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)

            camera_response = node.entity_state_response(CAMERA_ENTITY)
            camera_state_received_at = time.monotonic()
            camera_link_position, camera_link_q = entity_pose(camera_response)
            target_response = node.entity_state_response(args.target_entity)
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
            camera_state_stamp_s = stamp_seconds(camera_response.header.stamp)

            (
                color, depth_raw, color_message, depth_message, info, depth_info,
                camera_pose_age_s,
            ) = node.snapshot(
                args.frame_timeout_s, camera_state_stamp_s, camera_state_received_at,
            )

            camera_after_received_at = time.monotonic()
            camera_after_response = node.entity_state_response(CAMERA_ENTITY)
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

            height, width = color.shape[:2]
            if (
                (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT)
                or depth_raw.shape[:2] != (height, width)
                or (info.height, info.width) != (height, width)
                or (depth_info.height, depth_info.width) != (height, width)
            ):
                rejected.append({"sample_id": sample.sample_id, "reason": "RGB_DEPTH_CAMERAINFO_SIZE_MISMATCH"})
                continue
            if (not np.allclose(info.k, depth_info.k, atol=1e-6)
                    or not np.allclose(info.d, depth_info.d, atol=1e-6)):
                rejected.append({"sample_id": sample.sample_id, "reason": "RGB_DEPTH_CALIBRATION_MISMATCH"})
                continue
            fx, fy, cx, cy = (
                float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]),
            )
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
            vertices_local[:, 2] -= 0.05
            points_camera = pose_points_to_camera(
                vertices_local, target_position, target_q, (0.0, 0.0, 0.0),
                optical_position, optical_q,
            )
            try:
                bbox = projected_bbox(points_camera, intrinsics, width, height, distortion=distortion)
                depth_m = depth_to_meters(depth_raw, depth_message.encoding)
                visibility = depth_visibility(
                    points_camera, mesh.triangles, intrinsics, depth_m, distortion,
                )
                if visibility["visible_fraction"] < MIN_VISIBLE_FRACTION:
                    raise CaptureRejected(
                        f"OCCLUDED_OR_DEPTH_MISMATCH_{visibility['visible_fraction']:.3f}"
                    )
            except CaptureRejected as error:
                rejected.append({"sample_id": sample.sample_id, "reason": str(error)})
                continue

            normalized = yolo_bbox(bbox, width, height)
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
            preview = color.copy()
            cv2.rectangle(preview, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"{sample.scenario_id} azimuth={sample.azimuth_deg}deg",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
            preview_path = output_dir / "preview" / "samples" / f"{sample.sample_id}.jpg"
            cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

            rgb_stamp = stamp_seconds(color_message.header.stamp)
            depth_stamp = stamp_seconds(depth_message.header.stamp)
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
                "camera_pose_source": f"GetEntityState({CAMERA_ENTITY})",
                "camera_pose_age_at_frame_s": camera_pose_age_s,
                "camera_pose_age_after_frame_s": time.monotonic() - camera_after_received_at,
                "camera_link_position_world_m": camera_link_position,
                "camera_link_orientation_world_xyzw": camera_link_q,
                "camera_optical_position_world_m": optical_position,
                "camera_optical_orientation_world_xyzw": optical_q,
                "camera_post_frame_delta_m": camera_position_delta,
                "camera_post_frame_delta_rad": camera_orientation_delta,
                "camera_horizontal_distance_m": camera_distance,
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
                "target_position_world_m": target_position,
                "target_orientation_world_xyzw": target_q,
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "bbox_yolo_xywh_normalized": normalized,
                "depth_visibility": visibility,
                "label_origin": (
                    "projected vertices of the approved suitcase DAE; aligned depth verifies "
                    "visibility; never a YOLO prediction"
                ),
            })

        accepted_train = sum(sample["split"] == "train" for sample in manifest_samples)
        accepted_val = sum(sample["split"] == "val" for sample in manifest_samples)
        status = (
            "READY_FOR_AUTHOR_REVIEW"
            if len(manifest_samples) >= 12 and accepted_train and accepted_val
            else "INCOMPLETE"
        )
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
                "target_moved": False,
            },
        })
        (output_dir / "metadata" / "capture_manifest.json").write_text(
            json.dumps(base_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        (output_dir / "metadata" / "rejected_samples.json").write_text(
            json.dumps(rejected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
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
            'revision 928c0be3d1366adf3ce3a82e4c432fc1b0171b46, render.py; '
            'camera-sweep concept only, no source code copied.\n'
        )
        (output_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
        create_contact_sheet(output_dir, manifest_samples)
        print(
            f"Dataset status: {status}; accepted={len(manifest_samples)}; "
            f"rejected={len(rejected)}"
        )
        print(f"Output: {output_dir}")
        return output_dir
    finally:
        node.destroy_node()
        if rclpy.ok():
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
    parser.add_argument("--max-pose-age-s", type=float, default=DEFAULT_MAX_POSE_AGE_S)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    world_path, mesh_path = Path(args.world).resolve(), Path(args.mesh).resolve()
    try:
        if args.target_entity != TARGET_ENTITY:
            raise ValueError(f"Dataset generator only supports the approved {TARGET_ENTITY} model")
        if args.capture and (not math.isfinite(args.max_pose_age_s) or args.max_pose_age_s <= 0.0):
            raise ValueError("--max-pose-age-s must be a positive finite number")
        world_hash, mesh_hash = check_scene_files(world_path, mesh_path)
        mesh = load_collada_geometry(mesh_path)
        target_position, target_q = target_model_pose(world_path)
        target_center = suitcase_visual_center(target_position, target_q, mesh)
        samples = pilot_samples(target_center)
        obstacles = validate_camera_clearance(world_path, samples)
        projected_boxes = projected_pilot_boxes(
            samples, target_position, target_q, target_center, mesh,
        )
        if not args.capture:
            print("OFFLINE PILOT PLAN ONLY — Gazebo and ROS are not contacted.")
            print(f"Approved world SHA-256: {world_hash}")
            print(f"Approved suitcase mesh SHA-256: {mesh_hash}")
            print(f"Mesh: {mesh.geometry_count} geometries, {mesh.vertices.shape[0]} vertices, {mesh.triangle_count} triangles")
            print(f"Static world collision boxes checked: {len(obstacles)}; camera clearance radius: {CAMERA_CLEARANCE_RADIUS_M:.2f} m")
            print(f"Planned camera samples: {len(samples)} ({sum(sample.split == 'train' for sample in samples)} train, {sum(sample.split == 'val' for sample in samples)} val)")
            print("Target visual center (world m): " + ", ".join(f"{value:.3f}" for value in target_center))
            print("Camera horizontal orbit distances (m): " + ", ".join(f"{distance:.1f}" for distance in PILOT_CAMERA_DISTANCES_M))
            print("Train azimuths (deg): " + ", ".join(str(value) for value in PILOT_TRAIN_AZIMUTHS_DEG))
            print("Validation azimuths (deg): " + ", ".join(str(value) for value in PILOT_VAL_AZIMUTHS_DEG))
            print(f"Projected bounding boxes inside {CAMERA_WIDTH}x{CAMERA_HEIGHT}: {len(projected_boxes)}/{len(samples)}")
            print("Camera moves; the approved suitcase remains at its production pose.")
            print("Split is for pilot review only; final data must split by capture scenario.")
            return 0
        run_capture(args, mesh, world_hash, mesh_hash)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"dataset generator: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
