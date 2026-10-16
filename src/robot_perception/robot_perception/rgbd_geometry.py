"""Stateless per-detection RGB-D geometry for the qualified simulator frame."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray


CAMERA_FRAME_ID = "sim_camera_optical_frame"


@dataclass(frozen=True)
class DetectionPointEstimate:
    """A source detection and either its valid point or an explicit reason."""

    source_index: int
    source_detection: Detection2D
    point: PointStamped | None
    invalid_reason: str | None

    @property
    def valid(self) -> bool:
        return self.point is not None


def _invalid_estimates(detections, reason: str) -> list[DetectionPointEstimate]:
    return [
        DetectionPointEstimate(index, detection, None, reason)
        for index, detection in enumerate(detections)
    ]


def _same_header(left, right) -> bool:
    return (
        left.frame_id == right.frame_id
        and left.stamp.sec == right.stamp.sec
        and left.stamp.nanosec == right.stamp.nanosec
    )


def _camera_model_is_usable(color_info: CameraInfo, depth_info: CameraInfo) -> bool:
    color_k = np.asarray(color_info.k, dtype=np.float64)
    depth_k = np.asarray(depth_info.k, dtype=np.float64)
    color_d = np.asarray(color_info.d, dtype=np.float64)
    depth_d = np.asarray(depth_info.d, dtype=np.float64)
    if color_k.size != 9 or depth_k.size != 9:
        return False
    if not np.isfinite(color_k).all() or not np.isfinite(depth_k).all():
        return False
    if color_k[0] <= 0.0 or color_k[4] <= 0.0:
        return False
    canonical_entries = ((1, 0.0), (3, 0.0), (6, 0.0), (7, 0.0), (8, 1.0))
    if any(color_k[index] != value for index, value in canonical_entries):
        return False
    if color_k.tolist() != depth_k.tolist() or color_d.tolist() != depth_d.tolist():
        return False
    if not np.isfinite(color_d).all() or np.any(color_d != 0.0):
        return False
    return True


def estimate_detection_points(
    detections: Detection2DArray,
    depth_image: Image | None,
    color_camera_info: CameraInfo | None,
    depth_camera_info: CameraInfo | None,
    bridge: CvBridge | None = None,
) -> list[DetectionPointEstimate]:
    """Estimate one camera-frame point per detection in one exact-stamp bundle.

    This bounded offline converter expects already-associated messages from the
    qualified simulator camera. It does not synchronize streams or decide how
    old a complete bundle may be. Invalid detections never receive a point.
    """
    if detections is None:
        raise ValueError("Detection2DArray is required to preserve source correspondence")
    source = detections.detections
    if not source:
        return []

    if depth_image is None or color_camera_info is None or depth_camera_info is None:
        return _invalid_estimates(source, "MISSING_INPUT")

    common_header = detections.header
    if common_header.frame_id != CAMERA_FRAME_ID or not all(
        _same_header(common_header, message.header)
        for message in (depth_image, color_camera_info, depth_camera_info)
    ):
        return _invalid_estimates(source, "UNSYNCHRONIZED_INPUTS")

    width, height = depth_image.width, depth_image.height
    if (
        width <= 0
        or height <= 0
        or color_camera_info.width != width
        or color_camera_info.height != height
        or depth_camera_info.width != width
        or depth_camera_info.height != height
        or depth_image.encoding != "32FC1"
    ):
        return _invalid_estimates(source, "INVALID_DEPTH_IMAGE")

    if not _camera_model_is_usable(color_camera_info, depth_camera_info):
        return _invalid_estimates(source, "INVALID_CAMERA_INFO")

    try:
        image_bridge = bridge or CvBridge()
        depth = np.asarray(
            image_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        )
    except CvBridgeError:
        return _invalid_estimates(source, "INVALID_DEPTH_IMAGE")
    if depth.shape != (height, width) or depth.dtype.kind != "f" or depth.dtype.itemsize != 4:
        return _invalid_estimates(source, "INVALID_DEPTH_IMAGE")

    k = color_camera_info.k
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    estimates = []
    for index, detection in enumerate(source):
        if not _same_header(common_header, detection.header):
            estimates.append(
                DetectionPointEstimate(index, detection, None, "UNSYNCHRONIZED_DETECTION")
            )
            continue

        box = detection.bbox
        center_x = float(box.center.position.x)
        center_y = float(box.center.position.y)
        size_x = float(box.size_x)
        size_y = float(box.size_y)
        if not all(math.isfinite(value) for value in (center_x, center_y, size_x, size_y)):
            estimates.append(DetectionPointEstimate(index, detection, None, "INVALID_BOX"))
            continue
        if size_x <= 0.0 or size_y <= 0.0:
            estimates.append(DetectionPointEstimate(index, detection, None, "INVALID_BOX"))
            continue

        pixel_x, pixel_y = math.floor(center_x), math.floor(center_y)
        if pixel_x < 0 or pixel_y < 0 or pixel_x >= width or pixel_y >= height:
            estimates.append(
                DetectionPointEstimate(index, detection, None, "CENTER_OUT_OF_BOUNDS")
            )
            continue

        z = float(depth[pixel_y, pixel_x])
        if not math.isfinite(z) or z <= 0.0:
            estimates.append(
                DetectionPointEstimate(index, detection, None, "INVALID_CENTER_DEPTH")
            )
            continue

        point = PointStamped()
        point.header = common_header
        point.point.x = (pixel_x - cx) * z / fx
        point.point.y = (pixel_y - cy) * z / fy
        point.point.z = z
        estimates.append(DetectionPointEstimate(index, detection, point, None))

    return estimates
