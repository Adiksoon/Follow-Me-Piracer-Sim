import math
from copy import deepcopy

import numpy as np
import pytest
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray

from robot_perception.rgbd_geometry import CAMERA_FRAME_ID, estimate_detection_points


def test_floor_center_projects_each_detection_and_preserves_source_order_and_stamp():
    depth = np.full((4, 5), 2.0, dtype=np.float32)
    depth[1, 1] = 3.0
    array, image, color_info, depth_info = _bundle(
        depth,
        [
            _detection(1.9, 1.2, header=_header()),
            _detection(3.1, 2.8, header=_header()),
        ],
    )

    estimates = estimate_detection_points(array, image, color_info, depth_info)

    assert [item.source_index for item in estimates] == [0, 1]
    assert estimates[0].source_detection is array.detections[0]
    assert estimates[0].valid
    assert estimates[0].point.point.x == pytest.approx(-0.75)
    assert estimates[0].point.point.y == pytest.approx(-0.375)
    assert estimates[0].point.point.z == pytest.approx(3.0)
    assert estimates[1].valid
    assert estimates[1].point.point.x == pytest.approx(1.5)
    assert estimates[1].point.point.y == pytest.approx(0.25)
    assert estimates[1].point.point.z == pytest.approx(2.0)
    assert estimates[0].point.header.frame_id == CAMERA_FRAME_ID
    assert estimates[0].point.header.stamp == array.header.stamp


def test_static_ideal_target_metric_reference_round_trip():
    expected_x, expected_y, expected_z = 0.45, 0.0, 2.25
    width, height = 640, 480
    fx = fy = 320.0
    cx, cy = 320.0, 240.0
    pixel_x = fx * expected_x / expected_z + cx
    pixel_y = fy * expected_y / expected_z + cy

    assert pixel_x.is_integer()
    assert pixel_y.is_integer()
    target_range = math.hypot(expected_x, expected_z)
    assert 2.0 <= target_range <= 2.5

    depth = np.zeros((height, width), dtype=np.float32)
    depth[int(pixel_y), int(pixel_x)] = expected_z
    detection = _detection(
        pixel_x, pixel_y, size_x=32.0, size_y=48.0, header=_header()
    )
    array, image, color_info, depth_info = _bundle(depth, [detection])
    for camera_info in (color_info, depth_info):
        camera_info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert result.valid
    assert pixel_x > cx
    assert result.point.point.x > 0.0
    assert result.point.point.x == pytest.approx(expected_x, abs=1e-6)
    assert result.point.point.y == pytest.approx(expected_y, abs=1e-6)
    assert result.point.point.z == pytest.approx(expected_z, abs=1e-6)
    assert result.point.header.frame_id == CAMERA_FRAME_ID
    assert result.point.header.stamp == array.header.stamp


def test_last_image_pixel_is_valid():
    array, image, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32), [_detection(4.1, 3.9, header=_header())]
    )

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert result.valid
    assert result.point.point.z == pytest.approx(1.0)


@pytest.mark.parametrize(
    "center_x,center_y",
    [(-0.2, 1.0), (5.0, 1.0), (1.0, -0.2), (1.0, 4.0)],
)
def test_out_of_image_center_is_invalid_without_clamping(center_x, center_y):
    array, image, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32),
        [_detection(center_x, center_y, header=_header())],
    )

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert not result.valid
    assert result.point is None
    assert result.invalid_reason == "CENTER_OUT_OF_BOUNDS"


@pytest.mark.parametrize(
    "center_x,center_y,size_x,size_y",
    [
        (math.nan, 1.0, 1.0, 1.0), (math.inf, 1.0, 1.0, 1.0),
        (1.0, math.nan, 1.0, 1.0), (1.0, -math.inf, 1.0, 1.0),
        (1.0, 1.0, math.nan, 1.0), (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, -1.0, 1.0), (1.0, 1.0, 1.0, math.inf),
        (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, -1.0),
    ],
)
def test_nonfinite_or_nonpositive_box_is_invalid(center_x, center_y, size_x, size_y):
    array, image, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32),
        [_detection(center_x, center_y, size_x, size_y, _header())],
    )

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert result.point is None
    assert result.invalid_reason == "INVALID_BOX"


@pytest.mark.parametrize("center_depth", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_invalid_center_depth_does_not_fall_back_to_valid_neighbors(center_depth):
    depth = np.full((4, 5), 2.0, dtype=np.float32)
    depth[1, 1] = center_depth
    array, image, color_info, depth_info = _bundle(
        depth, [_detection(1.0, 1.0, header=_header())]
    )

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert result.point is None
    assert result.invalid_reason == "INVALID_CENTER_DEPTH"
    assert np.count_nonzero(np.isfinite(depth) & (depth > 0.0)) > 0


def test_missing_input_returns_explicit_invalid_results_for_each_detection():
    array, _, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32),
        [_detection(1.0, 1.0, header=_header()), _detection(2.0, 2.0, header=_header())],
    )

    estimates = estimate_detection_points(array, None, color_info, depth_info)

    assert len(estimates) == 2
    assert all(item.point is None and item.invalid_reason == "MISSING_INPUT" for item in estimates)


@pytest.mark.parametrize(
    "changed",
    ["depth_stamp", "depth_frame", "depth_info_stamp", "dimensions", "skew", "distortion"],
)
def test_mismatched_input_metadata_is_invalid(changed):
    array, image, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32), [_detection(1.0, 1.0, header=_header())]
    )
    if changed == "depth_stamp":
        image.header.stamp.nanosec += 1
    elif changed == "depth_frame":
        image.header.frame_id = "other_frame"
    elif changed == "depth_info_stamp":
        depth_info.header.stamp.sec += 1
    elif changed == "skew":
        depth_info.k[1] = 0.5
    elif changed == "distortion":
        depth_info.d[0] = 0.1
    else:
        depth_info.width += 1

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert result.point is None
    assert result.invalid_reason in (
        "UNSYNCHRONIZED_INPUTS",
        "INVALID_DEPTH_IMAGE",
        "INVALID_CAMERA_INFO",
    )


def test_mismatched_detection_stamp_is_invalid_without_reordering_results():
    array, image, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32),
        [_detection(1.0, 1.0, header=_header()), _detection(2.0, 2.0, header=_header(sec=9))],
    )

    estimates = estimate_detection_points(array, image, color_info, depth_info)

    assert estimates[0].source_detection is array.detections[0]
    assert estimates[1].source_detection is array.detections[1]
    assert estimates[0].valid
    assert estimates[1].point is None
    assert estimates[1].invalid_reason == "UNSYNCHRONIZED_DETECTION"


def test_camera_model_mismatch_is_invalid():
    array, image, color_info, depth_info = _bundle(
        np.ones((4, 5), dtype=np.float32), [_detection(1.0, 1.0, header=_header())]
    )
    depth_info.k[0] += 1.0

    result = estimate_detection_points(array, image, color_info, depth_info)[0]

    assert result.point is None
    assert result.invalid_reason == "INVALID_CAMERA_INFO"


def _header(sec=12):
    header = Header()
    header.frame_id = CAMERA_FRAME_ID
    header.stamp.sec = sec
    header.stamp.nanosec = 34
    return header


def _detection(center_x, center_y, size_x=1.0, size_y=1.0, header=None):
    detection = Detection2D()
    detection.header = header or _header()
    detection.bbox.center.position.x = center_x
    detection.bbox.center.position.y = center_y
    detection.bbox.size_x = size_x
    detection.bbox.size_y = size_y
    return detection


def _bundle(depth, items):
    bridge = CvBridge()
    header = _header()
    array = Detection2DArray()
    array.header = deepcopy(header)
    array.detections = items
    image = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
    image.header = deepcopy(header)
    color_info = _camera_info(deepcopy(header), depth.shape[1], depth.shape[0])
    depth_info = _camera_info(deepcopy(header), depth.shape[1], depth.shape[0])
    return array, image, color_info, depth_info


def _camera_info(header, width, height):
    info = CameraInfo()
    info.header = header
    info.width = width
    info.height = height
    info.k = [2.0, 0.0, 1.5, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0]
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return info
