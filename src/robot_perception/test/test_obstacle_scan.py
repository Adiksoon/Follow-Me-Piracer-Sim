import math

import numpy as np
import pytest
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo

from robot_perception.obstacle_scan import depth_obstacle_scan


def camera_scene(barrier_distance=None, encoding="32FC1"):
    info = CameraInfo()
    info.width, info.height = 96, 80
    info.k = [80., 0., 47.5, 0., 80., 39.5, 0., 0., 1.]
    info.header.frame_id = "sim_camera_optical_frame"
    info.header.stamp.sec = 5
    v, u = np.mgrid[:80, :96]
    depth = np.full((80, 96), 5., dtype=np.float32)
    below = v > 39.5
    depth[below] = .32*80/(v[below]-39.5)
    if barrier_distance is not None:
        d = barrier_distance + .246 - .236
        height = .32 - (v-39.5)*d/80
        in_box = ((height >= 0) & (height <= .18) & (abs((u-47.5)*d/80) < .6))
        depth[in_box] = d
    if encoding == "16UC1":
        depth = (depth*1000).astype(np.uint16)
    image = CvBridge().cv2_to_imgmsg(depth, encoding)
    image.header = info.header
    transform = TransformStamped()
    transform.header.frame_id = "base_link"
    transform.child_frame_id = image.header.frame_id
    transform.transform.translation.x, transform.transform.translation.z = .236, .32
    transform.transform.rotation.x = -.5
    transform.transform.rotation.y = .5
    transform.transform.rotation.z = -.5
    transform.transform.rotation.w = .5
    return image, info, transform


def test_valid_floor_only_scan_has_no_hits_and_keeps_source_stamp():
    image, info, tf = camera_scene()
    scan = depth_obstacle_scan(image, info, tf)
    assert all(math.isinf(v) for v in scan.ranges)
    assert scan.header.stamp == image.header.stamp and scan.header.frame_id == "base_link"


@pytest.mark.parametrize("encoding", ["32FC1", "16UC1"])
def test_low_barrier_two_meters_in_front_of_footprint_is_detected(encoding):
    image, info, tf = camera_scene(2., encoding)
    scan = depth_obstacle_scan(image, info, tf)
    hits = [v for v in scan.ranges if math.isfinite(v)]
    assert hits and min(hits) == pytest.approx(2.246, abs=.01)
    assert math.isinf(scan.ranges[0]) and math.isinf(scan.ranges[-1])


@pytest.mark.parametrize("invalid", ["zero", "nan", "stamp", "frame", "intrinsics", "rotation"])
def test_invalid_frame_does_not_generate_a_fresh_clear_scan(invalid):
    image, info, tf = camera_scene(2.)
    if invalid in ("zero", "nan"):
        values = np.full((80, 96), 0. if invalid == "zero" else np.nan, dtype=np.float32)
        image = CvBridge().cv2_to_imgmsg(values, "32FC1")
        image.header = info.header
    elif invalid == "stamp":
        from copy import deepcopy
        info = deepcopy(info)
        info.header.stamp.nanosec = 1
    elif invalid == "frame":
        tf.child_frame_id = "other_camera"
    elif invalid == "intrinsics":
        info.k[0] = 0.
    else:
        tf.transform.rotation.w = 0.
    with pytest.raises(ValueError):
        depth_obstacle_scan(image, info, tf)


def test_node_drops_invalid_frame_instead_of_publishing_a_clear_observation():
    import importlib.util
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from types import SimpleNamespace as NS
    source = Path(__file__).parents[1] / "scripts/obstacle_scan_node"
    loader = SourceFileLoader("obstacle_node_test", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    image, info, tf = camera_scene()
    node = object.__new__(module.ObstacleScanNode)
    node._bridge = CvBridge()
    node._buffer = NS(lookup_transform=lambda *_: tf)
    published = []
    node._publisher = NS(publish=published.append)
    node.get_parameter = lambda name: NS(value={"min_height": .03, "max_height": .5}[name])
    node.get_logger = lambda: NS(warning=lambda *_args, **_kwargs: None)
    node._on_depth(image, info)
    assert len(published) == 1 and all(math.isinf(v) for v in published[0].ranges)
    invalid = CvBridge().cv2_to_imgmsg(np.full((80, 96), np.nan, dtype=np.float32), "32FC1")
    invalid.header = image.header
    node._on_depth(invalid, info)
    assert len(published) == 1
