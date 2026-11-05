"""Project depth obstacles above the floor into a robot-frame laser scan."""

import math

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan


def depth_obstacle_scan(image, info, transform, bridge=None,
                        min_height=0.03, max_height=0.5):
    """Use the image stamp and static camera extrinsics; UNKNOWN bins stay infinite."""
    if (image.header != info.header or image.width != info.width
            or image.height != info.height or image.width <= 0 or image.height <= 0
            or image.encoding not in ("32FC1", "16UC1")):
        raise ValueError("invalid or unsynchronized depth camera inputs")
    k = np.asarray(info.k, dtype=float)
    if (not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0
            or any(float(d) != 0 for d in info.d)):
        raise ValueError("expected a rectified depth camera model")
    if (transform.child_frame_id != image.header.frame_id
            or transform.header.frame_id != "base_link"):
        raise ValueError("expected source camera to base_link transform")
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    t = transform.transform.translation
    if (not all(math.isfinite(v) for v in (x, y, z, w, t.x, t.y, t.z))
            or abs(x*x + y*y + z*z + w*w - 1) > .02):
        raise ValueError("invalid camera transform")
    depth = (bridge or CvBridge()).imgmsg_to_cv2(image, desired_encoding="passthrough")
    if depth.shape != (image.height, image.width):
        raise ValueError("depth dimensions do not match CameraInfo")
    # A sparse grid keeps processing inexpensive without selecting one image row.
    v, u = np.mgrid[0:image.height:8, 0:image.width:8]
    d = np.asarray(depth[::8, ::8], dtype=float)
    if image.encoding == "16UC1":
        d *= .001
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        raise ValueError("depth frame has no valid measurements")
    d, u, v = d[valid], u[valid], v[valid]
    camera = np.column_stack(((u - k[2])*d/k[0], (v - k[5])*d/k[4], d))
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    points = camera @ rotation.T + np.array([t.x, t.y, t.z])
    distance = np.hypot(points[:, 0], points[:, 1])
    obstacle = ((points[:, 0] > 0) & (points[:, 2] >= min_height)
                & (points[:, 2] <= max_height) & (distance >= .05) & (distance <= 3.))
    scan = LaserScan()
    scan.header.stamp = image.header.stamp
    scan.header.frame_id = "base_link"
    scan.angle_min = -math.pi/2
    scan.angle_max = math.pi/2
    scan.angle_increment = math.pi/180
    scan.scan_time = 1/30
    scan.range_min, scan.range_max = .05, 3.
    ranges = np.full(181, np.inf, dtype=np.float32)
    angles = np.arctan2(points[obstacle, 1], points[obstacle, 0])
    indices = np.rint((angles - scan.angle_min) / scan.angle_increment).astype(int)
    np.minimum.at(ranges, np.clip(indices, 0, 180), distance[obstacle])
    scan.ranges = ranges.tolist()
    return scan
