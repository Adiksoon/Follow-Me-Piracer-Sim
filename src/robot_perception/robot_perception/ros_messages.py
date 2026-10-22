"""ROS message adaptation for the model-independent Detection value type."""

from __future__ import annotations

from collections.abc import Iterable

from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from robot_perception.detection_pipeline import Detection, draw_debug_image


def image_message_to_rgb(image_message, bridge: CvBridge):
    """Convert a ROS image to the explicit RGB order used by preprocessing."""
    if image_message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding: {image_message.encoding!r}")
    return bridge.imgmsg_to_cv2(image_message, desired_encoding="rgb8")


def to_detection_array(detections: Iterable[Detection], header) -> Detection2DArray:
    """Convert source-image pixel boxes into the standard ROS 2 vision message."""
    result = Detection2DArray()
    result.header = header
    for item in detections:
        detection = Detection2D()
        detection.header = header
        detection.bbox.center.position.x = (item.x_min + item.x_max) / 2.0
        detection.bbox.center.position.y = (item.y_min + item.y_max) / 2.0
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = item.x_max - item.x_min
        detection.bbox.size_y = item.y_max - item.y_min

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = item.class_name or str(item.class_id)
        hypothesis.hypothesis.score = item.score
        detection.results.append(hypothesis)
        result.detections.append(detection)
    return result


def to_debug_image_message(image_message, rgb_image, detections, bridge: CvBridge):
    """Create an RGB diagnostic image and preserve the source header."""
    debug = draw_debug_image(rgb_image, detections)
    message = bridge.cv2_to_imgmsg(debug, encoding="rgb8")
    message.header = image_message.header
    return message
