import numpy as np
import pytest
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from robot_perception.detection_pipeline import Detection
from robot_perception.ros_messages import (
    image_message_to_rgb,
    publish_outputs,
    to_detection_array,
)


def test_detection_message_preserves_header_box_and_score():
    header = Header()
    header.frame_id = "camera_rgb_optical_frame"
    header.stamp.sec = 12
    header.stamp.nanosec = 34
    source = [Detection(28, 0.875, 10.0, 20.0, 110.0, 220.0, "suitcase")]

    message = to_detection_array(source, header)

    assert message.header.frame_id == header.frame_id
    assert len(message.detections) == 1
    detection = message.detections[0]
    assert detection.header.stamp.sec == 12
    assert detection.id == ""
    assert detection.bbox.center.position.x == 60.0
    assert detection.bbox.center.position.y == 120.0
    assert detection.bbox.size_x == 100.0
    assert detection.bbox.size_y == 200.0
    assert detection.results[0].hypothesis.class_id == "suitcase"
    assert detection.results[0].hypothesis.score == 0.875


def test_rgb8_image_message_round_trips_and_keeps_channel_order():
    bridge = CvBridge()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (240, 80, 10)
    message = bridge.cv2_to_imgmsg(image, encoding="rgb8")
    message.header.frame_id = "camera_rgb_optical_frame"

    decoded = image_message_to_rgb(message, bridge)

    np.testing.assert_array_equal(decoded, image)
    assert message.header.frame_id == "camera_rgb_optical_frame"


def test_publish_outputs_emits_detection_and_debug_with_source_header():
    bridge = CvBridge()
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[0, 0] = (240, 80, 10)
    image_message = bridge.cv2_to_imgmsg(source, encoding="rgb8")
    image_message.header.frame_id = "camera_rgb_optical_frame"
    image_message.header.stamp.sec = 12
    detection_publisher = _CapturePublisher()
    image_publisher = _CapturePublisher()
    detections = [Detection(28, 0.875, 4.0, 4.0, 24.0, 24.0)]

    publish_outputs(
        detection_publisher,
        image_publisher,
        image_message,
        source,
        detections,
        bridge,
    )

    assert len(detection_publisher.messages) == 1
    assert len(image_publisher.messages) == 1
    assert detection_publisher.messages[0].header.frame_id == "camera_rgb_optical_frame"
    debug_message = image_publisher.messages[0]
    assert debug_message.encoding == "rgb8"
    assert debug_message.header.stamp.sec == 12
    assert np.count_nonzero(bridge.imgmsg_to_cv2(debug_message, "rgb8")) > 0


def test_unsupported_image_encoding_is_rejected():
    message = Image()
    message.encoding = "mono8"
    with pytest.raises(ValueError, match="unsupported image encoding"):
        image_message_to_rgb(message, CvBridge())


class _CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)
