import numpy as np
import pytest

from robot_perception.detection_pipeline import (
    Detection,
    decode_yolo11_output,
    draw_debug_image,
    letterbox_rgb,
    map_box_to_source,
    preprocess_image,
)


def test_rgb_and_bgr_inputs_produce_the_same_rgb_tensor():
    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    rgb[:, :, 0] = 240
    rgb[:, :, 1] = 80
    rgb[:, :, 2] = 10
    bgr = rgb[:, :, ::-1].copy()

    rgb_tensor, _ = preprocess_image(rgb, 8, source_order="RGB")
    bgr_tensor, _ = preprocess_image(bgr, 8, source_order="BGR")

    np.testing.assert_array_equal(rgb_tensor, bgr_tensor)
    assert rgb_tensor.shape == (1, 3, 8, 8)
    assert rgb_tensor.dtype == np.float32
    assert rgb_tensor[0, :, 3, 3].tolist() == pytest.approx([240 / 255, 80 / 255, 10 / 255])


def test_letterbox_landscape_and_odd_padding_are_recorded_exactly():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    _, transform = letterbox_rgb(image, 640)
    assert (transform.resized_width, transform.resized_height) == (640, 360)
    assert (transform.pad_left, transform.pad_top) == (0, 140)

    odd_image = np.zeros((360, 641, 3), dtype=np.uint8)
    canvas, odd_transform = letterbox_rgb(odd_image, 640)
    assert canvas.shape == (640, 640, 3)
    assert odd_transform.resized_width == 640
    assert odd_transform.resized_height == 359
    assert odd_transform.pad_top == 140
    assert 640 - odd_transform.pad_top - odd_transform.resized_height == 141


def test_inverse_mapping_removes_padding_and_clips_to_source_bounds():
    _, transform = letterbox_rgb(np.zeros((360, 640, 3), dtype=np.uint8), 640)

    mapped = map_box_to_source((100, 120, 700, 620), transform)

    assert mapped == pytest.approx((100, 0, 640, 360))


def test_yolo_synthetic_output_uses_classwise_nms_and_filter():
    _, transform = letterbox_rgb(np.zeros((640, 640, 3), dtype=np.uint8), 640)
    output = np.zeros((1, 84, 3), dtype=np.float32)
    output[0, :4, 0] = (100, 100, 60, 60)
    output[0, :4, 1] = (102, 101, 60, 60)
    output[0, :4, 2] = (100, 100, 60, 60)
    output[0, 4 + 28, 0] = 0.90
    output[0, 4 + 28, 1] = 0.80
    output[0, 4 + 9, 2] = 0.70

    detections = decode_yolo11_output(
        output,
        transform,
        confidence_threshold=0.35,
        nms_iou=0.5,
        class_names={28: "suitcase", 9: "traffic light"},
    )

    assert [item.class_id for item in detections] == [28, 9]
    assert [item.score for item in detections] == pytest.approx([0.90, 0.70])
    assert [item.class_name for item in detections] == ["suitcase", "traffic light"]

    filtered = decode_yolo11_output(output, transform, class_filter={28})
    assert [item.class_id for item in filtered] == [28]


def test_yolo_single_class_output_decodes_suitcase_as_class_zero():
    _, transform = letterbox_rgb(np.zeros((360, 640, 3), dtype=np.uint8), 640)
    output = np.zeros((1, 5, 1), dtype=np.float32)
    output[0, :4, 0] = (320, 320, 200, 160)
    output[0, 4, 0] = 0.91

    detections = decode_yolo11_output(
        output,
        transform,
        class_filter={0},
        class_names={0: "suitcase"},
    )

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].class_name == "suitcase"
    assert detections[0].score == pytest.approx(0.91)
    assert (detections[0].x_min, detections[0].y_min) == (220.0, 100.0)
    assert (detections[0].x_max, detections[0].y_max) == (420.0, 260.0)


def test_yolo_synthetic_empty_and_invalid_outputs():
    _, transform = letterbox_rgb(np.zeros((640, 640, 3), dtype=np.uint8), 640)
    assert decode_yolo11_output(np.zeros((1, 84, 0)), transform) == []

    with pytest.raises(ValueError, match="4 \\+ classes"):
        decode_yolo11_output(np.zeros((1, 4, 3)), transform)

    non_finite = np.zeros((1, 84, 1), dtype=np.float32)
    non_finite[0, 4, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        decode_yolo11_output(non_finite, transform)


def test_debug_image_draws_into_copy_and_preserves_rgb_source():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = [Detection(28, 0.9, 4, 4, 24, 24)]

    debug = draw_debug_image(image, detections)

    assert not np.shares_memory(debug, image)
    assert np.count_nonzero(debug) > 0
    assert np.count_nonzero(image) == 0


def test_bad_channel_order_is_rejected():
    with pytest.raises(ValueError, match="source_order"):
        preprocess_image(np.zeros((2, 2, 3), dtype=np.uint8), 4, "GRAY")
