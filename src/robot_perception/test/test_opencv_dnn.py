import numpy as np
import pytest

from robot_perception.opencv_dnn import OpenCVDNNDetector


def test_opencv_dnn_detector_preprocesses_and_decodes_suitcase(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test placeholder")
    output = np.zeros((1, 84, 1), dtype=np.float32)
    output[0, :4, 0] = (320, 320, 100, 100)
    output[0, 4 + 28, 0] = 0.82

    class FakeNetwork:
        def __init__(self):
            self.input_tensor = None
            self.backend = None
            self.target = None

        def empty(self):
            return False

        def setPreferableBackend(self, backend):
            self.backend = backend

        def setPreferableTarget(self, target):
            self.target = target

        def getUnconnectedOutLayersNames(self):
            return ("output0",)

        def setInput(self, tensor):
            self.input_tensor = tensor.copy()

        def forward(self, output_name):
            assert output_name == "output0"
            return output

    network = FakeNetwork()
    monkeypatch.setattr(
        "robot_perception.opencv_dnn.cv2.dnn.readNetFromONNX",
        lambda _path: network,
    )

    detector = OpenCVDNNDetector(
        str(model_path),
        input_size=640,
        confidence_threshold=0.35,
        nms_iou=0.5,
        class_filter={28},
        num_threads=1,
    )
    image_rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    image_rgb[:, :, 0] = 255

    detections = detector.infer_rgb(image_rgb)

    assert detector.last_output_shape == (1, 84, 1)
    assert network.input_tensor.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(network.input_tensor[0, :, 300, 300], (1.0, 0.0, 0.0))
    assert len(detections) == 1
    assert detections[0].class_id == 28
    assert detections[0].class_name == "suitcase"
    assert detections[0].score == pytest.approx(0.82)
    assert (detections[0].x_min, detections[0].y_min) == (270.0, 130.0)
    assert (detections[0].x_max, detections[0].y_max) == (370.0, 230.0)
