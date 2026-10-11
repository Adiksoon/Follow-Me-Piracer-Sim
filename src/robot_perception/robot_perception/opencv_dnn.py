"""OpenCV DNN inference adapter for the static YOLO11n ONNX export."""

from pathlib import Path

import cv2

from robot_perception.detection_pipeline import Detection, decode_yolo11_output, preprocess_image


class OpenCVDNNDetector:
    """Run one RGB frame through a CPU OpenCV DNN network and decode it."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_iou: float = 0.5,
        class_filter: set[int] | None = None,
        num_threads: int = 1,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        cv2.setNumThreads(num_threads)
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as error:
            raise RuntimeError(
                f"OpenCV DNN could not load the ONNX model {path}: {error}"
            ) from error
        if self._net.empty():
            raise RuntimeError(f"OpenCV DNN loaded an empty network: {path}")

        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        output_names = tuple(self._net.getUnconnectedOutLayersNames())
        if len(output_names) != 1:
            raise ValueError(
                "YOLO11n ONNX must expose exactly one output; "
                f"found {len(output_names)}: {output_names}"
            )

        self._output_name = output_names[0]
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._nms_iou = nms_iou
        self._class_filter = class_filter
        self.last_output_shape: tuple[int, ...] | None = None

    def infer_rgb(self, image_rgb) -> list[Detection]:
        """Infer from a uint8 RGB image and return source-frame pixel boxes."""
        tensor, transform = preprocess_image(
            image_rgb,
            target_size=self._input_size,
            source_order="RGB",
        )
        self._net.setInput(tensor)
        output = self._net.forward(self._output_name)
        self.last_output_shape = tuple(int(size) for size in output.shape)
        return decode_yolo11_output(
            output,
            transform,
            confidence_threshold=self._confidence_threshold,
            nms_iou=self._nms_iou,
            class_filter=self._class_filter,
            class_names={28: "suitcase"},
        )
