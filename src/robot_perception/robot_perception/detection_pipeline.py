"""Image preprocessing and output decoding for the YOLO11n ONNX adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_name: str = ""


def _validate_rgb_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image dimensions must be positive")


def letterbox_rgb(
    image: np.ndarray,
    target_size: int,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize an RGB image into a square canvas, retaining its aspect ratio."""
    _validate_rgb_image(image)
    if target_size < 1:
        raise ValueError("target_size must be positive")
    if not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be in [0, 255]")

    source_height, source_width = image.shape[:2]
    scale = min(target_size / source_width, target_size / source_height)
    resized_width = max(1, min(target_size, int(round(source_width * scale))))
    resized_height = max(1, min(target_size, int(round(source_height * scale))))
    pad_width = target_size - resized_width
    pad_height = target_size - resized_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    canvas[
        pad_top:pad_top + resized_height,
        pad_left:pad_left + resized_width,
    ] = resized

    transform = LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_size,
        target_height=target_size,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return canvas, transform


def preprocess_image(
    image: np.ndarray,
    target_size: int,
    source_order: str = "RGB",
) -> tuple[np.ndarray, LetterboxTransform]:
    """Return normalized RGB float32 input in NCHW layout.

    ROS ``rgb8`` should use the default ``source_order="RGB"``. BGR callers
    can select ``source_order="BGR"`` to make the channel swap explicit.
    """
    _validate_rgb_image(image)
    order = source_order.upper()
    if order == "RGB":
        rgb = image
    elif order == "BGR":
        rgb = image[:, :, ::-1]
    else:
        raise ValueError("source_order must be RGB or BGR")

    letterboxed, transform = letterbox_rgb(rgb, target_size)
    tensor = np.ascontiguousarray(letterboxed.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, transform


def map_box_to_source(
    box_xyxy: Iterable[float],
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """Undo the recorded resize/padding and clip a box to the source frame."""
    x_min, y_min, x_max, y_max = (float(value) for value in box_xyxy)
    x_min = (x_min - transform.pad_left) / transform.scale_x
    x_max = (x_max - transform.pad_left) / transform.scale_x
    y_min = (y_min - transform.pad_top) / transform.scale_y
    y_max = (y_max - transform.pad_top) / transform.scale_y
    x_min = min(max(x_min, 0.0), float(transform.source_width))
    x_max = min(max(x_max, 0.0), float(transform.source_width))
    y_min = min(max(y_min, 0.0), float(transform.source_height))
    y_max = min(max(y_max, 0.0), float(transform.source_height))
    return x_min, y_min, x_max, y_max


def classwise_nms(
    detections: Iterable[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Greedy NMS, independently per class, with stable score ordering."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    candidates = list(detections)
    kept: list[Detection] = []
    for class_id in sorted({item.class_id for item in candidates}):
        class_candidates = [item for item in candidates if item.class_id == class_id]
        class_candidates.sort(key=lambda item: item.score, reverse=True)
        while class_candidates:
            best = class_candidates.pop(0)
            kept.append(best)
            survivors = []
            for candidate in class_candidates:
                intersection_width = max(
                    0.0,
                    min(best.x_max, candidate.x_max) - max(best.x_min, candidate.x_min),
                )
                intersection_height = max(
                    0.0,
                    min(best.y_max, candidate.y_max) - max(best.y_min, candidate.y_min),
                )
                intersection = intersection_width * intersection_height
                best_area = max(0.0, best.x_max - best.x_min) * max(
                    0.0, best.y_max - best.y_min
                )
                candidate_area = max(0.0, candidate.x_max - candidate.x_min) * max(
                    0.0, candidate.y_max - candidate.y_min
                )
                union = best_area + candidate_area - intersection
                iou = intersection / union if union > 0.0 else 0.0
                if iou <= iou_threshold:
                    survivors.append(candidate)
            class_candidates = survivors

    return sorted(kept, key=lambda item: item.score, reverse=True)


def decode_yolo11_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    confidence_threshold: float = 0.35,
    nms_iou: float = 0.5,
    class_filter: set[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> list[Detection]:
    """Decode YOLO11 ``[1, 4 + classes, N]`` center/size plus class scores.

    The selected ONNX artifacts use four box channels followed by class scores,
    without an objectness channel. This supports both the 80-class COCO export
    and the fine-tuned single-class export.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("nms_iou must be in [0, 1]")

    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] < 5:
        raise ValueError(
            "expected YOLO11 output [1, 4 + classes, N] or [4 + classes, N]"
        )
    if not np.isfinite(values).all():
        raise ValueError("model output contains non-finite values")
    if values.shape[1] == 0:
        return []

    class_scores = values[4:, :]
    class_ids = np.argmax(class_scores, axis=0)
    scores = class_scores[class_ids, np.arange(values.shape[1])]
    selected = scores >= confidence_threshold
    if class_filter is not None:
        selected &= np.isin(class_ids, tuple(class_filter))

    candidates: list[Detection] = []
    names = class_names or {}
    for index in np.flatnonzero(selected):
        center_x, center_y, width, height = values[:4, index]
        if width <= 0.0 or height <= 0.0:
            continue
        model_box = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        source_box = map_box_to_source(model_box, transform)
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            continue
        candidates.append(
            Detection(
                class_id=int(class_ids[index]),
                score=float(scores[index]),
                x_min=source_box[0],
                y_min=source_box[1],
                x_max=source_box[2],
                y_max=source_box[3],
                class_name=names.get(int(class_ids[index]), ""),
            )
        )

    return classwise_nms(candidates, nms_iou)


def draw_debug_image(
    rgb_image: np.ndarray,
    detections: Iterable[Detection],
) -> np.ndarray:
    """Draw red boxes onto an RGB copy for diagnostic-image publication."""
    _validate_rgb_image(rgb_image)
    debug_image = rgb_image.copy()
    height, width = debug_image.shape[:2]
    for detection in detections:
        x_min = max(0, min(width - 1, int(round(detection.x_min))))
        y_min = max(0, min(height - 1, int(round(detection.y_min))))
        x_max = max(0, min(width - 1, int(round(detection.x_max))))
        y_max = max(0, min(height - 1, int(round(detection.y_max))))
        cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        class_name = detection.class_name or f"class {detection.class_id}"
        label = f"{class_name} {detection.score:.2f}"
        cv2.putText(
            debug_image,
            label,
            (x_min, max(12, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return debug_image
