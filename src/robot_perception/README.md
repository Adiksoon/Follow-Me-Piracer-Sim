# `robot_perception` — CP-00D.1 YOLO11n ONNX

This is an independent detector package in the canonical `symulacja/`
workspace. Its source was written there and does not copy or adapt code from
the historical `ros2_ws/` workspace.

## Runtime path

- `opencv_dnn.py` loads ONNX with OpenCV DNN and selects its CPU backend.
- `detection_pipeline.py` converts explicit RGB input to float32 NCHW, scales
  values to `[0, 1]`, letterboxes to a square with padding value 114, reverses
  box scaling/padding, filters confidence and class, then applies classwise
  NMS.
- `scripts/yolo11n_node` subscribes to `sensor_msgs/Image` using sensor-data
  QoS and accepts `rgb8`/`bgr8` via `cv_bridge`. It publishes
  `vision_msgs/Detection2DArray` on `/piracer/yolo/detections` and the
  same-header RGB overlay on `/piracer/yolo/debug_image`.
- The approved filter is COCO `suitcase`, class ID 28. An empty
  `class_filter`, `suitcase`, or `28` selects that class.
- `piracer_sim.launch.py` defaults this backend to the retained model under
  `verification/yolo11n_same_frame_20261008/weights/yolo11n.onnx` when no
  explicit `yolo_model_path` is supplied. YOLO remains disabled by default in
  the launch.

The retained baseline ONNX has static input `images`: float32
`[1, 3, 640, 640]`, and output `output0`: `[1, 84, 8400]` (`cx, cy, w, h`
plus 80 COCO scores). The offline fine-tuned model uses the same input and
output `output0`: `[1, 5, 8400]` (four box channels plus one suitcase score,
class ID 0). Both layouts are decoded by the same pipeline. The isolated
`config/detector_ft01_test.yaml` selects the fine-tuned ONNX and class 0; pass
it as `yolo_params_file:=...` to `piracer_sim.launch.py` for a test run. The
default launch configuration continues to select the retained COCO model.

## Runtime compatibility

On this host, OpenCV 4.13.0 loads and executes the selected model with DNN CPU.
The system OpenCV 4.5.4 fails while importing its attention `Split` node, so
the installed system package alone is insufficient for this ONNX artifact.
The launch's YOLO11n branch uses the active `python3` environment; it must
resolve an OpenCV DNN build that supports this model. No CUDA provider,
Ultralytics runtime, or ONNX Runtime is used by the new node.

## Model and licensing

The ONNX and its source checkpoint remain verification artifacts and are not
copied into the ROS package. The model SHA-256 and local provenance are in the
CP-00D.1 report and `verification/yolo11n_same_frame_20261008/weights/`.
This package's new source remains MIT-licensed; the user-authorized academic
model use does not change the repository-wide license.

Unit tests use synthetic tensors and in-memory messages. They are separate
from the recorded real-model offline inference. No live ROS or Gazebo result is
claimed by this checkpoint.
