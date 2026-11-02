# `robot_perception` — CP-00D.1 YOLO11n ONNX

This is an independent perception package in the canonical `symulacja/`
workspace. Its source was written there and does not copy or adapt code from
the historical `ros2_ws/` workspace.

## Runtime path

- `opencv_dnn.py` loads ONNX with OpenCV DNN and selects its CPU backend.
- `detection_pipeline.py` converts explicit RGB input to float32 NCHW, scales
  values to `[0, 1]`, letterboxes to a square with padding value 114, reverses
  box scaling/padding, filters confidence and class, then applies classwise
  NMS.
- `scripts/yolo11n_node` subscribes to `sensor_msgs/Image` with best-effort,
  volatile QoS and depth one, and accepts `rgb8`/`bgr8` via `cv_bridge`. It publishes
  `vision_msgs/Detection2DArray` on `/yolo/detections` and the
  same-header RGB overlay on `/piracer/yolo/debug_image`. It also publishes
  per-frame ROS and monotonic timing diagnostics on `/piracer/yolo/timing`.
- In the generic COCO profile, an empty `class_filter`, `suitcase`, or `28`
  selects COCO `suitcase`, class ID 28. The full-system FT01 profile instead
  selects class ID 0 at confidence `0.35`, as configured in
  `config/detector_ft01_test.yaml`.
- `scripts/suitcase_point_node` synchronizes detections, depth, and color and
  depth `CameraInfo` messages by exact timestamp. It checks the qualified
  camera frame and calibration, samples depth at each detection center, and
  publishes the nearest valid 3D camera-frame point on
  `/piracer/follow/target_point`. If no valid point is available, it publishes
  nothing. The node does not track detections between frames.
- `piracer_sim_base.launch.py` defaults this backend to the retained model under
  `verification/yolo11n_same_frame_20261008/weights/yolo11n.onnx` when no
  explicit `yolo_model_path` is supplied. YOLO remains disabled by default in
  the base launch. The full-system launch starts `yolo11n_node` with
  `robot_perception/config/detector_ft01_test.yaml`.

The retained baseline ONNX has static input `images`: float32
`[1, 3, 640, 640]`, and output `output0`: `[1, 84, 8400]` (`cx, cy, w, h`
plus 80 COCO scores). The offline fine-tuned model uses the same input and
output `output0`: `[1, 5, 8400]` (four box channels plus one suitcase score,
class ID 0). Both layouts are decoded by the same pipeline. The full-system
`piracer_sim.launch.py` loads `config/detector_ft01_test.yaml`, which selects
this class-0 model at confidence `0.35`. Its weights are delivered externally
and must be present at the absolute path recorded in that YAML:
`/home/zygar/dev_praca_inz/symulacja/verification/yolo_ft01_20261012/runs/yolo11n_suitcase_ft01_final/weights/best.onnx`.
To relocate those weights, edit `model_path` in
`robot_perception/config/detector_ft01_test.yaml`, rebuild `robot_perception`,
and source the resulting install. The full launcher has no model-path override.
The base launch remains disabled by default and does not automatically select
the FT01 profile. To use that profile with the base launch, pass the YAML
through `yolo_params_file:=...` while setting `yolo_backend:=yolo11n_dnn` and
`yolo_enabled:=true`.

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

For a fresh checkout, obtain the retained COCO `yolo11n.onnx` separately from
the project author or the project's external artifact archive; Git and the
package build do not download or include it. Copy it to a persistent local
path and verify its SHA-256:
`ebc1c3f07246d5029ea165f30189a791a3140890f635a19c54547e3e4dae7c66`
(10,701,966 bytes). The current host keeps this file at
`/home/zygar/dev_praca_inz/symulacja/verification/yolo11n_same_frame_20261008/weights/yolo11n.onnx`.
The full-system `piracer_state_manager/piracer_sim.launch.py` reads the
installed `robot_perception/config/detector_ft01_test.yaml` and does not expose
a `yolo_model_path` override. Edit `model_path` in that YAML, rebuild the
package, and source the resulting install when relocating FT01 weights. The base
`piracer_sim_base.launch.py` supports `yolo_model_path` or
`yolo_params_file` when using `yolo_backend:=yolo11n_dnn` and
`yolo_enabled:=true`. Its built-in DNN default is the retained COCO artifact
and class `suitcase` (class ID 28); the FT01 profile uses class ID 0. Use a
compatible OpenCV environment as described above. Do not add the weights
to Git.

Unit tests use synthetic tensors and in-memory messages. They are separate
from the recorded real-model offline inference. No live ROS or Gazebo result is
claimed by this checkpoint.
