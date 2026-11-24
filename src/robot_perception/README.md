# `robot_perception` - CP-00D.1 YOLO11n ONNX

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
  selects class ID 0 at confidence 0.90, as configured in
  `config/detector_ft01_test.yaml`.
- `scripts/suitcase_point_node` synchronizes detections, depth, and color and
  depth `CameraInfo` messages by exact timestamp. It checks the qualified
  camera frame and calibration, samples depth at each detection center, and
  rejects bounding boxes clipped by the image boundary and publishes the
  nearest valid 3D camera-frame point on
  `/piracer/follow/target_point`. If no valid point is available, it publishes
  nothing. The node does not track detections between frames.
- `scripts/obstacle_scan_node` samples the depth image across its height and
  publishes a height-filtered `LaserScan` on `/piracer/obstacle_scan` using the
  original image timestamp and TF into `base_link`. The navigation launch starts
  this node for both Nav2 costmaps, ACQUIRE and integrated FOLLOW.
- The full simulation uses `config/detector_ft01_test.yaml` and takes
  the FT01 weights from the `yolo_model_path` launch argument. See the
  [workspace instructions](../../README.md).
- YOLO is disabled by default in `piracer_sim_base.launch.py`. For standalone
  inference, set `yolo_enabled:=true`, `yolo_backend:=yolo11n_dnn` and
  `yolo_model_path:=/absolute/path/to/model.onnx`. To use FT01 in this profile,
  also pass `yolo_params_file` pointing to `config/detector_ft01_test.yaml`.

The detector expects float32 NCHW input `[1, 3, 640, 640]`. The generic COCO
model has output shape `[1, 84, 8400]` (four box channels and 80 class scores).
FT01 has shape `[1, 5, 8400]` (four box channels and one suitcase score,
class ID 0). Both layouts use the same decoder.

## Runtime compatibility

On this host, OpenCV 4.13.0 loads and executes the selected model with DNN CPU.
The system OpenCV 4.5.4 fails while importing its attention `Split` node, so
the installed system package alone is insufficient for this ONNX artifact.
The launch's YOLO11n branch uses the active `python3` environment; it must
resolve an OpenCV DNN build that supports this model. No CUDA provider,
Ultralytics runtime, or ONNX Runtime is used by the new node.

## Model and licensing

The ONNX files and training checkpoint are stored outside the repository.
Obtain the applicable model separately from the author. The perception
package source is MIT-licensed. Model weights are subject to separate terms.

Unit tests use synthetic tensors and in-memory messages. Real-model loading
and inference can be checked separately with the chosen OpenCV environment.
