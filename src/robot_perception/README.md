# `robot_perception` — CP-00D offline skeleton

This package is an independent implementation in the canonical `symulacja/`
workspace. It does not copy or adapt code from `ros2_ws/`.

## Current boundary

- `detection_pipeline.py` handles explicit RGB/BGR input order, square
  letterboxing, normalized NCHW input preparation, inverse box mapping,
  synthetic YOLO11-shaped output decoding, confidence/class filtering,
  classwise NMS, and RGB diagnostic overlays.
- `ros_messages.py` converts `sensor_msgs/Image` (`rgb8` or `bgr8`) through
  `cv_bridge`, maps source-image pixel boxes to `Detection2DArray`, and provides
  a helper to publish detections plus a same-header RGB diagnostic image.
- `scripts/yolo11n_node` declares parameters expected by the canonical launch,
  then exits with an explicit licensing-gate error. It does not open a model
  path, subscribe to images, or publish detections/debug images. During its
  brief initialization, standard `rclpy.Node` administration and rosout
  entities are created.
- `config/detector.yaml` is a replaceable model/backend parameter profile; its
  values do not activate inference.
- No ONNX runtime/backend loader, model weights, Ultralytics package, or
  Ultralytics-derived source is included.

The decoder's synthetic contract is `[1, 84, N]`: four `cx, cy, width, height`
channels and 80 class-score channels, with no objectness. This comes from the
local export record and is **not confirmed against the model in this checkpoint**.
Synthetic unit tests establish only the behavior of these pure functions and
ROS message conversion. They do not establish model compatibility, inference
quality, live ROS topic transport, or runtime performance.

## Parameters reserved for the detector interface

The names match `piracer_sim.launch.py`: `image_topic`, `model_path`,
`confidence`, `nms_iou`, `imgsz`, `class_filter`, `opencv_num_threads`, and
`torch_num_threads`. `backend` is kept separate so a future approved model
adapter can replace the YOLO11-specific path without changing the output
message contract. The current node intentionally does not consume these values.

## Model origin and licensing boundary

No model file is bundled, copied, opened, or tested by this package. The local
Research Pack `research/packs/RQ-YOLO-SUITCASE-CHALLENGER-20261008-R3.md`
records the publisher's statement that YOLO weights are AGPL-3.0 by default,
that its free path requires the whole project to use AGPL-3.0, and that an
Enterprise path is offered. The pack does not establish a separate exemption
for this project's local academic use and is not legal advice. Therefore model
integration remains disabled pending a resolved author licensing decision.

The newly written package source is MIT-licensed in its own `LICENSE` file;
this does not change the license of the whole repository. The installed
dependencies are system packages, not vendored copies: ROS 2 `rclpy`,
`sensor_msgs`, `vision_msgs`, `cv_bridge`, Debian NumPy, and Debian OpenCV.
Their local package metadata identifies Apache-2.0/BSD licenses for the ROS
components and package/subcomponent licensing for OpenCV and NumPy; see the
installed package XML and Debian copyright files recorded in the CP-00D report.

## Local checks

Tests use only generated arrays and in-memory ROS messages. Do not present them
as an inference or simulator result.
