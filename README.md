# PiRacer Follow-Me

**An autonomous PiRacer that finds a suitcase and follows it through an indoor environment.**
This engineering thesis project combines RGB-D perception, visual odometry,
mapping and navigation on a small Ackermann-steered vehicle. This branch
contains the Gazebo Classic simulation.

| Gazebo classroom | Nav2 global costmap in RViz |
| :---: | :---: |
| ![Overhead view of the simulated classroom with four groups of tables and PiRacer near the center](docs/images/gazebo-classroom.png) | ![Global costmap showing areas of elevated cost around the same four groups of tables](docs/images/nav2-global-costmap.png) |

*Screenshots from the same paused SEARCH run. The costmap shows how Nav2
represents the classroom arrangement for planning.*

## How it works

- **Perception.** YOLO11n detects the suitcase in the RGB image. The depth
  image and camera calibration give its 3D position. A separate depth
  projection supplies obstacle observations.
- **Mapping and navigation.** RTAB-Map estimates motion and builds an
  occupancy map. Nav2 uses Smac Hybrid-A* and Regulated Pure Pursuit to
  plan and execute forward motion.
- **Robot behavior.** SEARCH explores boundaries between observed and unknown
  space, using frontier ranking from
  [m-explore-ros2](src/piracer_search/third_party/m-explore-ros2/README.md).
  ACQUIRE confirms or reacquires the target. FOLLOW regulates distance and
  heading directly and can use Nav2 to go around an obstacle. The State
  Manager coordinates transitions between these behaviors.

## Build

Use Ubuntu 22.04 with ROS 2 Humble and initialized `rosdep`.

```bash
git clone --branch main-sim https://github.com/Adiksoon/Follow-Me-Piracer.git
cd Follow-Me-Piracer
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build --symlink-install
source install/setup.bash
```

The detector needs OpenCV DNN with support for the exported YOLO11n model.
OpenCV 4.13.0 loaded this model in the project environment. Ubuntu 22.04's
OpenCV 4.5.4 failed to load an attention layer. Use an OpenCV build compatible
with the Python environment used by ROS 2 and `cv_bridge`.

## Run the full simulation

Obtain the fine-tuned single-class YOLO11n suitcase model (`best.onnx`) from
the project author. Git and the build do not download the weights. The default
COCO YOLO11n model uses a different class layout from the full simulation
profile. The FT01 model used here has this SHA-256 checksum:

```text
3556781d1c06c667a4b2f97f6fe4a028534302544946c8474bdabb74be7d5f82
```

```bash
ros2 launch piracer_state_manager piracer_sim.launch.py \
  yolo_model_path:="/absolute/path/to/best.onnx" \
  follow_publish_cmd_vel:=true
```

Gazebo starts paused. Press Play to start simulation time. SEARCH starts
after receiving fresh detector messages. The explicit `follow_publish_cmd_vel`
option enables drive commands from FOLLOW. SEARCH and ACQUIRE can still
issue Nav2 goals with that option disabled.

## Verification

At revision `ef9c37f`, all 10 packages built in a fresh local clone. The
offline checks passed 783 Python tests and the C++ frontier test suite.
The FT01 model also completed offline inference with OpenCV DNN.

## Package guides

[Simulation](src/robot_simulation/README.md),
[navigation](src/piracer_navigation/README.md),
[perception](src/robot_perception/README.md) and the
[State Manager](src/piracer_state_manager/README.md).
