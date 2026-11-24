# PiRacer Follow-Me

A ROS 2 simulation of a PiRacer that searches for, acquires and follows a
suitcase. Gazebo Classic supplies RGB-D data. RTAB-Map estimates motion and
builds an occupancy map, while Nav2 plans and executes navigation goals.
SEARCH uses frontier detection and ranking from
[m-explore-ros2](src/piracer_search/third_party/m-explore-ros2/README.md).

## Build

Use Ubuntu 22.04 with ROS 2 Humble and initialized `rosdep`.

```bash
git clone --branch main-sim https://github.com/Adiksoon/Follow-Me-Piracer.git
cd Follow-Me-Piracer
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y
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

See the package guides for [simulation](src/robot_simulation/README.md),
[navigation](src/piracer_navigation/README.md),
[perception](src/robot_perception/README.md) and the
[State Manager](src/piracer_state_manager/README.md).
