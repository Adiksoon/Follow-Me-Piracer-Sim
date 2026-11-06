# PiRacer RGB-D mapping and navigation

Source ROS 2 Humble and the workspace, then start mapping and Nav2.

```bash
ros2 launch piracer_navigation slam_nav2.launch.py
```

For SEARCH, ACQUIRE and FOLLOW, use the full simulation launch from the
[State Manager guide](../piracer_state_manager/README.md).

## Mapping and transforms

RTAB-Map consumes the RGB image, registered depth image and CameraInfo under
`/sim_camera`, together with `/visual_odom`. It publishes `/map`,
`/cloud_map` and `map -> vo_odom`. The launch starts one RGB-D visual odometry
node with TF enabled. It publishes `vo_odom -> base_link`. The robot model
provides the camera transforms.

The camera publishes 640 × 360 images at 30 Hz. Its optical frame is
`sim_camera_optical_frame`. The camera link sits at x = 0.236 m and z = 0.320 m
relative to the chassis, with zero pitch. Field of view and occlusion limit
coverage. Nav2 uses a depth scan for live obstacle observations. The full
workflow also provides `/piracer/obstacle_scan` from the RGB-D obstacle scan node.

## Map and motion

Nav2 reads `/nav2/map`. The map processor waits for simulation time, a source
map in frame `map` and a `map -> base_link` transform at most 0.20 s old.
It records the starting position once and marks unknown cells within 1.5 m
of it as free. Free and occupied cells retain their values. The circle
stays at the starting position for that launch. Set `start_zone_enabled:=false`
to pass through `/map`, or set `start_zone_radius` to change the radius.
The start zone is an engineering assumption about the initial area.

The Nav2 configuration uses Smac Hybrid and Regulated Pure Pursuit with a
0.50 m minimum planning radius and a 0.10 m/s desired linear speed. The
standard planner excludes unknown cells. The FOLLOW planner can include
them while live obstacle and collision checks apply. The configured paths
use forward motion without rotation in place.

Standalone navigation sends smoothed commands on `/cmd_vel`. Keep one
command source active in this profile. In the full workflow, the State
Manager confirms the command owner. The adapter uses `/cmd_vel` for SEARCH
and ACQUIRE, and `/cmd_vel_follow` for FOLLOW. It sends steering and drive
commands to Gazebo on `/piracer/ackermann_cmd_vel`.

## Evidence and provenance

The CP-MAP-MANUAL-01 review accepted the appearance of the 2D map and 3D
cloud. The stationary D1 trial was inconclusive, and RUN09 missed the
50 mm p95 XY accuracy criterion. VO accuracy and hardware readiness remain
unqualified by these runs.

The complete D0/D1/R9 notes, old TF setup and report paths are preserved at
the reviewed baseline.

```bash
git show 4fe02eb04b43ea93edb3e44d7d63a8adb7d3dc5f:src/piracer_navigation/README.md
```

This launch uses the upstream RTAB-Map wrapper and its mapping executable
directly. The historical integration used RTAB-Map Humble 0.23.7,
`depthimage_to_laserscan` 2.5.1 and Nav2 1.1.20.
[RTAB-Map wrapper source](https://github.com/introlab/rtabmap_ros/tree/c25a091c2a682f0e0f4e8d19d807953e7f1a8161)
