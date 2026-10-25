# PiRacer Gazebo drive integration

## Current model configuration

The robot description now loads the ROS Humble `gazebo_ros_ackermann_drive`
plugin from `gazebo_plugins` 3.9.0. It owns both front steering joints and both
rear drive joints, so each wheel pair has one actuator owner.

The adapter subscribes directly to `/cmd_vel`, which keeps the usual `Twist`
meaning: `linear.x` is speed and `angular.z` is yaw rate. `ackermann_twist_adapter` converts it to
the steering-angle command expected by the upstream plugin and publishes on
`/piracer/ackermann_cmd_vel`. At effectively zero speed it sends zero steering,
because Ackermann steering cannot turn the vehicle in place. Reverse-speed sign
handling is left to the upstream plugin.

The upstream Gazebo Ackermann plugin has `max_speed` set to 0.20 m/s. The
current Navigation2 controller requests 0.05 m/s and the velocity smoother
caps forward speed at 0.05 m/s. These are separate layer settings; the Nav2
command path is limited to 0.05 m/s. There is no command gate on the direct
topic, so this does not cap manual commands. The center steering limit is 0.554 rad,
corresponding to the current 0.25 m provisional minimum turn radius and 0.155 m
wheelbase. The three P-only gains are initial estimates from the current motion
and joint settings;
they have not been measured or runtime-validated. The upstream plugin has no
steering-rate limiter and no configurable PID output-torque cap. It also sets
the Gazebo world's friction model to `cone_model` when loaded, which affects the
whole world.

## Source

The implementation uses the installed ROS Humble `gazebo_plugins` package; no
upstream plugin source was copied. The inspected upstream source is pinned to
[`gazebo_ros_pkgs` commit `ae6e5793a31f5927df89cb58007d2a5eb567baa8](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/ae6e5793a31f5927df89cb58007d2a5eb567baa8/gazebo_plugins/src/gazebo_ros_ackermann_drive.cpp).
The matching Humble release tag is
[`release/humble/gazebo_plugins/3.9.0-1`](https://github.com/ros2-gbp/gazebo_ros_pkgs-release/blob/79bc1dc4cc90e0edba2251d8f9dd1d6089281e97/package.xml).
