Source: ros-simulation/gazebo_ros_pkgs, tag 3.9.0, gazebo_plugins/src/gazebo_ros_ackermann_drive.cpp.
https://github.com/ros-simulation/gazebo_ros_pkgs/blob/3.9.0/gazebo_plugins/src/gazebo_ros_ackermann_drive.cpp

Copyright 2019 Open Source Robotics Foundation, Inc. Apache-2.0.
Upstream source SHA-256: a3137a92c932017f94c0c587dc1d10e88684c5676325c52fc6ed885874060ea2

PiRacer modifications: independent P braking of each rear wheel only at exactly zero target speed, using the existing P gain and each joint effort limit; one startup identification log. Positive drive, steering, ROS interfaces and limits retain the upstream implementation. Built with a distinct library filename; does not replace the system library.
