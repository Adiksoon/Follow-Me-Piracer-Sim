"""Canonical entry point for the PiRacer simulation and mapping stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    navigation_share = get_package_share_directory("piracer_navigation")
    core_launch = os.path.join(navigation_share, "launch", "slam_nav2.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "nav2_enabled",
                default_value="false",
                description="Start Nav2 servers in addition to simulation and mapping.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(core_launch),
                launch_arguments={
                    "nav2_enabled": LaunchConfiguration("nav2_enabled"),
                }.items(),
            ),
        ]
    )
