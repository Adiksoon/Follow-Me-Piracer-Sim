"""Compose simulation, Search, Follow, and the single-owner Nav2 relay."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    coordinator_share = get_package_share_directory("piracer_navigation_coordinator")
    state_manager_share = get_package_share_directory("piracer_state_manager")
    navigation_share = get_package_share_directory("piracer_navigation")
    search_endpoint = "/navigation_coordinator/search"
    follow_endpoint = "/navigation_coordinator/follow"

    search_state_manager = GroupAction(
        scoped=True,
        actions=[
            SetParameter(
                name="navigate_to_pose_action",
                value=search_endpoint,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        state_manager_share,
                        "launch",
                        "search_state_manager.launch.py",
                    )
                )
            ),
        ],
    )
    follow_pipeline = GroupAction(
        scoped=True,
        actions=[
            SetParameter(
                name="navigate_to_pose_action",
                value=follow_endpoint,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(navigation_share, "launch", "follow.launch.py")
                ),
                launch_arguments={
                    "gui": LaunchConfiguration("gui"),
                    "world": LaunchConfiguration("world"),
                    "robot_spawn_x": LaunchConfiguration("robot_spawn_x"),
                    "robot_spawn_y": LaunchConfiguration("robot_spawn_y"),
                    "robot_spawn_yaw": LaunchConfiguration("robot_spawn_yaw"),
                    "enable_navigation": LaunchConfiguration("enable_follow_navigation"),
                }.items(),
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("world", default_value=""),
        DeclareLaunchArgument("robot_spawn_x", default_value="0.0"),
        DeclareLaunchArgument("robot_spawn_y", default_value="0.0"),
        DeclareLaunchArgument("robot_spawn_yaw", default_value="0.0"),
        DeclareLaunchArgument(
            "enable_follow_navigation",
            default_value="false",
            description="Forward the existing Follow navigation switch; defaults off.",
        ),
        Node(
            package="piracer_navigation_coordinator",
            executable="navigation_coordinator_node",
            name="piracer_navigation_coordinator",
            output="screen",
        ),
        search_state_manager,
        follow_pipeline,
    ])
