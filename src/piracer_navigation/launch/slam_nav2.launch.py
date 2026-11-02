"""Offline-prepared PiRacer RGB-D, RTAB-Map, and Nav2 integration launch."""

from datetime import datetime, timezone
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def _build_launch(context, *args, **kwargs):
    package_share = get_package_share_directory("piracer_navigation")
    simulation_share = get_package_share_directory("robot_simulation")
    nav2_share = get_package_share_directory("nav2_bringup")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(simulation_share, "launch", "piracer_sim_base.launch.py")
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "start_paused": LaunchConfiguration("start_paused"),
            "world": LaunchConfiguration("world"),
            "robot_spawn_x": LaunchConfiguration("robot_spawn_x"),
            "robot_spawn_y": LaunchConfiguration("robot_spawn_y"),
            "robot_spawn_yaw": LaunchConfiguration("robot_spawn_yaw"),
            "follow_nav2_avoidance": LaunchConfiguration("follow_nav2_avoidance"),
            "camera_enabled": "true",
            "camera_update_rate": "30.0",
            "camera_viewer_enabled": "false",
            "suitcase_control_gui": "false",
            "yolo_enabled": "false",
            "visual_odometry_enabled": "false",
        }.items(),
    )

    native_visual_odometry = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(simulation_share, "launch", "rgbd_visual_odometry.launch.py")
        ),
        launch_arguments={"publish_tf": "true"}.items(),
    )

    scan_converter = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "scan_time": 0.033,
                "range_min": 0.05,
                "range_max": 3.0,
                "scan_height": 1,
                # The converter requires a laser/body frame, not the optical frame.
                "output_frame": "sim_camera_link",
            }
        ],
        remappings=[
            ("depth", "/sim_camera/depth/image_raw"),
            ("depth_camera_info", "/sim_camera/depth/camera_info"),
            ("scan", "/piracer/scan"),
        ],
    )

    rtabmap_config = os.path.join(package_share, "config", "rtabmap.yaml")
    database_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    database_path = os.path.join(
        os.path.expanduser("~"),
        ".ros",
        "piracer_navigation_" + database_stamp + ".db",
    )
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[rtabmap_config, {"database_path": database_path}],
        remappings=[
            ("rgb/image", "/sim_camera/color/image_raw"),
            ("depth/image", "/sim_camera/depth/image_raw"),
            ("rgb/camera_info", "/sim_camera/color/camera_info"),
            ("odom", "/visual_odom"),
            ("map", "/map"),
        ],
    )

    start_zone_map = ExecuteProcess(
        cmd=[
            sys.executable,
            "-m",
            "piracer_navigation.start_zone_map",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            PythonExpression(
                ["'enabled:=' + '", LaunchConfiguration("start_zone_enabled"), "'"]
            ),
            "-p",
            PythonExpression(
                ["'radius:=' + '", LaunchConfiguration("start_zone_radius"), "'"]
            ),
            "-p",
            PythonExpression(
                ["'input_topic:=' + '", LaunchConfiguration("raw_map_topic"), "'"]
            ),
            "-p",
            PythonExpression(
                ["'output_topic:=' + '", LaunchConfiguration("map_topic"), "'"]
            ),
            "-p",
            "map_frame:=map",
            "-p",
            "base_frame:=base_link",
            "-p",
            "max_tf_age:=0.20",
        ],
        output="screen",
    )

    nav2_config = os.path.join(package_share, "config", "nav2_params.yaml")
    bt_path = os.path.join(package_share, "behavior_trees", "navigate_to_pose.xml")
    bt_through_poses_path = os.path.join(
        package_share,
        "behavior_trees",
        "navigate_through_poses_no_recovery.xml",
    )
    nav2_params = RewrittenYaml(
        source_file=nav2_config,
        root_key="",
        param_rewrites={
            "default_nav_to_pose_bt_xml": bt_path,
            "default_nav_through_poses_bt_xml": bt_through_poses_path,
            "map_topic": LaunchConfiguration("map_topic"),
        },
        convert_types=True,
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_share, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": nav2_params,
            "autostart": "true",
            "use_composition": "False",
            "use_respawn": "False",
        }.items(),
    )

    return [
        simulation,
        native_visual_odometry,
        scan_converter,
        rtabmap,
        start_zone_map,
        nav2,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("start_paused", default_value="false"),
            DeclareLaunchArgument("world", default_value=""),
            DeclareLaunchArgument("robot_spawn_x", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_y", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("follow_nav2_avoidance", default_value="false"),
            DeclareLaunchArgument("raw_map_topic", default_value="/map"),
            DeclareLaunchArgument("map_topic", default_value="/nav2/map"),
            DeclareLaunchArgument("start_zone_enabled", default_value="true"),
            DeclareLaunchArgument("start_zone_radius", default_value="1.5"),
            OpaqueFunction(function=_build_launch),
        ]
    )
