"""Offline-prepared PiRacer RGB-D, RTAB-Map, and Nav2 integration launch."""

from datetime import datetime, timezone
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def _build_launch(context, *args, **kwargs):
    package_share = get_package_share_directory("piracer_navigation")
    simulation_share = get_package_share_directory("robot_simulation")
    nav2_share = get_package_share_directory("nav2_bringup")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(simulation_share, "launch", "piracer_sim.launch.py")
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "world": LaunchConfiguration("world"),
            "robot_spawn_x": LaunchConfiguration("robot_spawn_x"),
            "robot_spawn_y": LaunchConfiguration("robot_spawn_y"),
            "robot_spawn_yaw": LaunchConfiguration("robot_spawn_yaw"),
            "camera_enabled": "true",
            "camera_update_rate": "30.0",
            "camera_viewer_enabled": "false",
            "suitcase_control_gui": "false",
            "yolo_enabled": "false",
        }.items(),
    )

    odom_bridge = Node(
        package="piracer_navigation",
        executable="odom_tf_bridge",
        name="piracer_odom_tf_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
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
            ("depth", "/piracer/sim_camera/depth/image_raw"),
            ("depth_camera_info", "/piracer/sim_camera/depth/camera_info"),
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
            ("rgb/image", "/piracer/sim_camera/color/image_raw"),
            ("depth/image", "/piracer/sim_camera/depth/image_raw"),
            ("rgb/camera_info", "/piracer/sim_camera/color/camera_info"),
            ("odom", "/visual_odom"),
            ("map", "/map"),
        ],
    )

    nav2_config = os.path.join(package_share, "config", "nav2_params.yaml")
    bt_path = os.path.join(package_share, "behavior_trees", "navigate_to_pose.xml")
    bt_through_poses_path = os.path.join(
        package_share, "behavior_trees", "navigate_through_poses_no_recovery.xml"
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

    command_gate = Node(
        package="piracer_navigation",
        executable="command_gate",
        name="piracer_command_gate",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "nav_cmd_topic": "/cmd_vel",
                "odom_valid_topic": "/piracer_navigation/odom_valid",
                "scan_topic": "/piracer/scan",
                "enable_topic": "/piracer_navigation/enable_motion",
                "stop_topic": "/piracer_navigation/stop",
                "output_topic": "/piracer/cmd_vel",
            }
        ],
    )

    stop_launch_on_gate_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=command_gate,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="command gate exited; stop the owned session")
                )
            ],
        )
    )

    return [
        stop_launch_on_gate_exit,
        simulation,
        odom_bridge,
        scan_converter,
        rtabmap,
        nav2,
        command_gate,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("world", default_value=""),
            DeclareLaunchArgument("robot_spawn_x", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_y", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("map_topic", default_value="/map"),
            OpaqueFunction(function=_build_launch),
        ]
    )
