"""Run RTAB-Map mapping against an already running RGB-D camera and VO."""

from datetime import datetime, timezone
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("piracer_navigation")
    rtabmap_config = os.path.join(package_share, "config", "rtabmap.yaml")
    database_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    database_path = os.path.join(
        os.path.expanduser("~"),
        ".ros",
        "piracer_mapping_test_" + database_stamp + ".db",
    )

    odom_bridge = Node(
        package="piracer_navigation",
        executable="odom_tf_bridge",
        name="piracer_odom_tf_bridge",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "diagnostics_enabled": ParameterValue(
                LaunchConfiguration("bridge_diagnostics"), value_type=bool
            ),
        }],
    )

    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            rtabmap_config,
            {"use_sim_time": True, "database_path": database_path},
        ],
        remappings=[
            ("rgb/image", "/piracer/sim_camera/color/image_raw"),
            ("depth/image", "/piracer/sim_camera/depth/image_raw"),
            ("rgb/camera_info", "/piracer/sim_camera/color/camera_info"),
            ("odom", "/visual_odom"),
            ("map", "/map"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "bridge_diagnostics",
            default_value="false",
            description="Enable periodic odom_tf_bridge diagnostics",
        ),
        odom_bridge,
        rtabmap,
    ])
