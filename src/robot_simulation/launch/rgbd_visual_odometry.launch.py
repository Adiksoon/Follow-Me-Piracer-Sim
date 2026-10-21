"""Launch RTAB-Map RGB-D visual odometry for the simulated PiRacer.

The main simulation launch includes this node whenever the camera is enabled.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    parameters_file = PathJoinSubstitution(
        [
            FindPackageShare("robot_simulation"),
            "config",
            "rtabmap_rgbd_odometry.yaml",
        ]
    )

    odometry = Node(
        package="rtabmap_odom",
        executable="rgbd_odometry",
        name="rgbd_odometry",
        namespace="visual_vo",
        output="screen",
        parameters=[
            parameters_file,
            {
                "publish_tf": ParameterValue(
                    LaunchConfiguration("publish_tf"), value_type=bool
                )
            },
        ],
        remappings=[
            (
                "/visual_vo/rgb/image",
                "/piracer/sim_camera/color/image_raw",
            ),
            (
                "/visual_vo/depth/image",
                "/piracer/sim_camera/depth/image_raw",
            ),
            (
                "/visual_vo/rgb/camera_info",
                "/piracer/sim_camera/color/camera_info",
            ),
            ("/visual_vo/odom", "/visual_odom"),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "publish_tf",
                default_value="false",
                description="Publish the visual odometry transform from this node.",
            ),
            odometry,
        ]
    )
