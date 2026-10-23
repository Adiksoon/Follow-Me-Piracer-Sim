from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("piracer_state_manager")
    return LaunchDescription([
        DeclareLaunchArgument(
            "search_enable_service",
            default_value="/search/enable",
            description="SetBool service used by the state manager to start or stop Search.",
        ),
        Node(
            package="piracer_state_manager",
            executable="state_manager_node",
            name="piracer_state_manager",
            output="screen",
            parameters=[
                package_share + "/config/state_manager.yaml",
                {"search_enable_service": LaunchConfiguration("search_enable_service")},
            ],
        ),
        Node(
            package="piracer_state_manager",
            executable="acquire_node",
            name="piracer_acquire",
            output="screen",
        ),
        Node(
            package="piracer_state_manager",
            executable="follow_node",
            name="piracer_follow",
            output="screen",
        ),
    ])
