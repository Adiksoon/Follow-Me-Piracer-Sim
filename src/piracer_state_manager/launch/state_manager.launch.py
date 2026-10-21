from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("piracer_state_manager")
    return LaunchDescription([
        Node(
            package="piracer_state_manager",
            executable="state_manager_node",
            name="piracer_state_manager",
            output="screen",
            parameters=[package_share + "/config/state_manager.yaml"],
        ),
    ])
