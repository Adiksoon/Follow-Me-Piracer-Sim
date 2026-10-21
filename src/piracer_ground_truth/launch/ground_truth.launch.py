from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='piracer_ground_truth',
            executable='ground_truth_publisher',
            name='piracer_ground_truth_publisher',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
