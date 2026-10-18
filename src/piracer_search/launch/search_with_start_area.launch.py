"""Explicit simulation profile for Search with a fixed startup-area assumption."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    navigation_share = get_package_share_directory("piracer_navigation")
    simulation_share = get_package_share_directory("robot_simulation")
    search_share = get_package_share_directory("piracer_search")
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            navigation_share + "/launch/slam_nav2.launch.py"
        ),
        launch_arguments={
            "gui": "false",
            "world": simulation_share + "/worlds/sala_laboratoryjna_production.world",
            "robot_spawn_x": "0.0",
            "robot_spawn_y": "0.0",
            "robot_spawn_yaw": "0.0",
            "map_topic": "/search_map",
        }.items(),
    )
    start_area = Node(
        package="piracer_search",
        executable="start_area_node",
        name="piracer_search_start_area",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    search = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(search_share + "/launch/search.launch.py"),
        launch_arguments={
            "map_topic": "/search_map",
            "use_sim_time": "true",
        }.items(),
    )
    return LaunchDescription([navigation, start_area, search])
