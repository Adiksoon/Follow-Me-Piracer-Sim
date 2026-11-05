"""Launch PiRacer simulation, perception, mapping and integrated FOLLOW/Nav2."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    navigation_share = FindPackageShare("piracer_navigation")
    perception_share = FindPackageShare("robot_perception")
    state_manager_share = FindPackageShare("piracer_state_manager")

    slam_launch = PathJoinSubstitution(
        [navigation_share, "launch", "slam_nav2.launch.py"]
    )
    search_state_manager_launch = PathJoinSubstitution(
        [state_manager_share, "launch", "search_state_manager.launch.py"]
    )
    detector_parameters = PathJoinSubstitution(
        [perception_share, "config", "detector_ft01_test.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("acquire_params_file", default_value=PathJoinSubstitution(
                [state_manager_share, "config", "acquire.yaml"])),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument(
                "start_paused",
                default_value="true",
                description=(
                    "Start Gazebo paused. Unpause it after checking the launched graph; "
                    "fresh detector messages activate the SEARCH/ACQUIRE/FOLLOW state machine."
                ),
            ),
            DeclareLaunchArgument("world", default_value=""),
            DeclareLaunchArgument("robot_spawn_x", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_y", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("raw_map_topic", default_value="/map"),
            DeclareLaunchArgument("map_topic", default_value="/nav2/map"),
            DeclareLaunchArgument("start_zone_enabled", default_value="true"),
            DeclareLaunchArgument("start_zone_radius", default_value="1.5"),
            DeclareLaunchArgument(
                "follow_max_speed_mps", default_value="0.05", choices=["0.05", "0.075"],
                description="Forward speed limit for manual FOLLOW in simulation.",
            ),
            DeclareLaunchArgument("nav2_autostart", default_value="true"),
            DeclareLaunchArgument("follow_nav2_avoidance", default_value="true"),
            DeclareLaunchArgument(
                "follow_target_distance_m", default_value="2.25",
                description="Desired FOLLOW distance in meters; must exceed 0.25.",
            ),
            DeclareLaunchArgument(
                "follow_publish_cmd_vel",
                default_value="false",
                description="Allow Follow to publish drive commands. Keep false for measurement-only runs.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_launch),
                launch_arguments={
                    "gui": LaunchConfiguration("gui"),
                    "start_paused": LaunchConfiguration("start_paused"),
                    "world": LaunchConfiguration("world"),
                    "robot_spawn_x": LaunchConfiguration("robot_spawn_x"),
                    "robot_spawn_y": LaunchConfiguration("robot_spawn_y"),
                    "robot_spawn_yaw": LaunchConfiguration("robot_spawn_yaw"),
                    "raw_map_topic": LaunchConfiguration("raw_map_topic"),
                    "map_topic": LaunchConfiguration("map_topic"),
                    "start_zone_enabled": LaunchConfiguration("start_zone_enabled"),
                    "start_zone_radius": LaunchConfiguration("start_zone_radius"),
                    "nav2_autostart": LaunchConfiguration("nav2_autostart"),
                    "follow_nav2_avoidance": LaunchConfiguration("follow_nav2_avoidance"),
                }.items(),
            ),
            Node(
                package="robot_perception",
                executable="yolo11n_node",
                name="yolo11n_node",
                parameters=[detector_parameters],
                output="screen",
            ),
            Node(
                package="robot_perception",
                executable="suitcase_point_node",
                name="suitcase_point_node",
                output="screen",
            ),
            Node(
                package="robot_perception",
                executable="obstacle_scan_node",
                name="obstacle_scan_node",
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(search_state_manager_launch),
                launch_arguments={
                    "map_topic": LaunchConfiguration("map_topic"),
                    "max_localization_age_sec": "1.0",
                    "tf_future_lead_sec": "0.2",
                    "use_sim_time": "true",
                    "acquire_params_file": LaunchConfiguration("acquire_params_file"),
                    "follow_publish_cmd_vel": LaunchConfiguration("follow_publish_cmd_vel"),
                    "follow_max_speed_mps": LaunchConfiguration("follow_max_speed_mps"),
                    "follow_target_distance_m": LaunchConfiguration("follow_target_distance_m"),
                    "follow_nav2_avoidance": LaunchConfiguration("follow_nav2_avoidance"),
                }.items(),
            ),
        ]
    )
