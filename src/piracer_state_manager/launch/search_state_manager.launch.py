from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("piracer_state_manager")
    search_share = get_package_share_directory("piracer_search")
    state_manager_launch = package_share + "/launch/state_manager.launch.py"
    search_service = LaunchConfiguration("search_enable_service")
    return LaunchDescription([
        DeclareLaunchArgument(
            "search_enable_service",
            default_value="/search/enable",
            description="SetBool service used by the state manager to start or stop Search.",
        ),
        DeclareLaunchArgument(
            "map_topic",
            default_value="/map",
            description="OccupancyGrid topic consumed by Search.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("acquire_params_file", default_value=package_share + "/config/acquire.yaml"),
        DeclareLaunchArgument(
            "follow_publish_cmd_vel",
            default_value="false",
            description="Allow Follow to publish drive commands. Keep false for measurement-only runs.",
        ),
        DeclareLaunchArgument(
            "follow_max_speed_mps", default_value="0.05", choices=["0.05", "0.075"],
            description="Forward speed limit for manual FOLLOW in simulation.",
        ),
        DeclareLaunchArgument("follow_nav2_avoidance", default_value="false"),
        DeclareLaunchArgument(
            "follow_target_distance_m", default_value="2.25",
            description="Desired FOLLOW distance in meters; must exceed 0.25.",
        ),
        DeclareLaunchArgument("max_localization_age_sec", default_value="0.5"),
        DeclareLaunchArgument("tf_future_lead_sec", default_value="0.1"),
        Node(
            package="piracer_search",
            executable="search_node",
            name="piracer_search",
            output="screen",
            parameters=[{
                "start_stop_service": search_service,
                "map_topic": LaunchConfiguration("map_topic"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "max_localization_age_sec": ParameterValue(
                    LaunchConfiguration("max_localization_age_sec"),
                    value_type=float,
                ),
                "tf_future_lead_sec": ParameterValue(
                    LaunchConfiguration("tf_future_lead_sec"),
                    value_type=float,
                ),
                "behavior_tree": search_share
                + "/behavior_trees/search_navigate_to_pose.xml",
            }],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(state_manager_launch),
            launch_arguments={
                "search_enable_service": search_service,
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "acquire_params_file": LaunchConfiguration("acquire_params_file"),
                "follow_use_sim_time": LaunchConfiguration("use_sim_time"),
                "follow_publish_cmd_vel": LaunchConfiguration("follow_publish_cmd_vel"),
                "follow_max_speed_mps": LaunchConfiguration("follow_max_speed_mps"),
                "follow_target_distance_m": LaunchConfiguration("follow_target_distance_m"),
                "follow_nav2_avoidance": LaunchConfiguration("follow_nav2_avoidance"),
            }.items(),
        ),
    ])
