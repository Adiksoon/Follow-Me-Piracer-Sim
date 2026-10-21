from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("piracer_state_manager")
    search_share = get_package_share_directory("piracer_search")
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
        DeclareLaunchArgument(
            "camera_horizontal_fov_deg",
            default_value="65.36016047954865",
        ),
        DeclareLaunchArgument("camera_max_range_m", default_value="5.0"),
        Node(
            package="piracer_search",
            executable="search_node",
            name="piracer_search",
            output="screen",
            parameters=[{
                "start_stop_service": search_service,
                "map_topic": LaunchConfiguration("map_topic"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "camera.horizontal_fov_deg": LaunchConfiguration("camera_horizontal_fov_deg"),
                "camera.max_range_m": LaunchConfiguration("camera_max_range_m"),
                "behavior_tree": search_share
                + "/behavior_trees/search_navigate_to_pose.xml",
            }],
        ),
        Node(
            package="piracer_state_manager",
            executable="state_manager_node",
            name="piracer_state_manager",
            output="screen",
            parameters=[
                package_share + "/config/state_manager.yaml",
                {"search_enable_service": search_service},
            ],
        ),
    ])
