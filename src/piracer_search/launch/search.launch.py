from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_horizontal_fov_deg",
            default_value="65.36016047954865",
            description="Simulation URDF HFOV converted from 1.14075 rad.",
        ),
        DeclareLaunchArgument(
            "camera_max_range_m",
            default_value="5.0",
            description="Simulation RTAB-Map Grid/RangeMax scoring bound; not physical depth qualification.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "map_topic",
            default_value="/map",
            description="OccupancyGrid topic used by Search.",
        ),
        DeclareLaunchArgument(
            "behavior_tree",
            default_value=get_package_share_directory("piracer_search") + "/behavior_trees/search_navigate_to_pose.xml",
        ),
        Node(
            package="piracer_search",
            executable="search_node",
            name="piracer_search",
            output="screen",
            parameters=[{
                "camera.horizontal_fov_deg": LaunchConfiguration("camera_horizontal_fov_deg"),
                "camera.max_range_m": LaunchConfiguration("camera_max_range_m"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "behavior_tree": LaunchConfiguration("behavior_tree"),
                "map_topic": LaunchConfiguration("map_topic"),
            }],
        ),
    ])
