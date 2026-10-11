"""Run a read-only telemetry recorder alongside the simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("output_file", default_value=""),
        DeclareLaunchArgument("sample_period_sec", default_value="1.0"),
        Node(
            package="simulation_telemetry",
            executable="telemetry_recorder",
            name="simulation_telemetry",
            output="screen",
            parameters=[{
                "output_file": ParameterValue(
                    LaunchConfiguration("output_file"), value_type=str
                ),
                "sample_period_sec": ParameterValue(
                    LaunchConfiguration("sample_period_sec"), value_type=float
                ),
            }],
        ),
    ])
