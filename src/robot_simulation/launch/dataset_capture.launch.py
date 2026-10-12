"""Run the approved room with only the independent DATA-01 RGB-D camera."""

import hashlib
import os
import shutil
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


APPROVED_WORLD_SHA256 = "71111ed9cfd933117c3ac7013bed41c85d342eb5cc6b8e4e0eddbeeb5ae39c7e"
_runtime_world_directories = []


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_runtime_world(production_world):
    actual_hash = _sha256(production_world)
    if actual_hash != APPROVED_WORLD_SHA256:
        raise RuntimeError(
            f"Refusing dataset launch for unapproved world SHA-256 {actual_hash}"
        )
    with open(production_world, encoding="utf-8") as stream:
        source = stream.read()
    if source.count("</world>") != 1 or "libgazebo_ros_state.so" in source:
        raise RuntimeError("Production world cannot be safely wrapped for dataset state services")
    state_plugin = """
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>1.0</update_rate>
    </plugin>
"""
    runtime_directory = tempfile.mkdtemp(prefix="data01-dataset-camera-")
    _runtime_world_directories.append(runtime_directory)
    runtime_world = os.path.join(runtime_directory, "sala_laboratoryjna_dataset.world")
    with open(runtime_world, "w", encoding="utf-8") as stream:
        stream.write(source.replace("</world>", state_plugin + "</world>"))
    return runtime_world


def _cleanup_runtime_world(context):
    for directory in _runtime_world_directories:
        shutil.rmtree(directory, ignore_errors=True)
    _runtime_world_directories.clear()
    return []


def _build_dataset_launch(context):
    simulation_share = get_package_share_directory("robot_simulation")
    assets_share = get_package_share_directory("simulation_assets")
    gazebo_share = get_package_share_directory("gazebo_ros")
    production_world = os.path.join(
        simulation_share, "worlds", "sala_laboratoryjna_production.world"
    )
    runtime_world = _make_runtime_world(production_world)
    camera_model = os.path.join(
        simulation_share, "models", "dataset_camera", "model.sdf"
    )
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": runtime_world, "factory": "true"}.items(),
    )
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "gzclient.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("gui")),
    )
    spawn_camera = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "dataset_camera",
            "-file", camera_model,
            "-x", "0.0", "-y", "0.0", "-z", "1.2",
            "-Y", "0.0", "-P", "0.25",
        ],
        output="screen",
    )
    return [
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            [
                os.path.join(simulation_share, "models"), ":",
                os.path.join(assets_share, "assets"), ":",
                EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
            ],
        ),
        SetEnvironmentVariable(
            "GAZEBO_PLUGIN_PATH",
            ["/opt/ros/humble/lib:", EnvironmentVariable("GAZEBO_PLUGIN_PATH", default_value="")],
        ),
        gazebo_server,
        gazebo_client,
        TimerAction(period=3.0, actions=[spawn_camera]),
        RegisterEventHandler(
            OnShutdown(on_shutdown=[OpaqueFunction(function=_cleanup_runtime_world)])
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        OpaqueFunction(function=_build_dataset_launch),
    ])
