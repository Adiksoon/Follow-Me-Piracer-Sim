"""Run the approved room with only the independent DATA-01 RGB-D camera."""

import hashlib
import os
import shutil
import tempfile
import copy
import xml.etree.ElementTree as ET

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


APPROVED_WORLD_SHA256 = "a2db8fa83e785b87629c93dd750ec76e4744f44113989ed68e1eee70cdefe188"
APPROVED_EMPTY_WORLD_SHA256 = "00738c78deca99ea6d770451a1664237ac335894c8297f91a9426084c526c321"
_runtime_world_directories = []


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_runtime_world(source_world, production_world, scene):
    expected_source_hash = {
        "production_room": APPROVED_WORLD_SHA256,
        "empty": APPROVED_EMPTY_WORLD_SHA256,
    }.get(scene)
    if expected_source_hash is None:
        raise RuntimeError(f"Unknown DATA-01 scene variant: {scene!r}")
    actual_hash = _sha256(source_world)
    if actual_hash != expected_source_hash:
        raise RuntimeError(f"Refusing dataset launch for unapproved {scene} world SHA-256 {actual_hash}")
    if _sha256(production_world) != APPROVED_WORLD_SHA256:
        raise RuntimeError("Refusing dataset launch: approved production target source changed")
    root = ET.parse(source_world).getroot()
    world = root.find("world")
    if world is None or world.find("plugin[@name='gazebo_ros_state']") is not None:
        raise RuntimeError("Selected world cannot be safely wrapped for dataset state services")
    if scene == "empty":
        production_root = ET.parse(production_world).getroot()
        target = production_root.find(".//world/model[@name='suitcase_target']")
        if target is None:
            raise RuntimeError("Approved production world has no suitcase_target model")
        target = copy.deepcopy(target)
        target_static = target.find("static")
        target_mesh = target.findtext(".//visual/geometry/mesh/uri", default="")
        if target_static is None or "hard_trolley_red/meshes/suitcase.dae" not in target_mesh:
            raise RuntimeError("Approved suitcase model is missing its static flag or expected mesh")
        target_static.text = "true"
        for plugin in target.findall("plugin"):
            target.remove(plugin)
        world.append(target)
    state_plugin = ET.fromstring("""
      <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
        <ros><namespace>/gazebo</namespace></ros>
        <update_rate>1.0</update_rate>
      </plugin>
    """)
    world.append(state_plugin)
    runtime_directory = tempfile.mkdtemp(prefix="data01-dataset-camera-")
    _runtime_world_directories.append(runtime_directory)
    runtime_world = os.path.join(runtime_directory, f"{scene}_dataset.world")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(runtime_world, encoding="utf-8", xml_declaration=True)
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
    empty_world = os.path.join(simulation_share, "worlds", "empty_sky.world")
    scene = LaunchConfiguration("scene").perform(context)
    source_world = {
        "production_room": production_world,
        "empty": empty_world,
    }.get(scene)
    if source_world is None:
        raise RuntimeError(f"Unknown DATA-01 scene variant: {scene!r}")
    runtime_world = _make_runtime_world(source_world, production_world, scene)
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
        DeclareLaunchArgument("scene", default_value="production_room"),
        OpaqueFunction(function=_build_dataset_launch),
    ])
