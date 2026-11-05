"""The A1 motion gate stays closed by default through the launch chain."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


SOURCE = Path(__file__).parents[1]


def _load_launch(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_argument(description, name):
    argument = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == name
    )
    return perform_substitutions(LaunchContext(), argument.default_value)


def test_state_manager_passes_closed_motion_gate_as_boolean_parameter():
    with patch(
        "ament_index_python.packages.get_package_share_directory",
        side_effect=lambda package: f"/share/{package}",
    ):
        launch = _load_launch(
            "follow_state_manager_launch_under_test",
            SOURCE / "launch/state_manager.launch.py",
        )
    description = launch.generate_launch_description()
    assert _default_argument(description, "follow_publish_cmd_vel") == "false"
    assert _default_argument(description, "follow_max_speed_mps") == "0.05"
    assert _default_argument(description, "follow_target_distance_m") == "2.25"

    follow = next(
        entity
        for entity in description.entities
        if isinstance(entity, Node)
        and entity._Node__node_name == "piracer_follow"
    )
    parameters = follow._Node__parameters[0]
    key = next(
        key
        for key in parameters
        if perform_substitutions(LaunchContext(), key) == "publish_cmd_vel"
    )
    speed_key = next(k for k in parameters if perform_substitutions(LaunchContext(), k) == "max_speed_mps")
    context = LaunchContext()
    context.launch_configurations["follow_max_speed_mps"] = "0.075"
    assert parameters[speed_key].evaluate(context) == 0.075
    target_key = next(k for k in parameters
                      if perform_substitutions(context, k) == "target_distance_m")
    context.launch_configurations["follow_target_distance_m"] = "1.0"
    assert parameters[target_key].value_type is float
    assert parameters[target_key].evaluate(context) == 1.0
    parameter = parameters[key]
    assert isinstance(parameter, ParameterValue)
    assert parameter.value_type is bool
    assert any(
        isinstance(value, LaunchConfiguration)
        and perform_substitutions(LaunchContext(), value.variable_name)
        == "follow_publish_cmd_vel"
        for value in parameter.value
    )


def test_full_simulation_forwards_motion_gate_without_enabling_it():
    launch = _load_launch(
        "follow_piracer_sim_launch_under_test",
        SOURCE / "launch/piracer_sim.launch.py",
    )
    description = launch.generate_launch_description()
    assert _default_argument(description, "follow_publish_cmd_vel") == "false"
    assert _default_argument(description, "follow_max_speed_mps") == "0.05"
    assert _default_argument(description, "follow_target_distance_m") == "2.25"
    search_state_manager = next(
        entity
        for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
        and "follow_publish_cmd_vel" in dict(entity.launch_arguments)
    )
    forwarded = dict(search_state_manager.launch_arguments)["follow_publish_cmd_vel"]
    context = LaunchContext()
    context.launch_configurations["follow_target_distance_m"] = "1.0"
    assert dict(search_state_manager.launch_arguments)["follow_target_distance_m"].perform(context) == "1.0"
    assert isinstance(forwarded, LaunchConfiguration)
    assert perform_substitutions(
        LaunchContext(), forwarded.variable_name
    ) == "follow_publish_cmd_vel"


def test_search_launch_forwards_motion_gate_to_state_manager():
    with patch(
        "ament_index_python.packages.get_package_share_directory",
        side_effect=lambda package: f"/share/{package}",
    ):
        launch = _load_launch(
            "follow_search_state_manager_launch_under_test",
            SOURCE / "launch/search_state_manager.launch.py",
        )
    description = launch.generate_launch_description()
    assert _default_argument(description, "follow_publish_cmd_vel") == "false"
    assert _default_argument(description, "follow_max_speed_mps") == "0.05"
    assert _default_argument(description, "follow_target_distance_m") == "2.25"
    manager = next(
        entity
        for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
        and "follow_publish_cmd_vel" in dict(entity.launch_arguments)
    )
    forwarded = dict(manager.launch_arguments)["follow_publish_cmd_vel"]
    context = LaunchContext()
    context.launch_configurations["follow_target_distance_m"] = "1.0"
    assert dict(manager.launch_arguments)["follow_target_distance_m"].perform(context) == "1.0"
    assert isinstance(forwarded, LaunchConfiguration)
    assert perform_substitutions(
        LaunchContext(), forwarded.variable_name
    ) == "follow_publish_cmd_vel"


def test_nav2_autostart_defaults_true_and_can_be_forwarded_false():
    sim = _load_launch("autostart_sim", SOURCE / "launch/piracer_sim.launch.py")
    description = sim.generate_launch_description()
    assert _default_argument(description, "nav2_autostart") == "true"
    include = next(e for e in description.entities if isinstance(e, IncludeLaunchDescription)
                   and "nav2_autostart" in dict(e.launch_arguments))
    ctx = LaunchContext()
    ctx.launch_configurations.update(nav2_autostart="false", follow_nav2_avoidance="false")
    assert dict(include.launch_arguments)["nav2_autostart"].perform(ctx) == "false"
    navigation = _load_launch("autostart_nav", SOURCE.parents[0] / "piracer_navigation/launch/slam_nav2.launch.py")
    assert _default_argument(navigation.generate_launch_description(), "nav2_autostart") == "true"
    with patch.object(navigation, "get_package_share_directory", side_effect=lambda name: "/share/" + name):
        child = navigation._build_launch(ctx)[-1]
    assert dict(child.launch_arguments)["autostart"].perform(ctx) == "false"
