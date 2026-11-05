"""ACQUIRE configuration reaches each launch entry without development overrides."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SOURCE = Path(__file__).parents[1]


def launch_description(filename):
    with patch("ament_index_python.packages.get_package_share_directory",
               side_effect=lambda package: "/share/" + package):
        spec = importlib.util.spec_from_file_location(filename, SOURCE / "launch" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_launch_description()


def test_launches_expose_acquire_configuration_without_forced_mode():
    for filename in ("state_manager.launch.py", "search_state_manager.launch.py", "piracer_sim.launch.py"):
        description = launch_description(filename)
        arguments = {entity.name for entity in description.entities
                     if isinstance(entity, DeclareLaunchArgument)}
        assert "acquire_params_file" in arguments
        assert "force_acquire" not in arguments


def test_acquire_receives_parameter_file_and_typed_simulation_time():
    description = launch_description("state_manager.launch.py")
    node = next(entity for entity in description.entities
                if isinstance(entity, Node) and entity._Node__node_name == "piracer_acquire")
    context = LaunchContext()
    context.launch_configurations.update(acquire_params_file="/chosen/acquire.yaml", use_sim_time="true")
    parameters = node._Node__parameters
    assert perform_substitutions(context, parameters[0].param_file) == "/chosen/acquire.yaml"
    values = parameters[1]
    key = next(key for key in values if perform_substitutions(context, key) == "use_sim_time")
    assert isinstance(values[key], ParameterValue) and values[key].value_type is bool
    assert ParameterValue(values[key].value, value_type=bool).evaluate(context) is True


def test_nested_launches_forward_acquire_file_and_preserve_follow_options():
    for filename in ("search_state_manager.launch.py", "piracer_sim.launch.py"):
        description = launch_description(filename)
        child = next(entity for entity in description.entities
                     if isinstance(entity, IncludeLaunchDescription)
                     and "acquire_params_file" in dict(entity.launch_arguments))
        arguments = dict(child.launch_arguments)
        context = LaunchContext()
        context.launch_configurations["acquire_params_file"] = "/chosen/acquire.yaml"
        assert arguments["acquire_params_file"].perform(context) == "/chosen/acquire.yaml"
        assert {"follow_publish_cmd_vel", "follow_nav2_avoidance"} <= set(arguments)
