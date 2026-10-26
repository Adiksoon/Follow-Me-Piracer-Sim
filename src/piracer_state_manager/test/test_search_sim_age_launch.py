"""Simulation and child Search age parameters remain explicit and typed."""
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


def load_launch(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_child_launch_defaults_to_half_second_and_types_parameter_as_float():
    with patch(
        "ament_index_python.packages.get_package_share_directory",
        return_value="/share/piracer_state_manager",
    ):
        child = load_launch(
            "search_state_manager_launch_under_test",
            SOURCE / "launch/search_state_manager.launch.py",
        )
    description = child.generate_launch_description()
    argument = next(
        entity for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
        and entity.name == "max_localization_age_sec"
    )
    assert perform_substitutions(LaunchContext(), argument.default_value) == "0.5"
    search_node = next(
        entity for entity in description.entities
        if isinstance(entity, Node)
        and entity._Node__node_name == "piracer_search"
    )
    parameters = search_node._Node__parameters[0]
    key = next(
        key for key in parameters
        if perform_substitutions(LaunchContext(), key) == "max_localization_age_sec"
    )
    value = parameters[key]
    assert isinstance(value, ParameterValue)
    assert value.value_type is float
    launch_configuration = next(
        substitution for substitution in value.value
        if isinstance(substitution, LaunchConfiguration)
    )
    assert perform_substitutions(
        LaunchContext(), launch_configuration.variable_name
    ) == "max_localization_age_sec"


def test_full_simulation_forwards_one_second_to_search_child():
    root = load_launch(
        "piracer_sim_launch_under_test", SOURCE / "launch/piracer_sim.launch.py"
    )
    description = root.generate_launch_description()
    include = next(
        entity for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
        and "max_localization_age_sec" in dict(entity.launch_arguments)
    )
    arguments = dict(include.launch_arguments)
    assert arguments["max_localization_age_sec"] == "1.0"
