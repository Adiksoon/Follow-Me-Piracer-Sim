"""One-factor planner candidate; runtime must confirm endpoint tracking."""
from pathlib import Path
import math
import yaml


def test_planner_leaves_steering_margin_without_relaxing_goal_requirements():
    config = yaml.safe_load((Path(__file__).parents[1] / "config/nav2_params.yaml").read_text())
    planner = config["planner_server"]["ros__parameters"]["Hybrid"]
    controller = config["controller_server"]["ros__parameters"]
    # Existing adapter wheelbase and Gazebo max_steer, not a measured turn radius.
    actuator_radius_m = 0.155 / math.tan(0.554)
    assert planner["minimum_turning_radius"] == 0.50
    assert planner["minimum_turning_radius"] > actuator_radius_m
    assert planner["motion_model_for_search"] == "DUBIN"
    assert planner["allow_reverse_expansion"] is False
    assert controller["FollowPath"]["regulated_linear_scaling_min_radius"] == 0.25
    assert controller["FollowPath"]["use_rotate_to_heading"] is False
    assert controller["FollowPath"]["allow_reversing"] is False
    goal = controller["general_goal_checker"]
    assert goal["xy_goal_tolerance"] == 0.05
    assert goal["yaw_goal_tolerance"] == 0.15
