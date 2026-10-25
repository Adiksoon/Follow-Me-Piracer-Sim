"""Short control-error patience keeps the existing TF and motion limits."""
from pathlib import Path
import yaml


def test_controller_error_patience_does_not_relax_transform_or_motion_limits():
    config = yaml.safe_load((Path(__file__).parents[1] / "config/nav2_params.yaml").read_text())
    controller = config["controller_server"]["ros__parameters"]
    assert controller["failure_tolerance"] == 0.3
    assert controller["controller_frequency"] == 20.0
    assert controller["progress_checker"]["required_movement_radius"] == 0.05
    assert controller["progress_checker"]["movement_time_allowance"] == 10.0
    assert controller["FollowPath"]["desired_linear_vel"] == 0.05
    for name in ("global_costmap", "local_costmap"):
        assert config[name][name]["ros__parameters"]["transform_tolerance"] == 0.2
    assert config["behavior_server"]["ros__parameters"]["transform_tolerance"] == 0.2
    smoother = config["velocity_smoother"]["ros__parameters"]
    assert smoother["max_velocity"] == [0.05, 0.0, 0.2]
    assert smoother["max_accel"] == [0.05, 0.0, 0.2]
    assert smoother["max_decel"] == [-0.1, 0.0, -0.4]
