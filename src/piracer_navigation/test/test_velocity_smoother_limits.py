"""Configured ramp prediction, not a test of the Nav2 implementation."""
from pathlib import Path
import pytest
import yaml


def test_scaled_turn_ramp_matches_linear_acceleration_and_stop():
    config = yaml.safe_load((Path(__file__).parents[1] / "config/nav2_params.yaml").read_text())
    p = config["velocity_smoother"]["ros__parameters"]
    assert p["scale_velocities"] is True
    assert p["max_velocity"] == [0.05, 0.0, 0.2]
    assert p["max_accel"] == [0.05, 0.0, 0.2]
    assert p["max_decel"] == [-0.10, 0.0, -0.4]
    # With proportional scaling the slowest component sets the ramp time.
    command = [0.05, 0.1950569203869006]
    ramp_s = max(command[0] / p["max_accel"][0], command[1] / p["max_accel"][2])
    stop_s = max(command[0] / abs(p["max_decel"][0]), command[1] / abs(p["max_decel"][2]))
    assert ramp_s == pytest.approx(1.0)
    assert stop_s == pytest.approx(0.5)
    assert p["velocity_timeout"] == 0.20
    progress = config["controller_server"]["ros__parameters"]["progress_checker"]
    assert progress["required_movement_radius"] == 0.05
    assert progress["movement_time_allowance"] == 10.0
