"""Check remapping in the actual navigation launch scope without starting ROS."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from launch import LaunchContext
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_ros.actions import SetRemap


@pytest.mark.parametrize("integrated", [False, True])
def test_private_nav2_endpoints_only_in_follow(integrated):
    path = Path(__file__).parents[1] / "launch/slam_nav2.launch.py"
    spec = importlib.util.spec_from_file_location("test_navigation_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = LaunchContext()
    ctx.launch_configurations["follow_nav2_avoidance"] = str(integrated).lower()
    with patch.object(module, "get_package_share_directory", side_effect=lambda name: "/share/" + name):
        nav = module._build_launch(ctx)[-1]
    if not integrated:
        assert isinstance(nav, IncludeLaunchDescription)
        return
    assert isinstance(nav, GroupAction)
    children = nav.get_sub_entities()
    include_index = next(i for i, child in enumerate(children) if isinstance(child, IncludeLaunchDescription))
    for rule in (x for x in children[:include_index] if isinstance(x, SetRemap)):
        rule.execute(ctx)
    assert dict(ctx.launch_configurations["ros_remaps"]) == {
        "behavior_server:cmd_vel": "/piracer/nav2_behavior_cmd_vel",
        **{f"follow_path/_action/{e}": f"/piracer/follow_nav/follow_path/_action/{e}"
           for e in ("send_goal", "get_result", "cancel_goal", "feedback", "status")},
    }
