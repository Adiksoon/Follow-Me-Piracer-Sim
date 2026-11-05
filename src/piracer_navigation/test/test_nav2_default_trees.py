"""Default BTs must use action servers provided by the PiRacer launch."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).parents[1]


def test_default_navigate_to_pose_propagates_follow_path_success():
    tree = ET.parse(ROOT / "behavior_trees/navigate_to_pose.xml")
    assert tree.getroot().get("main_tree_to_execute") == "MainTree"
    sequence = tree.find("./BehaviorTree[@ID='MainTree']/Sequence")
    assert sequence is not None
    assert [node.tag for node in sequence] == ["ComputePathToPose", "FollowPath"]
    assert sequence[0].attrib == {
        "goal": "{goal}", "path": "{path}", "planner_id": "Hybrid",
    }
    assert sequence[1].attrib == {
        "path": "{path}", "controller_id": "FollowPath",
        "goal_checker_id": "general_goal_checker",
    }


def test_through_poses_tree_has_no_unavailable_recovery_actions():
    path = ROOT / "behavior_trees/navigate_through_poses_no_recovery.xml"
    tree = ET.parse(path)
    sequence = tree.find("./BehaviorTree/Sequence")
    assert [node.tag for node in sequence] == ["ComputePathThroughPoses", "FollowPath"]
    assert sequence[0].attrib == {"goals": "{goals}", "path": "{path}", "planner_id": "Hybrid"}
    assert sequence[1].attrib == {
        "path": "{path}", "controller_id": "FollowPath",
        "goal_checker_id": "general_goal_checker",
    }
    launch = (ROOT / "launch/slam_nav2.launch.py").read_text()
    assert '"navigate_through_poses_no_recovery.xml"' in launch
    assert "navigate_through_poses_w_replanning_and_recovery.xml" not in launch
    config = yaml.safe_load((ROOT / "config/nav2_params.yaml").read_text())
    assert config["behavior_server"]["ros__parameters"]["behavior_plugins"] == ["wait"]


def test_search_uses_xy_checker_and_other_trees_keep_general_checker():
    config = yaml.safe_load((ROOT / "config/nav2_params.yaml").read_text())
    controller = config["controller_server"]["ros__parameters"]
    assert controller["goal_checker_plugins"] == [
        "general_goal_checker", "search_goal_checker", "acquire_goal_checker",
    ]
    assert controller["search_goal_checker"] == {
        "plugin": "nav2_controller::PositionGoalChecker",
        "stateful": False, "xy_goal_tolerance": 0.15,
    }
    assert controller["general_goal_checker"] == {
        "plugin": "nav2_controller::SimpleGoalChecker",
        "stateful": True, "xy_goal_tolerance": 0.05,
        "yaw_goal_tolerance": 0.15,
    }
    assert controller["acquire_goal_checker"] == {
        "plugin": "nav2_controller::PositionGoalChecker",
        "stateful": False, "xy_goal_tolerance": 0.05,
    }
    search_bt = (ROOT.parent / "piracer_search/behavior_trees"
                 / "search_navigate_to_pose.xml")
    for bt, checker in (
        (search_bt, "search_goal_checker"),
        (ROOT / "behavior_trees/navigate_to_pose.xml", "general_goal_checker"),
        (ROOT / "behavior_trees/navigate_through_poses_no_recovery.xml",
         "general_goal_checker"),
    ):
        nodes = list(ET.parse(bt).iter("FollowPath"))
        assert nodes
        assert all(node.get("goal_checker_id") == checker for node in nodes)
