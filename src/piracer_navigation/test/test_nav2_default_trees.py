"""Default BTs must use action servers provided by the PiRacer launch."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).parents[1]


def test_through_poses_tree_has_no_unavailable_recovery_actions():
    path = ROOT / "behavior_trees/navigate_through_poses_no_recovery.xml"
    tree = ET.parse(path)
    sequence = tree.find("./BehaviorTree/Sequence")
    assert [node.tag for node in sequence] == ["ComputePathThroughPoses", "FollowPath"]
    assert sequence[0].attrib == {"goals": "{goals}", "path": "{path}", "planner_id": "Hybrid"}
    assert sequence[1].attrib == {"path": "{path}", "controller_id": "FollowPath"}
    launch = (ROOT / "launch/slam_nav2.launch.py").read_text()
    assert '"navigate_through_poses_no_recovery.xml"' in launch
    assert "navigate_through_poses_w_replanning_and_recovery.xml" not in launch
    config = yaml.safe_load((ROOT / "config/nav2_params.yaml").read_text())
    assert config["behavior_server"]["ros__parameters"]["behavior_plugins"] == ["wait"]
