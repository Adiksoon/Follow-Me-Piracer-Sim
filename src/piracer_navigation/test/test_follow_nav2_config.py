from pathlib import Path
import ast

import yaml


ROOT = Path(__file__).parents[1]


def test_local_and_global_costmaps_share_static_and_live_scan_layers():
    config = yaml.safe_load((ROOT / "config/nav2_params.yaml").read_text())
    global_params = config["global_costmap"]["global_costmap"]["ros__parameters"]
    local_params = config["local_costmap"]["local_costmap"]["ros__parameters"]
    assert global_params["plugins"] == ["static_layer", "obstacle_layer", "inflation_layer"]
    assert local_params["plugins"] == ["static_layer", "obstacle_layer", "inflation_layer"]
    assert global_params["obstacle_layer"]["scan"]["max_obstacle_height"] == 2.0
    assert local_params["obstacle_layer"]["scan"]["max_obstacle_height"] == 2.0
    for params in (global_params, local_params):
        assert params["static_layer"]["map_topic"] == "/map"
        assert params["static_layer"]["map_subscribe_transient_local"] is True
        assert params["obstacle_layer"]["scan"] == global_params["obstacle_layer"]["scan"]
        assert params["obstacle_layer"]["scan"]["topic"] == "/piracer/scan"
        assert params["obstacle_layer"]["scan"]["data_type"] == "LaserScan"
        assert params["footprint"] == global_params["footprint"]
    assert local_params["global_frame"] == "vo_odom"
    assert local_params["rolling_window"] is True
    assert local_params["resolution"] == global_params["resolution"] == 0.05
    assert config["planner_server"]["ros__parameters"]["Hybrid"]["allow_reverse_expansion"] is False
    assert config["controller_server"]["ros__parameters"]["FollowPath"]["allow_reversing"] is False
    assert config["controller_server"]["ros__parameters"]["FollowPath"]["use_collision_detection"] is True


def test_inflation_radius_covers_configured_footprint_circumradius():
    config = yaml.safe_load((ROOT / "config/nav2_params.yaml").read_text())
    global_params = config["global_costmap"]["global_costmap"]["ros__parameters"]
    local_params = config["local_costmap"]["local_costmap"]["ros__parameters"]
    footprint = ast.literal_eval(global_params["footprint"])
    circumradius = max((x * x + y * y) ** 0.5 for x, y in footprint)

    assert ast.literal_eval(local_params["footprint"]) == footprint
    for params in (global_params, local_params):
        assert params["inflation_layer"]["inflation_radius"] >= circumradius


def test_integration_flag_defaults_and_propagation_preserve_standalone_routing():
    state_launch = (ROOT.parent / "piracer_state_manager/launch/state_manager.launch.py").read_text()
    search_launch = (ROOT.parent / "piracer_state_manager/launch/search_state_manager.launch.py").read_text()
    sim_launch = (ROOT.parent / "piracer_state_manager/launch/piracer_sim.launch.py").read_text()
    nav_launch = (ROOT / "launch/slam_nav2.launch.py").read_text()
    base_launch = (ROOT.parent / "robot_simulation/launch/piracer_sim_base.launch.py").read_text()

    assert 'DeclareLaunchArgument("follow_nav2_avoidance", default_value="false")' in state_launch
    assert 'DeclareLaunchArgument("follow_nav2_avoidance", default_value="false")' in search_launch
    assert 'DeclareLaunchArgument("follow_nav2_avoidance", default_value="false")' in nav_launch
    assert 'DeclareLaunchArgument("follow_nav2_avoidance", default_value="false")' in base_launch
    assert 'DeclareLaunchArgument("follow_nav2_avoidance", default_value="true")' in sim_launch
    assert '"follow_nav2_avoidance": LaunchConfiguration("follow_nav2_avoidance")' in search_launch
    assert '"follow_nav2_avoidance": LaunchConfiguration("follow_nav2_avoidance")' in nav_launch
    assert '"require_behavior_owner"' in base_launch
    assert "cmd_vel_follow" in state_launch and "cmd_vel" in state_launch
    assert '"follow_publish_cmd_vel",\n                default_value="false"' in sim_launch


def test_state_manager_command_topic_flag_is_case_insensitive():
    state_launch = (ROOT.parent / "piracer_state_manager/launch/state_manager.launch.py").read_text()
    assert "'.lower() == 'true' else '/cmd_vel'" in state_launch
    for value, expected in (
        ("true", "/cmd_vel_follow"),
        ("True", "/cmd_vel_follow"),
        ("false", "/cmd_vel"),
        ("False", "/cmd_vel"),
    ):
        expression = f"'/cmd_vel_follow' if '{value}'.lower() == 'true' else '/cmd_vel'"
        assert eval(expression) == expected


def test_adapter_gate_accepts_only_confirmed_owner_and_selected_input():
    adapter = (ROOT.parent / "piracer_gazebo_plugins/src/ackermann_twist_adapter.cpp").read_text()
    assert 'declare_parameter<bool>("require_behavior_owner", false)' in adapter
    assert 'if (!require_behavior_owner_ || command_owner_ == "search")' in adapter
    assert 'if (require_behavior_owner_ && command_owner_ == "follow")' in adapter
    assert 'command_publisher_->publish(geometry_msgs::msg::Twist{});' in adapter
    assert "have_received_command_ = false;" in adapter
    accept = adapter.split("void accept_command", 1)[1].split("double wheelbase_m_", 1)[0]
    assert "last_command_time_ = std::chrono::steady_clock::now();" in accept
    assert "have_received_command_ = true;" in accept


def test_adapter_first_owner_update_emits_zero_and_unselected_input_cannot_refresh_timeout():
    adapter = (ROOT.parent / "piracer_gazebo_plugins/src/ackermann_twist_adapter.cpp").read_text()
    owner_callback = adapter.split("command_owner_subscription_ =", 1)[1].split("timeout_timer_ =", 1)[0]
    assert "if (!require_behavior_owner_)" in owner_callback
    assert "have_received_owner_ && input->data == command_owner_" in owner_callback
    assert "command_owner_ = input->data;" in owner_callback
    assert "have_received_owner_ = true;" in owner_callback
    assert owner_callback.index("command_owner_ = input->data;") < owner_callback.index(
        "command_publisher_->publish(geometry_msgs::msg::Twist{});"
    )
    assert "bool have_received_owner_{false};" in adapter

    search_subscription = adapter.split('"cmd_vel", 10,', 1)[1].split('"/cmd_vel_follow", 10,', 1)[0]
    follow_subscription = adapter.split('"/cmd_vel_follow", 10,', 1)[1].split('"/piracer/state/command_owner",', 1)[0]
    assert 'if (!require_behavior_owner_ || command_owner_ == "search")' in search_subscription
    assert 'if (require_behavior_owner_ && command_owner_ == "follow")' in follow_subscription
    assert search_subscription.count("accept_command(*input)") == 1
    assert follow_subscription.count("accept_command(*input)") == 1
    for callback in (search_subscription, follow_subscription):
        assert "last_command_time_" not in callback
        assert "have_received_command_" not in callback
