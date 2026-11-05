# PiRacer state manager and FOLLOW

The package runs the detector-driven state manager, Search, Acquire's service scaffold, and Follow. The current implementation temporarily forces FOLLOW after each fresh detector message. SEARCH and ACQUIRE transition code remains, but automatic SEARCH/ACQUIRE/FOLLOW behavior is not active. The controller and its measurement telemetry remain available.

## Launch profiles

- Full simulation: `ros2 launch piracer_state_manager piracer_sim.launch.py`. This includes SLAM/Nav2 and Search, starts paused by default, selects `robot_perception/config/detector_ft01_test.yaml`, and enables the integrated Follow/Nav2 avoidance path. `follow_publish_cmd_vel` still defaults to `false`, so Follow reports commands but does not actuate. The configured detector weights must exist at the absolute path in that YAML: `/home/zygar/dev_praca_inz/symulacja/verification/yolo_ft01_20261012/runs/yolo11n_suitcase_ft01_final/weights/best.onnx`. The weights are delivered separately and are not part of this repository.
- Standalone state manager: `ros2 launch piracer_state_manager state_manager.launch.py`. `follow_nav2_avoidance` defaults to `false`; Follow uses direct `/cmd_vel` control and has no costmap-based obstacle avoidance. `search_state_manager.launch.py` adds Search and also defaults the integrated Follow/Nav2 option to `false`.

Set `follow_publish_cmd_vel:=true` only when drive output is intended. In the full simulation profile, integrated Follow/Nav2 routes Follow output separately from Search and uses command-owner handoff. In direct mode, Follow and other publishers can share `/cmd_vel`, so command arbitration is not provided by this Follow option.

## Control and safeguards

The default Follow controller targets 2.25 m, stops at or inside 2.0 m, limits forward speed to 0.05 m/s, and limits steering to 0.554 rad. Distance and direction use P control by default. Optional I and D gains default to zero; no gains are claimed to be optimal. Search's Nav2 desired speed remains 0.10 m/s, with a configured minimum turning radius of 0.50 m.

Set the desired distance once at startup, for example `ros2 launch piracer_state_manager piracer_sim.launch.py follow_target_distance_m:=1.0`. The same value drives the distance controller, Nav2 stand-off goals, remaining-path obstacle checks and reported error. The minimum STOP distance is always 0.25 m below this setpoint: 1.0 m gives 0.75 m, while the default 2.25 m retains 2.0 m. The value must be finite and greater than 0.25 m. The node's `target_distance_m` parameter is read-only after startup; restart to change it. Camera visibility and depth validity still constrain achievable following distances.

The simulation launch accepts `follow_max_speed_mps:=0.075` as an optional Follow speed cap; its default remains `0.05`. Integrated Follow checks the depth obstacle scan before transferring control to Nav2. It can continue toward the last observed suitcase position during an existing bypass and returns to direct following after fresh target observations confirm a clear corridor. Nav2 updates its approach heading while retaining the current action's destination. An unavailable path or invalid localization stops the selected output.

`/piracer/state/enable` is a `std_srvs/srv/SetBool` service; `data: false` latches the manager stopped and requests behavior disable. Follow also exposes `/follow/enable`. The integrated simulation adapter accepts commands for the currently confirmed behavior owner and emits zero when its command stream has been quiet for 0.5 seconds. These software mechanisms are not a validated hardware safety system and do not establish production approval.

Direct Follow requires a fresh, valid target point and usable TF. Integrated mode also checks localization and costmap freshness, the current and swept footprint, and Nav2 goal ownership before forwarding commands, including continuation toward a remembered position. Stale required inputs stop the selected output. The `FollowHybrid` planner permits unknown map cells while live obstacle sensing and collision checks remain active; Search retains its separate planner policy. Costmaps, RPP and the velocity smoother are shared by both behaviors. Controller samples on `/piracer/follow/control_sample` are observational and do not drive the robot; accumulated diagnostics are published on `/piracer/follow/control_diagnostics`.

## Development evidence

Implementation and offline evidence are maintained on development branch `codex/follow-suitcase`, under `/home/zygar/dev_praca_inz/worktrees/follow-suitcase/verification/follow_01/`. The implementation archive is `/home/zygar/dev_praca_inz/worktrees/follow-suitcase/verification/follow_01/IMPLEMENTATION/`; the dated preparation and source regression receipts are under `/home/zygar/dev_praca_inz/worktrees/follow-suitcase/verification/follow_production_preparation_20261102/`. These are development records, not evidence of live hardware validation or production readiness.
