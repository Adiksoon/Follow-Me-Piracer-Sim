# PiRacer SEARCH, ACQUIRE and FOLLOW

The State Manager starts SEARCH after fresh detector data arrives. Two positive YOLO frames in a three-frame window, together with an eligible 3D suitcase point, select ACQUIRE. Five new valid 3D points after that entry select FOLLOW; repeated points do not count. Three consecutive misses in direct FOLLOW select ACQUIRE if its last point is at most 3 s old, otherwise SEARCH. A fresh FOLLOW status pauses this miss counter while an already-owned bypass is being completed to its selected observation goal. An unowned planning intent does not pause it. Detector silence of 2 s still pauses the manager, and manual STOP always takes priority. There is no forced development state.

## Launch profiles

- Full simulation: `ros2 launch piracer_state_manager piracer_sim.launch.py`. This includes SLAM/Nav2 and Search, starts paused by default, selects `robot_perception/config/detector_ft01_test.yaml`, and enables the integrated Follow/Nav2 avoidance path. `follow_publish_cmd_vel` still defaults to `false`, so Follow reports commands but does not actuate. The configured detector weights must exist at the absolute path in that YAML: `/home/zygar/dev_praca_inz/symulacja/verification/yolo_ft01_20261012/runs/yolo11n_suitcase_ft01_final/weights/best.onnx`. The weights are delivered separately and are not part of this repository.
- Standalone state manager: `ros2 launch piracer_state_manager state_manager.launch.py`. `follow_nav2_avoidance` defaults to `false`; Follow uses direct `/cmd_vel` control and has no costmap-based obstacle avoidance. `search_state_manager.launch.py` adds Search and also defaults the integrated Follow/Nav2 option to `false`.

`follow_publish_cmd_vel` controls FOLLOW output only: SEARCH and ACQUIRE can move through Nav2 even when it is false. Keep physics paused during setup and use `/piracer/state/enable` with `data: false` to disable all behaviors. Set `follow_publish_cmd_vel:=true` only when FOLLOW drive output is intended. In the full simulation profile, integrated Follow/Nav2 routes Follow output separately from Search and uses command-owner handoff. In direct mode, Follow and other publishers can share `/cmd_vel`, so command arbitration is not provided by this Follow option.

## Control and safeguards

The default Follow controller targets 2.25 m, stops at or inside 2.0 m, limits forward speed to 0.05 m/s, and limits steering to 0.554 rad. Distance and direction use P control by default. Optional I and D gains default to zero; no gains are claimed to be optimal. Search's Nav2 desired speed remains 0.10 m/s, with a configured minimum turning radius of 0.50 m.

Set the desired distance once at startup, for example `ros2 launch piracer_state_manager piracer_sim.launch.py follow_target_distance_m:=1.0`. The same value drives the distance controller, Nav2 stand-off goals, remaining-path obstacle checks, ACQUIRE observation clearance and reported error. The minimum STOP distance is always 0.25 m below this setpoint: 1.0 m gives 0.75 m, while the default 2.25 m retains 2.0 m. The value must be finite and greater than 0.25 m. The node's `target_distance_m` parameter is read-only after startup; restart to change it. Camera visibility and depth validity still constrain achievable following distances.

The simulation launch accepts `follow_max_speed_mps:=0.075` as an optional Follow speed cap; its default remains `0.05`. Integrated Follow checks the depth obstacle scan before transferring control to Nav2. It can continue toward the last observed suitcase position during an existing bypass and returns to direct following after fresh target observations confirm a clear corridor. Nav2 updates its approach heading while retaining the current action's destination. An unavailable path or invalid localization stops the selected output.

`/piracer/state/enable` is a `std_srvs/srv/SetBool` service; `data: false` latches the manager stopped and requests behavior disable. Follow also exposes `/follow/enable`. The integrated simulation adapter accepts commands for the currently confirmed behavior owner and emits zero when its command stream has been quiet for 0.5 seconds. These software mechanisms are not a validated hardware safety system and do not establish production approval.

Direct Follow requires a fresh, valid target point and usable TF. Integrated mode also checks localization and costmap freshness, the current and swept footprint, and Nav2 goal ownership before forwarding commands, including continuation to the committed observation goal after target loss. After target loss, FOLLOW finishes that same goal without sending a second goal to the remembered object surface. Successful arrival clears target memory; the existing Nav2 result and both 0.3 s drain clocks resolve before the miss counter can select ACQUIRE or SEARCH. Fresh reacquisition uses the existing direct-path checks. Stale required inputs stop the selected output. SEARCH, ACQUIRE, and FOLLOW share the Hybrid planner with unknown cells enabled and path smoothing disabled; live obstacle sensing and collision checks remain active. Costmaps, RPP and the velocity smoother are shared by both behaviors. Controller samples on `/piracer/follow/control_sample` are observational and do not drive the robot; accumulated diagnostics are published on `/piracer/follow/control_diagnostics`.

## Optional FOLLOW terms and diagnostic interface

FOLLOW defaults to P/P: both I gains and both D gains are zero. Optional D
terms now use the unfiltered error difference divided by the interval between
new measurement stamps; direction uses the normalized angular difference.
Repeated stamps hold the last derivative. Reset, regressing stamps and hard
stop clear derivative history. I terms, their limits and command limits remain
available with the same rules.

The controller no longer accepts `distance_filter_tau_s` or
`direction_filter_tau_s`. The ROS node no longer declares these parameters,
and these keys are also removed from `FOLLOW_CONFIG`.
`FollowCommand` no longer exposes `distance_d_filtered_mps` or
`direction_d_filtered_radps`. The control-sample JSON (schema version 2) drops
`distance_filtered_derivative_mps` and replaces
`direction_filtered_derivative_radps` with `direction_raw_derivative_radps`.
The retained `distance_raw_derivative_mps` and new direction key report the
derivative used by D, including the held value on a repeated stamp and zero
after a reset. Gain-weighted `distance_d_mps` and `direction_d_rad` keep their
names. No field-specific consumer of these samples exists in the tracked
runtime code. External consumers must use the raw keys and remove tau
overrides. Older PD/PID experiments with positive D gains used filtered
derivatives and therefore describe a different controller version; their
recorded data and analyses remain unchanged.

## Development evidence

Implementation and offline evidence are maintained on development branch `codex/follow-suitcase`, under `/home/zygar/dev_praca_inz/worktrees/follow-suitcase/verification/follow_01/`. The implementation archive is `/home/zygar/dev_praca_inz/worktrees/follow-suitcase/verification/follow_01/IMPLEMENTATION/`; the dated preparation and source regression receipts are under `/home/zygar/dev_praca_inz/worktrees/follow-suitcase/verification/follow_production_preparation_20261102/`. These are development records, not evidence of live hardware validation or production readiness.


## ACQUIRE and handoff

ACQUIRE receives `/piracer/follow/target_point` even while disabled. It transforms the original stamped point to `vo_odom`; admission requires an observation no older than 3 s. The admitted position is frozen for a bounded attempt. Its three forward maneuvers are a short approach, an observation arc toward the last observed side, and an arc to the opposite side. They use the existing 0.50 m minimum planning radius, without reversing or turning in place. Defaults are in `config/acquire.yaml`; `acquire_params_file` is forwarded by all three launchers. The shared `follow_target_distance_m` argument overrides ACQUIRE's stand-off distance after loading that file.

ACQUIRE uses ComputePathToPose, validates the whole path and passes that same path to FollowPath, without replanning it. The launcher selects `/piracer/follow_nav/follow_path` when integrated avoidance is enabled and `/follow_path` otherwise, matching the active Nav2 action endpoint. Localization, scan, travel and time checks remain active. The attempt is limited to three maneuvers, 30 s and 1.2 m total travel with a 0.15 m stopping margin. Terminal motion requires 2 s of fresh, stationary odometry and TF before ownership is released. These are simulation safeguards, not a measured hardware stopping guarantee.

The manager correlates `/piracer/acquire/result` with the enable response's attempt ID. Normal exhaustion, path rejection or ordinary cancellation returns to SEARCH after confirmed release. `already_close` waits without repeating motion for the five 3D confirmations or the 30 s deadline. Localization/TF faults and genuine action failures latch STOP instead of silently restarting SEARCH. Five confirmations can select FOLLOW before a maneuver ends, but the coordinator still waits for ACQUIRE disable, its terminal result and confirmed stop before enabling FOLLOW.

FOLLOW publishes `/piracer/follow/status` as a standard String: DISABLED, DIRECT, AVOIDING or FAULT. This is the functional handoff signal; measurement JSON is not used to control transitions. AVOIDING includes owned goal admission, execution, cancellation/drain and remembered continuation. Missing status for 0.5 s while FOLLOW is confirmed active, or a reported fault, latches STOP. FOLLOW and ACQUIRE also carry their fault latch in every disable response, covering a fault during handoff even if its result topic is delayed or its old attempt ID was invalidated. A new authorized enable clears old status evidence. Original ROS stamp age and monotonic receipt age are checked separately for 3D points; points never renew the detector heartbeat.

This integration is developed on `codex/search-acquire-follow`. Existing ACQUIRE offline evidence is under `/home/zygar/dev_praca_inz/worktrees/acquire-clean/verification/acquire/DIRECTIONAL_TURN_20261103/`. The earlier fixed-order normal-workflow run is under `verification/acquire/MERGE_READY_20261103/runtime/run_06/` in that worktree; it predates directional selection and this integration. Natural integration run_05 passed on commit `645bee4`; its results and review are preserved in `/home/zygar/piracer_archives/workflow/full-scenario-20261105-645bee4/`. Separate blind run07 passed arrival and handoff checks but remained PARTIAL overall because SEARCH did not resume motion; run08 stopped on predicted collision and did not pass. These simulation results do not qualify the physical robot.
