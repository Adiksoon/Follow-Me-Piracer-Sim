# Production Gazebo configuration

The production scene is sala_laboratoryjna_production.world. It explicitly uses
ODE world with a 0.001 s step, 1000 Hz target update rate and the cone_model
setting. The standard launch selects NVIDIA rendering under WSLg.

The author selected this configuration after stationary comparisons on
2026-10-20. In one 20 s passive test, the retained model before simplification
with world had practically zero XY/yaw drift, no invalid samples and no LCP
errors. The simplified model with quick had 33.40 mm / 3.31 deg maximum drift;
switching it to world reduced this to 1.72 mm / 0.053 deg but produced LCP errors.
Do not silently replace the production solver with quick or an unspecified
default. A solver change requires an explicit comparison.

The successful passive test had the drive plugin and camera disabled. It does
not qualify the active Ackermann controller, camera-equipped model or driving.
The author-approved production selection retains the existing mechanical model
and official libgazebo_ros_ackermann_drive.so; it adds no pose anchoring.
The remaining cone_model label does not prove identical friction calculations
between QuickStep and WorldStep.

The author subsequently accepted the visible Gazebo/camera preview with the
production Ackermann plugin enabled. That session still logged LCP errors in
the world log. The positive visual assessment is retained separately from
numerical qualification and does not establish error-free driving.

Evidence and source hashes are retained in
[the solver finding](../../verification/vehicle_jitter_damping_ab_20261019/steering_inertia_fix_20261019/simple_physics_01/DRIFT_SOLVER_FINDING.md)
and its linked run records. Source backups and this production decision are in
the adjacent production_world_preview_20261020_01 directory.

## Gazebo and camera preview without VO

After sourcing ROS 2 Humble and this workspace:

```bash
ros2 launch robot_simulation piracer_sim_base.launch.py \
  gui:=true camera_enabled:=true camera_viewer_enabled:=true \
  visual_odometry_enabled:=false suitcase_control_gui:=false yolo_enabled:=false
```

This opens Gazebo and one RGB camera viewer with the existing Ackermann command
adapter and drive plugin. It does not start VO, YOLO, RViz, SLAM or Nav2 and sends
no movement commands. The camera topic is /sim_camera/color/image_raw.
The visual_odometry_enabled argument defaults to true to preserve the previous
launch behavior; VO is always skipped when the camera is disabled.

## Full program

The one-command simulation, VO, RTAB-Map, Nav2, perception, and state-managed
behavior entrypoint is:

```bash
ros2 launch piracer_state_manager piracer_sim.launch.py
```

Gazebo starts paused. After checking the ROS graph, press Play to advance
simulation time. The State Manager starts SEARCH after fresh detector data
arrives and selects ACQUIRE and FOLLOW as observations are confirmed.

`follow_publish_cmd_vel` defaults to `false`. SEARCH and ACQUIRE can still move
through Nav2. To disable all behaviors, call `/piracer/state/enable` with
`data: false`. Set `follow_publish_cmd_vel:=true` to allow FOLLOW output after
its motion checks pass. See the [State Manager guide](../piracer_state_manager/README.md)
for behavior transitions, command ownership and detector weights.

## Historical DATA-01 suitcase dataset capture

The commands below refer to the frozen DATA-01 inputs. The current classroom
world differs from the approved capture world and fails its hash check.

The dataset generator is installed as `suitcase_dataset_generator.py`. Without
`--capture`, it plans camera viewpoints and checks the approved world and mesh
offline; it does not contact ROS or start Gazebo. For example, to inspect a
20-view pilot plan:

```bash
ros2 run robot_simulation suitcase_dataset_generator.py --sample-count 20
```

To capture that pilot, start the independent dataset-camera simulation in one
terminal:

```bash
ros2 launch robot_simulation dataset_capture.launch.py scene:=production_room
```

After the launch is ready, run the generator in a second terminal:

```bash
ros2 run robot_simulation suitcase_dataset_generator.py --capture --sample-count 20
```

The default capture output is written under `verification/datasets/` in a
timestamped `suitcase_pilot_*` directory. Capture mode uses the already-running
DATA-01 camera session; it does not launch Gazebo itself. Diagnostic single
captures use the separate `--diagnostic-single-shot` mode and require an
explicit output directory.

DATA-01 pins the exact source worlds, suitcase mesh, and furniture models before
planning or capture. After removing the unused world-level `gazebo_ros_state`
plugin on 2026-10-25, the production world SHA-256 is
`a2db8fa83e785b87629c93dd750ec76e4744f44113989ed68e1eee70cdefe188`. This is now
the production-world pin in both `dataset_capture.launch.py` and
`suitcase_dataset_generator.py`. The independent empty-world, mesh, and furniture
hashes remain unchanged; all integrity checks stay enabled.

## Production classroom scene details

The production classroom scene now has three closed, divided windows with
wooden sills, a closed wooden door, a chalkboard with chalk marks, two teaching
posters, a low shelf with colored books, and two planters. The wall decorations
and procedural planter asset were authored locally for this scene; no external
meshes or textures were copied. These details are intended to add visual
texture for a future VO evaluation. Their appearance still needs author visual
review, and no VO improvement has been established. Existing room collisions,
furniture, suitcase, spawn positions, and physics
remain unchanged. The unused world-level gazebo_ros_state plugin in the
piracer_ground_truth namespace was removed. The separate joint-state
publisher in the PiRacer URDF remains.
