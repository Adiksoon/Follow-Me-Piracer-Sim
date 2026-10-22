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
ros2 launch robot_simulation piracer_sim.launch.py \
  gui:=true camera_enabled:=true camera_viewer_enabled:=true \
  visual_odometry_enabled:=false suitcase_control_gui:=false yolo_enabled:=false
```

This opens Gazebo and one RGB camera viewer with the existing Ackermann command
adapter and drive plugin. It does not start VO, YOLO, RViz, SLAM or Nav2 and sends
no movement commands. The camera topic is /piracer/sim_camera/color/image_raw.
The visual_odometry_enabled argument defaults to true to preserve the previous
launch behavior; VO is always skipped when the camera is disabled.

## DATA-01 suitcase dataset capture

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

Current limitation (2026-10-22): DATA-01's approved production-world hash is
pinned to `71111ed9cfd933117c3ac7013bed41c85d342eb5cc6b8e4e0eddbeeb5ae39c7e`,
but the current `sala_laboratoryjna_production.world` has SHA-256
`76570add543fbd17cbcd3030ab53e9306c0bec802805f965138390f84cfb1947`. The
offline plan command rejects this world, and the DATA-01 launch has a matching
pre-launch integrity check. The commands above describe the intended workflow;
DATA-01 cannot currently use the production world until this pin is reconciled
against the approved world revision. Do not bypass the check.

## Production classroom scene details

The production classroom scene now has three closed, divided windows with
wooden sills, a closed wooden door, a chalkboard with chalk marks, two teaching
posters, a low shelf with colored books, and two planters. The wall decorations
and procedural planter asset were authored locally for this scene; no external
meshes or textures were copied. These details are intended to add visual
texture for a future VO evaluation. Their appearance still needs author visual
review, and no VO improvement has been established. Existing room collisions,
furniture, suitcase, spawn positions, physics and plugins are unchanged.
