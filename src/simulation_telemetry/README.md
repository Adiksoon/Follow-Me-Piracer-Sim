# Simulation telemetry

This read-only ROS 2 package records simulation signals as one JSON object per
line. It does not change robot commands or the simulation launch.

Build from the `symulacja/` workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src --packages-select simulation_telemetry
source install/setup.bash
```

Start the simulation in one terminal, then record telemetry in another:

```bash
ros2 launch simulation_telemetry telemetry.launch.py \
  output_file:=verification/telemetry/run.jsonl
```

The recorder writes a sample every second by default. Samples include simulation
time and real-time factor, rates for the configured topics, latest PiRacer and
suitcase odometry and velocity commands, RGB-D image metadata, CameraInfo
metadata, and `/diagnostics`. Image messages are received to count their rate,
but pixel data is not written to the JSONL file. The output file is appended;
Ctrl+C stops the recorder and closes it. Omitting `output_file` writes JSONL to
stdout. `sample_period_sec` and topic names can be overridden as ROS parameters.
For example, run the node directly to change the odometry topic:

```bash
ros2 run simulation_telemetry telemetry_recorder --ros-args \
  -p robot_odom_topic:=/robot/odom -p sample_period_sec:=2.0
```

The defaults match this workspace: `/odom`, `/sim012_target/odom`,
`/piracer/cmd_vel`, `/sim012_target/cmd_vel`, the four `/piracer/sim_camera/*`
image and CameraInfo topics, `/diagnostics`, and `/clock`.
