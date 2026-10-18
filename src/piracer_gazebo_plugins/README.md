# PiRacer Gazebo steering adapter

`piracer_kinematic_steering` subscribes to the PiRacer `cmd_vel` command and
sets the front left/right steering joints to Ackermann angles. It clamps the
angles to the configured range and limits their rate. Rear wheel motion is
handled by `gazebo_ros_diff_drive` in the robot description.

The optional `trace_path` SDF element enables per-update JSONL diagnostics for
verification runs. Omitting it disables file output, as the production model
does.

The current working wheel-angle limit is 45 degrees, based on the author's
approximate observation rather than hardware calibration. The configured
steering rate remains 0.10 rad/s. New tighter-turn dynamics require a separate
runtime observation; changing the limit does not remove gate rate rejection.
