# PiRacer RGB-D, RTAB-Map, and Nav2

## Current accepted mapping state

The author accepted the RTAB-Map 2D map and its `/cloud_map` visual inspection
from CP-MAP-MANUAL-01 for the engineering work. This is visual acceptance, not
metric validation of VO accuracy. CP-MAP-PRODUCTION-01 sets the native TF path
as the default and adds the Nav2-only initial free-zone assumption documented
below. The D0/D1 material below is retained as a historical integration
snapshot; statements marked historical do not describe the current accepted
mapping configuration.

This package prepares one small simulation integration:

`RGB-D + existing VO -> RTAB-Map -> /map -> Nav2 -> /cmd_vel -> Ackermann adapter -> PiRacer`

D0 offline preparation passed 88/88 checks on 2026-10-16. Reproduce it with:
python3 verification/slam_nav2_integration_01/offline_checks.py
The checker does not launch ROS or Gazebo. No system packages were installed.

The latest D1 diagnostic, R9R1, received Astra PASS and passed a 16/16
same-shell preflight. Its one stationary run ended INCONCLUSIVE after
15.474858462 s of the required 30 s because Gazebo XY displacement reached
0.005006698 m against the unchanged 0.005 m stop limit. Yaw change was
0.023148366 rad, under the 0.05 rad limit. The independent monitor recorded
405 exact-zero /piracer/cmd_vel messages and cleanup passed. D1 remains
NOT_ESTABLISHED; this run's authority is consumed. The recorder retained one
full /map message with 995 known cells, but one partial snapshot does not
establish map quality. Offline tire-orientation analysis found that the large
world-z angular-velocity components are not direct steering or wheel-spin
rates; the projected axle component remains an absolute link rate. The cause
of pose drift remains unknown. See
verification/slam_nav2_integration_01/RESULT.md and
verification/slam_nav2_integration_01/d1_acceptance_20261017_05/R9_POSE_DRIFT_OFFLINE_ANALYSIS.md.

The existing VO remains a prototype for its tested simulation profile. RUN09
failed the frozen 50 mm p95 XY gate; no general accuracy or physical readiness
claim is made.

## Historical D1 candidate: data and TF ownership

This table records the earlier D1 candidate before the accepted manual mapping
inspection and native-TF production default.

| Data | Topic or edge | Producer | Consumer / owner | Status |
| --- | --- | --- | --- | --- |
| RGB image | `/sim_camera/color/image_raw` | Gazebo camera | VO and RTAB-Map | Seen in the 2026-10-15 VO smoke |
| Depth image | `/sim_camera/depth/image_raw` | Gazebo camera | VO, RTAB-Map, depth scan converter | Seen in the VO smoke |
| Color CameraInfo | `/sim_camera/color/camera_info` | Gazebo camera | VO and RTAB-Map | Seen in the VO smoke |
| Depth CameraInfo | `/sim_camera/depth/camera_info` | Gazebo camera | `depthimage_to_laserscan` | Seen in the VO smoke |
| Odometry | `/visual_odom` (`nav_msgs/msg/Odometry`) | Existing RTAB-Map VO | RTAB-Map and Nav2 | Interface smoke passed; not generally accuracy-qualified |
| VO status | /visual_vo/odom_info (rtabmap_msgs/msg/OdomInfo) | Existing VO node | TF bridge | 280 matching source-stamp samples with /visual_odom in R9R1; D1 window incomplete |
| Occupancy grid | /map (nav_msgs/msg/OccupancyGrid) | rtabmap_slam/rtabmap | Global costmap | One R9R1 message had 995 known cells; full data retained, map quality not established |
| Current obstacle scan | `/piracer/scan` (`sensor_msgs/msg/LaserScan`) | `depthimage_to_laserscan` | Local rolling costmap and command gate freshness check | Not runtime-verified |
| Nav2 controller output | `/cmd_vel_nav` | Nav2 controller server | Nav2 velocity smoother | Standard Humble remapping |
| Smoothed Nav2 command | `/cmd_vel` (`geometry_msgs/msg/Twist`) | Nav2 velocity smoother | Command gate | Standard Humble remapping; gate does not forward before enable |
| Final drive command | `/piracer/cmd_vel` | Command gate only | Rear diff-drive and steering plugins | Gate starts closed and outputs zero |
| Enable / STOP | `/piracer_navigation/enable_motion`, `/piracer_navigation/stop` | Explicit operator messages | Command gate | STOP is latched; a trip requires disable then explicit re-enable |

| TF edge | Sole owner | Source |
| --- | --- | --- |
| map -> vo_odom | RTAB-Map | 310 transforms recorded in R9R1; D1 acceptance remains open |
| vo_odom -> base_link | piracer_odom_tf_bridge | Copies valid VO pose and original stamp; 115 transforms recorded in R9R1 |
| base_link -> ... -> sim_camera_link -> sim_camera_optical_frame | robot_state_publisher | 12 /tf_static edges recorded in R9R1, including the camera chain |

In that historical D1 candidate, the VO node retained `publish_tf: false` and
the bridge owned `vo_odom -> base_link`. The current native-TF default is
documented in the production section below. The standalone VO launch still
defaults `publish_tf` to false, while `slam_nav2.launch.py` explicitly enables
it in native mode. Diff-drive odometry and wheel TF remain disabled. No
`odom -> vo_odom` edge, AMCL, map server, second mapper, teleop, suitcase GUI,
Follow-Me node, or custom planner is added by the mapping launch.

The listed `/sim_camera/sim_camera/points` topic is not used because its
type, frame, data quality, and freshness were not confirmed by the existing
smoke. The standard installed `depthimage_to_laserscan` package is used instead.
Its output frame is `sim_camera_link`, which is body-aligned; the input image
uses the optical frame. The configured one-row scan observes only a forward
horizontal slice. The camera has a 1.14075 rad horizontal FOV, is 0.12 m above
the base frame, and has no rear or full side coverage. Low floor objects may be
missed. A future controlled path must avoid relying on this scan as a general
safety sensor.

## Historical D1 candidate: motion limits and gate

The steering plugin source uses wheelbase 0.155 m, front track 0.148 m, maximum
front wheel angle 0.7853981633974483 rad (45 degrees), and steering rate 0.10 rad/s.
The angle is a provisional author-observed simulation assumption, not a measured
hardware specification. Its inner-wheel bound gives a center turning radius of
0.229 m; planner and gate use 0.25 m with margin. The forward speed limit remains
0.05 m/s and the smoother yaw-rate ceiling is 0.2 rad/s (speed divided by radius).
The rate-rejection policy is unchanged; this configuration does not establish
dynamic turn execution. These are
kinematic/configuration bounds, not measured vehicle performance.

The collision geometry gives an overall unpadded XY envelope of approximately
`x=[-0.010, 0.236] m`, `y=[-0.0985, 0.0985] m` around `base_link`. The costmap
footprint adds 0.010 m on each side. The global costmap uses `/map` and retains
UNKNOWN cells; the Hybrid planner rejects unknown space. The local rolling
costmap uses `vo_odom` and live scan observations. A map is not treated as a
replacement for current obstacle data.

The command gate copies only feasible forward planar Twist commands. It rejects
reverse, turn-in-place, non-finite, over-speed, over-curvature, and too-fast
steering-target changes instead of clipping them. It requires fresh valid VO
status, scan, and `/clock` input. The gate estimates front-wheel angles from
the last published command, advancing that estimate only by `/clock` time and
the adapter's 0.10 rad/s steering limit. A moving command must request a wheel
angle reachable within one 0.05 s control interval. STOP publishes a zero
target immediately but does not reset the estimated angle; rearming also keeps
that estimate. If simulation time moves backward, the gate stays disarmed
until the owned integration session is restarted. Receipt timeouts use the
system monotonic clock so the watchdog continues if simulation time pauses.
Missing or stale clock, invalid or stale state, STOP, or active-command timeout
closes the gate and publishes zero. The launch shuts down its owned session if
the gate process exits.

The integrated `slam_nav2.launch.py` always starts Nav2 and keeps the scan
interlock enabled. It rejects `require_scan:=false`, so this stack cannot be
launched with scan validity removed from the command-gate safety checks. The
scan interlock does not stop the depth-to-scan converter or remove the Nav2
obstacle layer. If a manual command is issued while the scan is invalid or
stale, `command_gate` stops the robot.

The steering angle is a software estimate, not joint feedback. Its initial
zero-angle assumption comes from the simulation model's initial steering pose.
The 0.05 s horizon and 0.25 s receipt-age limits are provisional configuration
values. Offline checks do not validate actuator tracking, clock delivery, or
the estimate against Gazebo joint state; those remain outside D0 and require
separately authorized runtime review.

## Historical D0/D1 runtime status (2026-10-17)

- D0 offline: PASS, 88/88 checks; no ROS/Gazebo launch. The check covers
  source syntax, VO/OdomInfo freshness, steering feasibility, STOP and timeout
  logic, RTAB-Map parameters, Nav2 plugin IDs and manifests, TF/remappings, BT,
  lifecycle, and footprint bounds.
- Configuration: PASS_CANDIDATE for offline integration. Source and installed
  copies match for the launch, YAML files, BT, TF bridge, motion rules, and
  command gate. RTAB-Map 0.23.7 comes from the task-local overlay; the system
  package is absent. Installed Nav2 packages are 1.1.20 and
  depthimage_to_laserscan is 2.5.1.
- D1 mapping: INCONCLUSIVE, acceptance NOT_ESTABLISHED. R9R1 passed the exact
  Astra review and 16/16 preflight, then ended after 15.474858462 s of the
  required 30 s when Gazebo XY displacement reached 0.005006698 m, 6.7
  micrometres over the 0.005 m stop limit. Yaw change was 0.023148366 rad,
  below its limit. The independent monitor recorded 405 exact-zero commands;
  gate and monitor traces validated. R9R1 recorded 280 matching
  /visual_odom and OdomInfo stamps, 310 map -> vo_odom and 115 vo_odom ->
  base_link transforms, 12 /tf_static edges, 328 Gazebo model poses, and one
  full /map message (105x94 at 0.05 m, 995 known cells). This partial map does
  not establish geometry quality. The pose-change cause remains UNKNOWN; see
  the offline analysis in d1_acceptance_20261017_05/. D1 remains
  NOT_ESTABLISHED.
- D2: NOT RUN. It requires D1 acceptance and its own bounded plan with the
  motion gate closed, using only ComputePathToPose.
- D3: NOT RUN. It requires D2 acceptance and the author's explicit approval
  for the exact path, movement limits, stop method, and visual observation.

RETRY05, RETRY06, RETRY07, and R9R1 authorizations are consumed. Under the
author's current policy, Astra PASS for a new exact non-visual LIVE plan
authorizes that plan after a complete fresh preflight; a stage requiring
visual judgment still needs the author's direct participation. No further
LIVE run has been started under the consumed R9R1 authority. The end-to-end
goal remains BLOCKED_AT_D1; D2 and D3 have not been evaluated.

## Existing components and provenance

- RTAB-Map source reference: [Humble 0.23.7 pinned launch and wrapper source](https://github.com/introlab/rtabmap_ros/tree/c25a091c2a682f0e0f4e8d19d807953e7f1a8161).
  This package starts the official `rtabmap_slam/rtabmap` node directly, so no
  internal odometry estimator is launched and no `cfg` INI is misused as ROS
  YAML. The system package is absent; the task-local overlay described above
  supplies the executable for the prepared launch.
- Historical D0/D1 obstacle conversion: ROS 2 Humble `depthimage_to_laserscan` 2.5.1.
  No upstream source code is copied.
- Navigation: installed ROS 2 Humble Nav2 1.1.20, using its registered Smac
  Hybrid planner and Regulated Pure Pursuit controller plugins.
# Production mapping and Nav2 map assumptions

The accepted mapping setup uses the production PiRacer model and RGB-D camera,
RTAB-Map RGB-D visual odometry, and RTAB-Map mapping. Mapping consumes the
camera RGB image, registered depth image, color CameraInfo, and `/visual_odom`.
RTAB-Map publishes `/map`, `/cloud_map`, and `map -> vo_odom`; native visual
odometry publishes `vo_odom -> base_link`. The visual inspection recorded in
`verification/CP-MAP-MANUAL-01/REPORT.md` accepted the 2D map and 3D cloud for
the engineering work. This was a visual acceptance, not a metric validation of
visual-odometry accuracy.

The simulation camera is enabled by the launch and publishes color, depth,
and CameraInfo below `/sim_camera`; its optical frame is
`sim_camera_optical_frame`. The Xacro attaches `sim_camera_link` to `chassis`
at x=0.236 m, z=0.320 m, with a 0 rad pitch, and attaches the optical
frame with the standard fixed optical rotation. The simulated sensor is
640 x 360 at 30 Hz. Its finite horizontal field of view and scene occlusion
limit which space can be observed from the camera.

`slam_nav2.launch.py` uses native VO TF as its only active mode. It disables
the VO instance inside the simulation base launch, then starts one
`rgbd_visual_odometry.launch.py` instance with `publish_tf=true`. The former
bridge fallback and its tests are preserved under
`artefakty/piracer_navigation/odom_tf_bridge_snapshot/`.

The current command path is direct: Nav2's velocity smoother, `piracer_follow`,
or an operator may publish `geometry_msgs/msg/Twist` on `/cmd_vel`. The
`ackermann_twist_adapter` subscribes to that topic and publishes the converted
Twist on `/piracer/ackermann_cmd_vel`, which the Gazebo Ackermann plugin
consumes. There is no command gate, central STOP latch, or arbitration between
simultaneous publishers. Keep only one command source active. Nav2 and Follow
retain their own configured speed limits; the adapter itself does not enforce
the removed gate's scan, VO, TF, clock, command-shape, or manual-arm checks.

`slam_nav2.launch.py` starts `robot_perception/obstacle_scan_node`, which
publishes a height-filtered `LaserScan` on `/piracer/obstacle_scan`. The full
`piracer_sim.launch.py` includes this launch and uses the same publisher.
This topic is the sole scan source for both Nav2 costmaps and also supplies
ACQUIRE and integrated FOLLOW.

The previous odometry bridge fallback and standalone mapping profile are
archived under `artefakty/piracer_navigation/`; they are not part of the active
navigation package.

Nav2 receives a derived map on `/nav2/map`. The source `/map` from RTAB-Map is
not modified. At each launch, `start_zone_map` waits for the simulation clock,
the latest map in frame `map`, and a current `map -> base_link` transform no
older than 0.20 s. It captures the initial base position once and converts only
unknown (`-1`) cell centers inside the configurable 1.5 m radius to free
(`0`). Existing free and occupied values are preserved, so a detected obstacle
has precedence. The circle is fixed for that process lifetime and is not moved
with the robot. A fresh launch starts with no retained center. Set
`start_zone_enabled:=false` to pass the raw map through, or change
`start_zone_radius` to configure the assumption.

The 1.5 m region is an approved engineering assumption for the starting area;
it is not evidence that all physical space there is clear. The RGB-D camera has
a limited field of view, so unknown space outside observed sensor coverage can
remain. The 2D map is generated from RGB-D depth, while `/cloud_map` provides a
3D geometric view for visual validation.
