# piracer_search

A small ROS 2 package with a pure-Python frontier evaluator and a single
Search node. It ranks safe frontier observation poses with the existing
GAIN_COST_VISIBLE_OPAQUE_W050 rule: opaque UNKNOWN visibility gain and cardinal
FREE-grid path cost are independently normalized and combined with equal
weights. Grid path cost is a ranking estimate; Nav2 plans routes and handles
obstacle avoidance.

The 3x3 FREE-cell check is only a coarse grid prefilter; it does not establish
the physical robot footprint or Ackermann reachability.

The node subscribes to /map, obtains map -> base_link from TF, and owns one
NavigateToPose action client. Its only control service is /search/enable
(std_srvs/SetBool): true starts Search and false stops it and cancels its
current goal. Rejected, successful, or failed goals are marked attempted while
the frontier-cell set remains unchanged, so the node advances through the
ranking without immediately retrying the same candidate. A later frontier
change allows candidates to be considered again.

Launch defaults use the local simulation URDF HFOV of 1.14075 rad converted to
degrees and the RTAB-Map configuration `Grid/RangeMax` of 5.0 m as the effective
Search scoring bound. The URDF depth far clip is 20.0 m, but Search scores only
within the 5.0 m map range. These are simulation configuration values, not
physical camera calibration or physical depth qualification. Physical hardware
values remain TBD. Override both launch arguments only with locally verified
values for the selected runtime.

Occupancy values 0 are FREE, -1 are UNKNOWN, and positive values are blocked;
intermediate values 1 through 99 are treated as UNKNOWN. No suitcase detection
or application-state decision is included. The package has no search-complete
or search-failed protocol. No ROS, TF, Nav2, or robot runtime behavior has been
verified by the offline tests.

Run offline tests from the package directory:

    PYTHONPATH=. python3 -m unittest discover -s test -v

Launch:

    ros2 launch piracer_search search.launch.py

The launch supplies an installed `search_navigate_to_pose.xml` behavior tree
and enables `use_sim_time` by default. Direct node use with an empty
`behavior_tree` parameter waits without dispatching a goal. The current
recorded map can still leave the robot start cell unknown, which blocks safe
candidate generation until a valid localized free start is available.

For the verified production simulation at the fixed zero spawn only, the
explicit `search_with_start_area.launch.py` profile publishes a separate
`/search_map`. It seeds UNKNOWN cells wholly inside the fixed rectangle
[-0.25, 0.70] m by [-0.25, 0.25] m relative to the initial map-to-base pose;
positive occupancy and the raw `/map` remain unchanged. Search and Nav2 use
the derived map with transient-local QoS. The seeded region is an explicit
simulation startup assumption, not RGB-D evidence or a physical-hardware
setting. The profile pins the production world and zero spawn; start a fresh
profile after an RTAB-Map reset because the fixed anchor is valid for one map
frame only. Search remains disabled until `/search/enable` is called.
