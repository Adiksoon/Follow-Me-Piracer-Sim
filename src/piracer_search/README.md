# piracer_search

A small ROS 2 package with a pure-Python frontier evaluator and one Search
node. It ranks frontier observation poses from an OccupancyGrid with the
GAIN_COST_VISIBLE_OPAQUE_W050 rule. Grid path cost is a ranking estimate;
Nav2 plans the route and handles obstacle and costmap feasibility.

The node subscribes to its configured map topic and `/visual_vo/odom_info`. It
uses the newest eligible positive OdomInfo as a health sample and checks the
latest `vo_odom -> base_link` transform and latest full `map -> base_link`
transform independently. The selected Info and both returned transform stamps
must each be no more than 0.5 seconds old. After localization loss, Info and
both transform streams must each advance beyond the recovery barrier before
localization is accepted. The maximum age is available as the
`max_localization_age_sec` parameter. The node owns one `NavigateToPose`
action client. `/search/enable`
(std_srvs/SetBool) starts or stops Search. While enabled, it periodically
ranks candidates, publishes the six highest-scoring markers on
`/piracer/frontier_markers`, and sends the selected goal to Nav2. Nav2 rejection
or failure is retried once; after a second failure Search advances to the next
eligible candidate. A challenger must exceed the current score by the configured
margin in two consecutive evaluations before replacing the active goal.

The 3x3 FREE-cell check is a coarse candidate prefilter; it does not establish
the physical robot footprint or Ackermann reachability. Nav2 remains responsible
for route planning and costmap-based obstacle handling.

The simulation launch uses the local URDF HFOV and the RTAB-Map `Grid/RangeMax`
value as Search scoring inputs. These are simulation values, not physical
camera calibration. Physical hardware values remain TBD.

Occupancy values 0 are FREE, -1 are UNKNOWN, and positive values are blocked;
intermediate values 1 through 99 are treated as UNKNOWN. No suitcase detection
or application-state decision is included. No live ROS, TF, Nav2, or robot
runtime behavior is verified by the offline tests.

Run offline tests from the package directory:

    PYTHONPATH=. python3 -m unittest discover -s test -v

The full application starts Search through `piracer_state_manager/launch/search_state_manager.launch.py`.
