# piracer_search

A small ROS 2 package with a pure-Python frontier evaluator and one Search
node. It groups FREE cells bordering UNKNOWN on a side into deterministic
8-neighbor clusters. It ranks eligible clusters by increasing cost
`C = r * (3.0 * d - 1.0 * n)`, using the weights from
[m-explore-ros2](https://github.com/robo-friends/m-explore-ros2/blob/326cf8a0b487c34246bb8f3326afbcd69576dc60/explore/config/params.yaml#L9-L14).
Here `r` is the map resolution, `d` is the minimum Euclidean distance in
world metres from the robot position to a frontier cell centre in the cluster,
and `n` is its frontier cell count. The node uses `score = -C`. Equal scores
are ordered by increasing BFS access distance, then by the anchor's world Y
and X. Candidate poses receive a full known-FREE footprint check at their
approach yaw, with nearby alternatives limited to 0.30 m. Nav2 plans the route.

The node subscribes to its configured map topic and `/visual_vo/odom_info_lite`.
It validates the latest eligible OdomInfo metadata and one coherent full
`map -> base_link` transform from TF2 `Time(0)`. The full simulation retains
the 1.0 second maximum age and 0.2 second TF future lead; direct/hardware
defaults remain 0.5 and 0.1 seconds. Lost samples, clock resets and the existing
recovery barrier still stop or prevent navigation. TF has its own executor;
OdomInfo keeps its separate callback group and metadata snapshot.
The node owns one `NavigateToPose` action client. `/search/enable`
(std_srvs/SetBool) starts or stops Search. While enabled, it periodically
ranks clusters. It checks the complete footprint for up to three safe points,
continuing to later clusters if earlier ones fail. Only these checked points
are displayed on `/piracer/frontier_markers` (the marker helper's cap is six).
Nav2 ABORTED exhausts a point; a rejected action request may be retried once.
Completed/exhausted point histories survive ordinary cluster-map growth and
reset on explicit disable/re-enable, as does the existing attempt budget.
After an accepted goal has been active for 10 seconds, Search may compare one
fresh result for a score-based switch. It matches the active goal's saved
frontier sources to the current coarse clusters and uses the highest score of
the matched clusters as the active score. All matched clusters are excluded
from the challengers. A different ready, footprint-checked cluster must improve
the active cost by at least `goal_switch_margin` (default 0.10 on the frontier
cost scale, with a numerical tolerance). Missing or invalid matches do
not switch. Goals within 0.30 m of the robot are left alone. After each
comparison, the next opportunity is 15 seconds later.

A switch is a single-use proposal. Search requests cancellation, retains
ownership until Nav2 reports a terminal result, then checks the saved context,
current map, localization, point history and full footprint before sending the
replacement. Cancellation acknowledgment alone does not release ownership.

An accepted ordinary goal starts its 10-second comparison delay at acceptance;
an accepted dynamic replacement starts a 15-second delay at acceptance.
Terminal SUCCEEDED and genuine ABORTED results request advancement without
dispatching in the action callback. The next existing 1-second ROS timer checks
the saved ranking, current map,
world-coordinate frontier source, full footprint, localization and history.
Only then can it send one next goal. Lower-ranked coarse points are never sent
without that check. If no saved point is still admissible, the existing single
worker computes a new ranking. There is no new scheduler or worker. BFS
filters reachable viewpoints and orders them within each cluster.
`cluster_access_distance_m` records the minimum BFS access distance, and
`goal_cost_m` records the BFS distance to the chosen footprint-safe point,
which can be farther after shifting.

The evaluator requires the entire configured rectangular footprint to intersect
only known-FREE map cells at the natural approach yaw. This map check does not
establish Ackermann reachability; Nav2 remains responsible for route planning
and costmap-based obstacle handling.

Camera configuration is not used by the pure cluster evaluator.

Occupancy value 0 is FREE, 100 is OCCUPIED, and -1 and intermediate values
1 through 99 are UNKNOWN. Only known-FREE cells are eligible. No suitcase detection
or application-state decision is included. No live ROS, TF, Nav2, or robot
runtime behavior is verified by the offline tests.

With ROS 2 Humble and the workspace install sourced, run the offline tests
from the workspace root:

    python3 -m pytest -q src/piracer_search/test src/piracer_state_manager/test src/piracer_navigation/test

The mocked SEARCH fixtures restore the real ROS module imports, so they can
share a pytest process with the native TF tests. Regression input files live in
`test/fixtures`; the tests do not require experiment logs or rosbag files.

The full application starts Search through `piracer_state_manager/launch/search_state_manager.launch.py`.
