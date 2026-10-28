# piracer_search

A small ROS 2 package with a pure-Python frontier evaluator and one Search
node. It groups cardinally detected frontier cells into deterministic 8-neighbor
clusters and ranks eligible clusters with `0.5 * G/Gmax - 0.5 * D/Dmax`,
where G is boundary-cell count and D is the minimum safe-cell BFS access
distance. D is a grid access estimate, not a Nav2 route length. Candidate
poses receive a full known-FREE footprint check at their approach yaw, with
nearby alternatives limited to 0.30 m. Nav2 plans the route.

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
from the challengers. A different ready, footprint-checked cluster must exceed
the active score by the configured margin. Missing or invalid matches do not
switch. Goals within 0.30 m of the robot are left alone. After each comparison,
the next opportunity is 15 seconds later.

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
worker computes a new ranking. There is no new scheduler or worker. Cluster
access distance is the score input; `goal_cost_m` separately records the BFS
cost of the chosen footprint-safe point, which can be farther after shifting.

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
