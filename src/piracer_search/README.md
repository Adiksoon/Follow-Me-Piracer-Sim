# piracer_search

This package owns the Search lifecycle and its single `/navigate_to_pose`
action client. The node starts disabled and the state manager controls it via
`/search/enable` (`std_srvs/SetBool`). It consumes the prepared `/nav2/map`
and checks localization freshness using the existing TF and OdomInfo contract.

Frontier detection, clustering, ranking and centroid selection are provided by
the vendored `FrontierSearch` implementation from
[robo-friends/m-explore-ros2 at revision 326cf8a0b487c34246bb8f3326afbcd69576dc60](https://github.com/robo-friends/m-explore-ros2/tree/326cf8a0b487c34246bb8f3326afbcd69576dc60/explore).
The adapter uses `potential_scale=3.0`, `gain_scale=1.0`, sends the selected
cluster centroid in the map frame with yaw 0, and does not add a footprint
search or local path planner. Original upstream source notices and BSD license
are retained in `piracer_explore_core/third_party/m-explore-ros2`.

Search owns only the goal it submitted. Disable or a qualifying goal switch
stops new submissions and requests cancellation of that goal only. Search
retains ownership until the action result is terminal. Goal replacement uses a
fresh upstream ranking after the previous result; the 0.10 switch margin is
compared on one ranking update. Localization loss prevents further goal
submission and retains the existing stop behavior.

The offline regression suite can be run from the workspace root:

    python3 -m pytest -q src/piracer_search/test src/piracer_state_manager/test src/piracer_navigation/test

No live Nav2, Gazebo, RViz, or robot behavior is established by offline tests.
