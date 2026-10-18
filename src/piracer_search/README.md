# piracer_search

A pure-Python frontier candidate and ranking core. It has no ROS client or
experimental fixture dependency. The evaluate_search(grid, robot_pose) function
accepts an OccupancyGrid-shaped mapping with width, height, resolution, origin
[x, y, yaw], and row-major data values 0, 100, or -1. The pose is world x, y,
yaw. The world_to_cell helper handles a rotated map origin.

The core identifies FREE cells beside UNKNOWN, generates candidate approach
poses, rejects footprints touching UNKNOWN, OCCUPIED, or map edges, and finds
cardinal shortest paths over safe FREE cells. Optional camera settings count
UNKNOWN cells inside the configured HFOV/range, with occupied-cell ray
occlusion. Optional freshness history counts visible FREE cells whose recorded
last-seen age reaches ttl_s. Ranking normalizes combined visible gain and
path distance and applies gain_weight (default 0.5).

This is a compact runtime adaptation of the offline CP1 geometry and SCORE
calculations. It omits the offline fixture's explicit path table and terminal
feasibility flags: runtime paths are computed from FREE occupancy cells, and
no Nav2 planner/action feasibility is asserted here. The score combines
unknown and stale-free cell counts directly; it does not yet reproduce the
offline evaluator's separately normalized strategy variants. No ROS node,
Nav2 client, or START/STOP state-machine interface is included.

Run the standard-library tests from the package directory:

    PYTHONPATH=. python3 -m unittest discover -s test -v
