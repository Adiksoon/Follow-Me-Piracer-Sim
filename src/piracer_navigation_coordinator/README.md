# piracer_navigation_coordinator

Provides `/navigation_coordinator/search` and `/navigation_coordinator/follow` as `NavigateToPose` action servers. Both forward through one downstream client to `/navigate_to_pose`. A new upstream goal is rejected while either owner is active; there is no queue or automatic preemption. A request is also rejected when Nav2 is not ready.

Feedback and terminal result values are forwarded from Nav2. Upstream cancellation requests cancellation downstream, including when cancellation arrives before Nav2 returns its goal handle. The owner remains reserved until a terminal downstream result arrives. A downstream rejection releases the reservation because Nav2 confirmed that it did not accept a goal. An uncertain submission or result transport failure keeps the reservation to avoid accepting a second owner while the first downstream state is unknown.

If an upstream cancel was accepted, the upstream goal ends as canceled after Nav2 returns its terminal result, even if Nav2 completed successfully or aborted during the cancellation race; Nav2's result payload is still returned.

If forwarding or result retrieval fails after an upstream cancel was accepted, the upstream handle is also completed as canceled to respect its `CANCELING` state; ownership stays reserved when the downstream outcome is uncertain.

Run `ros2 launch piracer_navigation_coordinator navigation_coordinator.launch.py` to compose the existing Follow/Nav2/simulation launch, the Search/state-manager launch, and this coordinator. Launch-scoped `navigate_to_pose_action` parameters route Search and Follow to their separate coordinator endpoints. Follow navigation remains disabled by default through `enable_follow_navigation:=false`.

This package does not publish `/cmd_vel` or modify the command arbiter.
