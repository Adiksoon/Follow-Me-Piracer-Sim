# piracer_state_manager

ROS 2 state publisher consuming target-filtered suitcase Detection2DArray messages. It publishes SEARCH, ACQUIRE, or FOLLOW as the requested behavior, plus `/piracer/state/run`, a Bool indicating that the manager is enabled and the detector stream is fresh. It starts false and becomes equal to the manager's enabled state after a valid, strictly newer detector message; it does not indicate that a target was detected. Operation starts automatically in SEARCH after the first valid detector result.

The state manager binds Search through its SetBool service: Search is requested enabled exactly when the internal active state is true and state is SEARCH. It does not bind Follow, call Nav2, publish movement or gate commands, or verify a physical stop. The `/piracer/state/run` output reflects manual enable AND a fresh detector stream.

Run `ros2 launch piracer_state_manager search_state_manager.launch.py` to co-launch Search and the state manager. The `search_enable_service` launch argument configures the shared endpoint and defaults to `/search/enable`. `ros2 launch piracer_state_manager state_manager.launch.py` starts the manager alone. Other startup parameters are in `config/state_manager.yaml`. The `/piracer/state/enable` SetBool service accepts false to latch manual stop and true to restart in SEARCH.
