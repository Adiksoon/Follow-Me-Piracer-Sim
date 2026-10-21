# piracer_state_manager

ROS 2 state publisher consuming target-filtered suitcase Detection2DArray messages. It publishes SEARCH, ACQUIRE, or FOLLOW as the requested behavior, plus whether the detector stream is active. Operation starts automatically in SEARCH after the first valid detector result.

The state manager binds Search through its SetBool service: Search is requested enabled exactly when active is true and state is SEARCH. It does not bind Follow, call Nav2, publish movement or gate commands, or verify a physical stop. The active output keeps its meaning as manual enable AND a fresh detector stream.

Run `ros2 launch piracer_state_manager search_state_manager.launch.py` to co-launch Search and the state manager. The `search_enable_service` launch argument configures the shared endpoint and defaults to `/search/enable`. `ros2 launch piracer_state_manager state_manager.launch.py` starts the manager alone. Other startup parameters are in `config/state_manager.yaml`. The `/piracer/state/enable` SetBool service accepts false to latch manual stop and true to restart in SEARCH.
