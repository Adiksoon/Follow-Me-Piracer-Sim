# piracer_state_manager

Standalone ROS 2 state publisher consuming target-filtered suitcase Detection2DArray messages. It publishes SEARCH, ACQUIRE, or FOLLOW as the requested behavior, plus whether the detector stream is active. Operation starts automatically in SEARCH after the first valid detector result.

This package only labels behavior. It does not start or cancel Search or Follow, call Nav2, publish movement or gate commands, or verify a physical stop. A later coordinator must consume both outputs and handle movement ownership and cancellation.

Run: ros2 launch piracer_state_manager state_manager.launch.py. Startup parameters are in config/state_manager.yaml. The /piracer/state/enable SetBool service accepts false to latch manual stop and true to restart in SEARCH.
