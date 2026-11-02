from piracer_state_manager.state_machine import (
    ACQUIRE,
    FOLLOW,
    SEARCH,
    BehaviorEnableCoordinator,
    confirmed_command_owner,
)


def test_command_owner_is_stop_until_exclusive_service_confirmation():
    coordinator = BehaviorEnableCoordinator(("search", "acquire", "follow"))
    assert confirmed_command_owner(SEARCH, True, coordinator.confirmed,
                                  coordinator.pending) == "stop"

    assert coordinator.begin("search", False)
    assert confirmed_command_owner(SEARCH, True, coordinator.confirmed,
                                  coordinator.pending) == "stop"
    assert coordinator.complete("search", False, True)
    assert coordinator.begin("acquire", False)
    assert coordinator.complete("acquire", False, True)
    assert coordinator.begin("follow", False)
    assert coordinator.complete("follow", False, True)
    assert coordinator.begin("search", True)
    assert confirmed_command_owner(SEARCH, True, coordinator.confirmed,
                                  coordinator.pending) == "stop"
    assert coordinator.complete("search", True, True)
    assert confirmed_command_owner(SEARCH, True, coordinator.confirmed,
                                  coordinator.pending) == "search"


def test_owner_stops_on_manual_disable_failure_or_unactuated_acquire():
    confirmed = {"search": False, "acquire": False, "follow": True}
    assert confirmed_command_owner(FOLLOW, True, confirmed, None) == "follow"
    assert confirmed_command_owner(FOLLOW, True, confirmed, ("follow", False)) == "stop"
    assert confirmed_command_owner(FOLLOW, False, confirmed, None) == "stop"
    assert confirmed_command_owner(ACQUIRE, True, confirmed, None) == "stop"
    assert confirmed_command_owner(
        FOLLOW, True, {"search": None, "acquire": False, "follow": True}, None
    ) == "stop"
