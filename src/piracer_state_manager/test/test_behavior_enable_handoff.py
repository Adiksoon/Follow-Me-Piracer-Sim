from piracer_state_manager.state_machine import BehaviorEnableCoordinator


def _complete(coordinator, target, services_ready):
    request = coordinator.next_request(target, services_ready)
    assert request is not None
    assert coordinator.begin(*request)
    assert coordinator.complete(*request, True)
    return request


def test_search_is_enabled_only_after_acquire_and_follow_are_disabled():
    names = ("search", "acquire", "follow")
    coordinator = BehaviorEnableCoordinator(names)
    ready = {name: True for name in names}

    assert _complete(coordinator, "search", ready) == ("search", False)
    assert _complete(coordinator, "search", ready) == ("acquire", False)
    assert _complete(coordinator, "search", ready) == ("follow", False)
    assert _complete(coordinator, "search", ready) == ("search", True)
    assert coordinator.next_request("search", ready) is None
