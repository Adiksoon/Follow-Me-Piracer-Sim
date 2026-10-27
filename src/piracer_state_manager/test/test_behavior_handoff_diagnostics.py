"""Handoff diagnostics describe waiting without altering service decisions."""
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock
import pytest

from piracer_state_manager.state_node import StateManagerNode
from piracer_state_manager.state_machine import BehaviorEnableCoordinator, SEARCH


def manager(ready=True):
    node = SimpleNamespace(
        machine=SimpleNamespace(active=True, state=SEARCH),
        behavior_coordinator=BehaviorEnableCoordinator(("search", "acquire", "follow")),
        _handoff_started_monotonic=None, _handoff_future=None,
        _handoff_wait_logged=False, _blocking_service_logged=None,
        _handoff_failure_logged=False, get_logger=Mock(return_value=Mock()),
        get_parameter=Mock(return_value=SimpleNamespace(value=False)),
    )
    node.behavior_clients = {}
    for name in ("search", "acquire", "follow"):
        node.behavior_clients[name] = SimpleNamespace(
            srv_name=f"/{name}/enable", service_is_ready=Mock(return_value=ready),
            call_async=Mock(side_effect=lambda request: Future()),
            remove_pending_request=Mock(),
        )
    node._sync_behavior_services = MethodType(StateManagerNode._sync_behavior_services, node)
    node._on_behavior_enable_result = MethodType(StateManagerNode._on_behavior_enable_result, node)
    return node


def test_unavailable_service_logged_once_without_sending():
    node = manager(False)
    for _ in range(3):
        node._sync_behavior_services()
    node.get_logger().info.assert_called_once_with("Behavior waiting for service: /search/enable")
    assert node.behavior_coordinator.pending is None
    assert all(not c.call_async.called for c in node.behavior_clients.values())


def test_pending_future_logged_once_after_five_seconds(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node._sync_behavior_services()
    now[0] = 14.9; node._sync_behavior_services()
    node.get_logger().warning.assert_not_called()
    now[0] = 15.0; node._sync_behavior_services(); node._sync_behavior_services()
    node.get_logger().warning.assert_called_once()
    assert "future_done=False" in node.get_logger().warning.call_args.args[0]
    assert node.behavior_coordinator.pending == ("search", False)
    assert node.behavior_clients["search"].call_async.call_count == 1


def test_completion_preserves_disable_before_enable_order():
    node = manager()
    node._sync_behavior_services()
    observed = []
    for _ in range(4):
        observed.append(node.behavior_coordinator.pending)
        node._handoff_future.set_result(SimpleNamespace(success=True, message="ok"))
    assert observed == [("search", False), ("acquire", False), ("follow", False), ("search", True)]
    assert node.behavior_coordinator.pending is None
    assert node.behavior_coordinator.confirmed == {"search": True, "acquire": False, "follow": False}


def test_synchronous_dispatch_error_is_logged_and_still_raised():
    node = manager()
    node.behavior_clients["search"].call_async.side_effect = RuntimeError("synthetic")
    with pytest.raises(RuntimeError, match="synthetic"):
        node._sync_behavior_services()
    node.get_logger().error.assert_called_once()
    assert node.behavior_coordinator.pending == ("search", False)


def test_simulation_initial_search_disable_retries_at_five_seconds(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node._sync_behavior_services()
    old_future = node._handoff_future
    now[0] = 14.9
    node._sync_behavior_services()
    assert node.behavior_clients["search"].call_async.call_count == 1
    now[0] = 15.0
    node._sync_behavior_services()
    assert node.behavior_clients["search"].call_async.call_count == 1
    node._sync_behavior_services()
    assert node.behavior_clients["search"].call_async.call_count == 2
    assert node.behavior_clients["search"].call_async.call_args.args[0].data is False
    assert node.behavior_clients["search"].remove_pending_request.call_args.args == (old_future,)
    assert old_future.cancelled()
    assert node.behavior_coordinator.pending == ("search", False)
    assert node.behavior_coordinator.confirmed == {"search": None, "acquire": None, "follow": None}


@pytest.mark.parametrize("confirmed", [
    {"search": True, "acquire": None, "follow": None},
    {"search": None, "acquire": False, "follow": None},
])
def test_simulation_noninitial_search_disable_is_not_retried(monkeypatch, confirmed):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node.behavior_coordinator.confirmed = confirmed.copy()
    node.behavior_coordinator.pending = ("search", False)
    node._handoff_started_monotonic = 0.0
    node._handoff_future = Future()
    now[0] = 5.0
    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == ("search", False)
    assert not node.behavior_clients["search"].remove_pending_request.called


def test_cancelled_old_future_callback_after_retry_cannot_change_new_request(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node._sync_behavior_services()
    old_future = node._handoff_future
    now[0] = 5.0
    node._sync_behavior_services()
    node._sync_behavior_services()
    new_future = node._handoff_future
    assert old_future.cancelled()
    assert new_future is not old_future
    before = node.behavior_coordinator.confirmed.copy()
    node._on_behavior_enable_result("search", False, old_future)
    assert node._handoff_future is new_future
    assert node.behavior_coordinator.pending == ("search", False)
    assert node.behavior_coordinator.confirmed == before


def test_stale_callback_is_ignored_before_coordinator_mutation():
    node = manager()
    node._sync_behavior_services()
    current = node._handoff_future
    stale = Future()
    stale.set_result(SimpleNamespace(success=True, message="late"))
    node._on_behavior_enable_result("search", False, stale)
    assert node.behavior_coordinator.pending == ("search", False)
    assert node.behavior_coordinator.confirmed == {"search": None, "acquire": None, "follow": None}
    assert node._handoff_future is current


def test_simulation_repeated_retries_wait_five_seconds_each(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node._sync_behavior_services()
    for deadline in (5.0, 10.0, 15.0):
        now[0] = deadline - 0.1
        node._sync_behavior_services()
        assert node.behavior_clients["search"].call_async.call_count == int(deadline / 5)
        now[0] = deadline
        node._sync_behavior_services()
        assert node.behavior_clients["search"].call_async.call_count == int(deadline / 5)
        node._sync_behavior_services()
        assert node.behavior_clients["search"].call_async.call_count == int(deadline / 5) + 1
        assert node.behavior_coordinator.pending == ("search", False)


def test_retry_preserves_acknowledged_disable_before_enable_sequence(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node._sync_behavior_services()
    now[0] = 5.0
    node._sync_behavior_services()
    node._sync_behavior_services()
    observed = []
    for _ in range(4):
        observed.append(node.behavior_coordinator.pending)
        node._handoff_future.set_result(SimpleNamespace(success=True, message="ok"))
    assert observed == [("search", False), ("acquire", False), ("follow", False), ("search", True)]


def test_hardware_initial_timeout_keeps_existing_pending_request(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node._sync_behavior_services()
    now[0] = 5.0
    node._sync_behavior_services()
    assert node.behavior_clients["search"].call_async.call_count == 1
    assert node.behavior_coordinator.pending == ("search", False)


@pytest.mark.parametrize("pending", [("search", True), ("acquire", False), ("follow", False)])
def test_simulation_other_pending_requests_are_not_retried(monkeypatch, pending):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node.behavior_coordinator.pending = pending
    node._handoff_started_monotonic = 0.0
    node._handoff_future = Future()
    now[0] = 5.0
    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == pending
    assert not any(client.remove_pending_request.called for client in node.behavior_clients.values())


def test_completed_future_at_timeout_boundary_is_not_discarded(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node._sync_behavior_services()
    future = node._handoff_future
    future.set_result(SimpleNamespace(success=True, message="ok"))
    now[0] = 5.0
    node._sync_behavior_services()
    assert node.behavior_coordinator.confirmed["search"] is False
    assert not node.behavior_clients["search"].remove_pending_request.called
