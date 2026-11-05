"""Handoff diagnostics describe waiting without altering service decisions."""
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock
import pytest

from piracer_state_manager.state_node import StateManagerNode
from piracer_state_manager.state_machine import BehaviorEnableCoordinator, FOLLOW, SEARCH


def manager(ready=True, state=SEARCH):
    node = SimpleNamespace(
        machine=SimpleNamespace(active=True, state=state),
        behavior_coordinator=BehaviorEnableCoordinator(("search", "acquire", "follow")),
        _handoff_started_monotonic=None, _handoff_future=None,
        _handoff_wait_logged=False, _blocking_service_logged=None,
        _handoff_failure_logged=False, _behavior_enable_dispatched=False,
        get_logger=Mock(return_value=Mock()),
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


def test_follow_target_disables_search_and_acquire_before_enabling_follow():
    node = manager(state=FOLLOW)
    node._sync_behavior_services()
    observed = []
    for _ in range(4):
        observed.append(node.behavior_coordinator.pending)
        node._handoff_future.set_result(SimpleNamespace(success=True, message="ok"))

    assert observed == [
        ("search", False),
        ("acquire", False),
        ("follow", False),
        ("follow", True),
    ]
    assert node.behavior_coordinator.pending is None
    assert node.behavior_coordinator.confirmed == {
        "search": False, "acquire": False, "follow": True
    }


def test_unconfirmed_search_disable_blocks_follow_enable():
    node = manager(state=FOLLOW)
    node._sync_behavior_services()
    node._handoff_future.set_result(
        SimpleNamespace(success=False, message="Nav2 goal still owned")
    )

    assert node.behavior_coordinator.confirmed == {
        "search": None, "acquire": None, "follow": None
    }
    assert node.behavior_clients["acquire"].call_async.call_count == 0
    assert node.behavior_clients["follow"].call_async.call_count == 0

    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == ("search", False)
    assert node.behavior_clients["search"].call_async.call_count == 2
    assert node.behavior_clients["follow"].call_async.call_count == 0


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


def _node_waiting_on_initial_follow_disable(monkeypatch, now):
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager(state=FOLLOW)
    node.get_parameter.return_value.value = True
    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == ("search", False)
    node._handoff_future.set_result(SimpleNamespace(success=True, message="search stopped"))
    assert node.behavior_coordinator.pending == ("acquire", False)
    node._handoff_future.set_result(SimpleNamespace(success=True, message="acquire stopped"))
    assert node.behavior_coordinator.pending == ("follow", False)
    assert node.behavior_coordinator.confirmed == {
        "search": False, "acquire": False, "follow": None
    }
    return node


def test_simulation_initial_follow_disable_retries_only_after_five_seconds(monkeypatch):
    now = [10.0]
    node = _node_waiting_on_initial_follow_disable(monkeypatch, now)
    old_future = node._handoff_future
    now[0] = 14.999
    node._sync_behavior_services()
    assert node.behavior_clients["follow"].call_async.call_count == 1
    now[0] = 15.0
    node._sync_behavior_services()
    assert node.behavior_clients["follow"].call_async.call_count == 1
    node._sync_behavior_services()
    assert node.behavior_clients["follow"].call_async.call_count == 2
    assert node.behavior_clients["follow"].remove_pending_request.call_args.args == (old_future,)
    assert old_future.cancelled()
    assert node.behavior_coordinator.pending == ("follow", False)
    assert node.behavior_coordinator.confirmed == {
        "search": False, "acquire": False, "follow": None
    }
    assert node._handoff_future is not old_future
    node.behavior_clients["search"].remove_pending_request.assert_not_called()
    node.behavior_clients["acquire"].remove_pending_request.assert_not_called()


@pytest.mark.parametrize("late_callback", ["success", "failure", "cancelled"])
def test_late_initial_follow_disable_callback_cannot_change_retry(late_callback, monkeypatch):
    now = [0.0]
    node = _node_waiting_on_initial_follow_disable(monkeypatch, now)
    old_future = node._handoff_future
    now[0] = 5.0
    node._sync_behavior_services()
    node._sync_behavior_services()
    replacement = node._handoff_future
    assert old_future.cancelled()
    stale_future = {"success": object(), "failure": object(), "cancelled": old_future}[late_callback]
    before = node.behavior_coordinator.confirmed.copy()
    node._on_behavior_enable_result("follow", False, stale_future)
    assert node._handoff_future is replacement
    assert node.behavior_coordinator.pending == ("follow", False)
    assert node.behavior_coordinator.confirmed == before == {
        "search": False, "acquire": False, "follow": None
    }


def test_repeated_follow_disable_timeouts_and_rejection_never_enable(monkeypatch):
    now = [0.0]
    node = _node_waiting_on_initial_follow_disable(monkeypatch, now)
    total_follow_calls = 1
    for deadline in (5.0, 10.0, 15.0):
        now[0] = deadline
        node._sync_behavior_services()
        assert node.behavior_clients["follow"].call_async.call_count == total_follow_calls
        node._sync_behavior_services()
        total_follow_calls += 1
        assert node.behavior_coordinator.pending == ("follow", False)
        assert node.behavior_coordinator.confirmed == {
            "search": False, "acquire": False, "follow": None
        }
    assert node.behavior_clients["follow"].call_async.call_count == total_follow_calls
    assert node.behavior_clients["follow"].call_async.call_args.args[0].data is False

    node._handoff_future.set_result(SimpleNamespace(success=False, message="rejected"))
    assert node.behavior_coordinator.pending is None
    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == ("follow", False)
    assert node.behavior_coordinator.confirmed == {
        "search": False, "acquire": False, "follow": None
    }
    assert node.behavior_clients["follow"].call_async.call_args.args[0].data is False


def test_genuine_follow_disable_retry_success_allows_normal_enable(monkeypatch):
    now = [0.0]
    node = _node_waiting_on_initial_follow_disable(monkeypatch, now)
    first = node._handoff_future
    now[0] = 5.0
    node._sync_behavior_services()
    node._sync_behavior_services()
    replacement = node._handoff_future
    assert first.cancelled()
    replacement.set_result(SimpleNamespace(success=True, message="Follow disabled"))
    assert node.behavior_coordinator.pending == ("follow", True)
    assert node.behavior_coordinator.confirmed == {
        "search": False, "acquire": False, "follow": False
    }
    assert node.behavior_clients["follow"].call_async.call_args.args[0].data is True
    assert node._behavior_enable_dispatched is True


@pytest.mark.parametrize("confirmed", [
    {"search": None, "acquire": False, "follow": None},
    {"search": False, "acquire": None, "follow": None},
    {"search": True, "acquire": False, "follow": None},
])
def test_follow_disable_timeout_requires_confirmed_initial_search_and_acquire(
        monkeypatch, confirmed):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager(state=FOLLOW)
    node.get_parameter.return_value.value = True
    node.behavior_coordinator.confirmed = confirmed.copy()
    node.behavior_coordinator.pending = ("follow", False)
    node._handoff_started_monotonic = 0.0
    node._handoff_future = Future()
    now[0] = 5.0
    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == ("follow", False)
    assert not any(client.remove_pending_request.called
                   for client in node.behavior_clients.values())


def test_failed_enable_dispatch_latches_lifetime_guard_before_client_call():
    node = manager(state=SEARCH)
    node.behavior_coordinator.confirmed = {
        "search": False, "acquire": False, "follow": False
    }
    node.behavior_clients["search"].call_async.side_effect = RuntimeError("dispatch failed")
    with pytest.raises(RuntimeError, match="dispatch failed"):
        node._sync_behavior_services()
    assert node.behavior_coordinator.pending == ("search", True)
    assert node._behavior_enable_dispatched is True


def test_enable_dispatch_guard_survives_manual_stop_and_reenable():
    node = manager()
    node._behavior_enable_dispatched = True
    node.machine.set_enabled = Mock(return_value=[])
    node._acquire_results_pending = {}
    node._freshness_now = Mock(return_value=1.0)
    node._log_events = Mock()
    node._publish_changes = Mock()
    node._sync_behavior_services = Mock()
    response = SimpleNamespace()
    StateManagerNode._on_enable(node, SimpleNamespace(data=False), response)
    assert response.success is True
    assert node._behavior_enable_dispatched is True
    StateManagerNode._on_enable(node, SimpleNamespace(data=True), response)
    assert response.success is True
    assert node._behavior_enable_dispatched is True


@pytest.mark.parametrize("pending,confirmed", [
    (("search", False), {"search": None, "acquire": None, "follow": None}),
    (("follow", False), {"search": False, "acquire": False, "follow": None}),
    (("search", False), {"search": False, "acquire": False, "follow": False}),
])
def test_no_initial_disable_timeout_retry_after_any_enable_dispatch(
        monkeypatch, pending, confirmed):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    node = manager()
    node.get_parameter.return_value.value = True
    node.behavior_coordinator.confirmed = confirmed.copy()
    node.behavior_coordinator.pending = pending
    node._handoff_started_monotonic = 0.0
    node._handoff_future = Future()
    node._behavior_enable_dispatched = True
    now[0] = 5.0
    node._sync_behavior_services()
    assert node.behavior_coordinator.pending == pending
    assert not any(client.remove_pending_request.called
                   for client in node.behavior_clients.values())


def test_initial_follow_disable_timeout_is_not_retried_in_hardware_or_for_done_future(
        monkeypatch):
    now = [0.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: now[0])
    hardware = _node_waiting_on_initial_follow_disable(monkeypatch, now)
    hardware.get_parameter.return_value.value = False
    hardware_future = hardware._handoff_future
    now[0] = 5.0
    hardware._sync_behavior_services()
    assert hardware.behavior_coordinator.pending == ("follow", False)
    assert hardware.behavior_clients["follow"].remove_pending_request.call_count == 0
    assert not hardware_future.done()

    now[0] = 10.0
    completed = _node_waiting_on_initial_follow_disable(monkeypatch, now)
    completed_future = completed._handoff_future
    completed_future.set_result(SimpleNamespace(success=True, message="Follow disabled"))
    assert completed.behavior_coordinator.pending == ("follow", True)
    assert completed._behavior_enable_dispatched is True
    assert not completed.behavior_clients["follow"].remove_pending_request.called


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
