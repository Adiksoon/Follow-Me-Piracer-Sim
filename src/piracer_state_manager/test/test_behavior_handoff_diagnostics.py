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
    )
    node.behavior_clients = {}
    for name in ("search", "acquire", "follow"):
        node.behavior_clients[name] = SimpleNamespace(
            srv_name=f"/{name}/enable", service_is_ready=Mock(return_value=ready),
            call_async=Mock(side_effect=lambda request: Future()),
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
