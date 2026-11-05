"""State Manager and Acquire callback integration without a running ROS graph."""
from concurrent.futures import Future
import json
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from action_msgs.msg import GoalStatus
from std_msgs.msg import String

from piracer_state_manager.state_machine import ACQUIRE, FOLLOW, StateMachine, BehaviorEnableCoordinator
from piracer_state_manager.state_node import StateManagerNode
from test_acquire_path import acquire
from test_acquire_sequence import dispatch_leg, finish_leg
from test_state_machine import frame


def manager(acquire):
    result = SimpleNamespace(
        machine=StateMachine(),
        behavior_coordinator=BehaviorEnableCoordinator(("search", "acquire", "follow")),
        _handoff_started_monotonic=None, _handoff_future=None,
        _handoff_wait_logged=False, _blocking_service_logged=None, _handoff_failure_logged=False,
        _behavior_enable_dispatched=False, owners=[],
        _acquire_attempt_id=None, _acquire_results_pending={},
        _publish_changes=Mock(), _log_events=Mock(), get_logger=Mock(return_value=Mock()),
        get_parameter=Mock(return_value=SimpleNamespace(value=False)),
        enabled={"search": False, "follow": False}, requests=[],
    )
    result._publish_command_owner = result.owners.append
    for name in ("_sync_behavior_services", "_on_behavior_enable_result", "_on_acquire_result"):
        setattr(result, name, MethodType(getattr(StateManagerNode, name), result))

    def service(name, request):
        result.requests.append((name, request.data))
        if name == "acquire":
            if request.data:
                assert not any(result.enabled.values())
            response = acquire._set_enabled(request, SimpleNamespace())
        else:
            if request.data:
                assert not acquire._enabled and acquire._operation is None
                assert not result.enabled["follow" if name == "search" else "search"]
            result.enabled[name] = request.data
            response = SimpleNamespace(success=True, message="ok")
        future = Future()
        future.set_result(response)
        return future

    result.behavior_clients = {
        name: SimpleNamespace(srv_name=f"/{name}/enable", service_is_ready=lambda: True,
                              call_async=lambda request, name=name: service(name, request))
        for name in ("search", "acquire", "follow")
    }
    acquire._result_pub.publish.side_effect = result._on_acquire_result
    frame(result.machine, 1, True, 100.0)
    frame(result.machine, 2, True, 100.1)
    return result


def test_completion_is_reported_then_latched_for_current_state_entry(acquire):
    m = manager(acquire)
    m._sync_behavior_services()
    assert acquire._enabled and m._acquire_attempt_id == acquire._attempt_id
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    assert m.machine.state == "SEARCH" and not m.machine.acquire_finished and m.machine.active
    assert not acquire._enabled and acquire._operation is None
    assert m.behavior_coordinator.confirmed == {"search": True, "acquire": False, "follow": False}
    for stamp in range(3, 8):
        frame(m.machine, stamp, False, 100.0 + stamp / 10)
        m._sync_behavior_services()
    acquire._tick()
    assert not acquire._enabled and len(acquire._controller.goals) == 1


def test_follow_handoff_waits_for_terminal_result_and_confirmed_acquire_stop(acquire):
    m = manager(acquire)
    for stamp in (1, 2):
        frame(m.machine, stamp, True, 100.0 + stamp / 10)
    assert m.machine.state == ACQUIRE
    m._sync_behavior_services()
    follower, goal = dispatch_leg(acquire)
    for stamp in range(3, 8):
        frame(m.machine, stamp, True, 100.0 + stamp / 10)
        m._sync_behavior_services()
    assert m.machine.state == FOLLOW
    assert follower.cancel_count == 1 and not m.enabled["follow"]
    assert m.owners[-1] == "stop"
    assert acquire._operation is not None
    finish_leg(acquire, follower, goal, GoalStatus.STATUS_CANCELED)
    m._sync_behavior_services()
    assert acquire._operation is None and not acquire._enabled
    assert m.enabled == {"search": False, "follow": True}
    assert m.owners[-1] == "follow"
    assert m.requests[-2:] == [("acquire", False), ("follow", True)]


def test_normal_completion_leaves_detection_decisions_in_state_manager(acquire):
    m = manager(acquire)
    frame(m.machine, 1, True, 100.0)
    frame(m.machine, 2, True, 100.1)
    m._sync_behavior_services()
    follower, goal = dispatch_leg(acquire)
    frame(m.machine, 3, False, 101.9)
    frame(m.machine, 4, False, 103.7)
    finish_leg(acquire, follower, goal)
    assert m.machine.state == "SEARCH" and not m.machine.acquire_finished
    assert m.enabled == {"search": True, "follow": False} and not acquire._enabled
    for stamp in range(5, 12):
        frame(m.machine, stamp, True, 104.4 + stamp / 10)
    m._sync_behavior_services()
    assert m.machine.state == FOLLOW and m.enabled["follow"]


def test_old_completion_cannot_finish_a_reenabled_attempt(acquire):
    m = manager(acquire)
    m._sync_behavior_services()
    old_id = m._acquire_attempt_id
    follower, goal = dispatch_leg(acquire)
    finish_leg(acquire, follower, goal)
    m.machine.set_enabled(True, 104.4)
    frame(m.machine, 3, True, 104.5)
    frame(m.machine, 4, True, 104.6)
    m._sync_behavior_services()
    assert m._acquire_attempt_id != old_id and acquire._enabled
    m._on_acquire_result(String(data=json.dumps({"attempt_id": old_id, "outcome": "exhausted"})))
    assert m.machine.active and not m.machine.acquire_finished and acquire._enabled


@pytest.mark.parametrize("stale", [False, True])
def test_completion_before_enable_response_is_correlated_after_ack(acquire, stale):
    m = manager(acquire)
    m._sync_behavior_services()
    token = acquire._attempt_id
    m._acquire_attempt_id = None
    m.behavior_coordinator.pending = ("acquire", True)
    future = Future()
    m._handoff_future = future
    m._on_acquire_result(String(data=json.dumps({"attempt_id": "old" if stale else token, "outcome": "exhausted"})))
    assert m.machine.active
    future.set_result(SimpleNamespace(success=True, message=f"Acquire enabled; attempt_id={token}"))
    m._on_behavior_enable_result("acquire", True, future)
    assert (m.machine.state == "SEARCH") is not stale
    assert m.machine.active


@pytest.mark.parametrize("payload", ["not-json", "[]", "{}", '{"attempt_id": 1, "outcome": "exhausted"}',
                                     '{"attempt_id": "x", "outcome": "running"}'])
def test_malformed_or_nonterminal_result_does_not_change_state(acquire, payload):
    m = manager(acquire)
    m._sync_behavior_services()
    m._on_acquire_result(String(data=payload))
    assert m.machine.active and acquire._enabled and not m.machine.acquire_finished


@pytest.mark.parametrize("valid_first", [True, False])
def test_mixed_early_results_preserve_only_ack_matching_attempt(acquire, valid_first):
    m = manager(acquire)
    m._sync_behavior_services()
    token = acquire._attempt_id
    m._acquire_attempt_id = None
    m.behavior_coordinator.pending = ("acquire", True)
    future = Future()
    m._handoff_future = future
    ids = [token, "stale-attempt"] if valid_first else ["stale-attempt", token]
    for attempt in ids:
        m._on_acquire_result(String(data=json.dumps(
            {"attempt_id": attempt, "outcome": "exhausted"})))
    assert not m.machine.acquire_finished
    future.set_result(SimpleNamespace(success=True, message=f"Acquire enabled; attempt_id={token}"))
    m._on_behavior_enable_result("acquire", True, future)
    assert m.machine.state == "SEARCH" and not acquire._enabled
    assert m._acquire_results_pending == {}
    assert m.owners[-1] == "search"


def test_obsolete_enable_ack_cannot_consume_current_pending_results(acquire):
    m = manager(acquire)
    m._sync_behavior_services()
    token = acquire._attempt_id
    current, obsolete = Future(), Future()
    m._handoff_future = current
    m.behavior_coordinator.pending = ("acquire", True)
    message = String(data=json.dumps({"attempt_id": token, "outcome": "exhausted"}))
    m._on_acquire_result(message)
    obsolete.set_result(SimpleNamespace(success=True, message="Acquire enabled; attempt_id=old"))
    m._on_behavior_enable_result("acquire", True, obsolete)
    assert not m.machine.acquire_finished and acquire._enabled
    assert m._handoff_future is current
    assert m._acquire_results_pending[token] is message


def test_rejected_enable_discards_early_results_without_finishing_entry(acquire):
    m = manager(acquire)
    m._sync_behavior_services()
    token = acquire._attempt_id
    future = Future()
    m._handoff_future = future
    m.behavior_coordinator.pending = ("acquire", True)
    m._on_acquire_result(String(data=json.dumps({"attempt_id": token, "outcome": "exhausted"})))
    future.set_result(SimpleNamespace(success=False, message="not ready"))
    m._on_behavior_enable_result("acquire", True, future)
    assert not m.machine.acquire_finished
    assert m._acquire_attempt_id is None and m._acquire_results_pending == {}
    m.behavior_clients["acquire"].service_is_ready = lambda: False
    m._sync_behavior_services()
    assert "acquire" not in m.owners[-1:]
