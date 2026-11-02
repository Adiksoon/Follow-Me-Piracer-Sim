from types import SimpleNamespace

import pytest

from piracer_state_manager.state_machine import (
    ACQUIRE,
    FOLLOW,
    SEARCH,
    BehaviorEnableCoordinator,
)
from piracer_state_manager.state_node import StateManagerNode


class _Client:
    def __init__(self, name, ready=True):
        self.srv_name = f'/{name}/enable'
        self.ready = ready

    def service_is_ready(self):
        return self.ready


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class _StateNodeHarness:
    """Minimal fields consumed by the real StateManagerNode method."""

    def __init__(self, state, active, confirmed, pending=None, services_ready=True):
        self.machine = SimpleNamespace(state=state, active=active)
        self.behavior_coordinator = BehaviorEnableCoordinator(
            ('search', 'acquire', 'follow'))
        self.behavior_coordinator.confirmed.update(confirmed)
        self.behavior_coordinator.pending = pending
        self.behavior_clients = {
            name: _Client(name, services_ready)
            for name in ('search', 'acquire', 'follow')}
        self._handoff_started_monotonic = None
        self._handoff_future = None
        self._handoff_wait_logged = False
        self._blocking_service_logged = None
        self._behavior_enable_dispatched = False
        self.owners = []

    def _publish_command_owner(self, owner):
        self.owners.append(owner)

    def get_logger(self):
        return _Logger()


def _owner_through_state_node(state, active, confirmed, pending=None, services_ready=True):
    node = _StateNodeHarness(state, active, confirmed, pending, services_ready)
    StateManagerNode._sync_behavior_services(node)
    assert len(node.owners) == 1
    return node.owners[0]


def test_real_state_node_sync_publishes_lowercase_follow_and_search_owners():
    assert _owner_through_state_node(
        FOLLOW, True,
        {'search': False, 'acquire': False, 'follow': True}) == 'follow'
    assert _owner_through_state_node(
        SEARCH, True,
        {'search': True, 'acquire': False, 'follow': False}) == 'search'


@pytest.mark.parametrize('state,active,confirmed,pending,services_ready', [
    (FOLLOW, False, {'search': False, 'acquire': False, 'follow': False}, None, True),
    (FOLLOW, True, {'search': False, 'acquire': False, 'follow': True}, ('follow', True), True),
    (FOLLOW, True, {'search': False, 'acquire': False, 'follow': None}, ('follow', False), True),
    (ACQUIRE, True, {'search': False, 'acquire': True, 'follow': False}, None, True),
    (FOLLOW, True, {'search': None, 'acquire': False, 'follow': True}, None, False),
])
def test_real_state_node_sync_fails_closed_without_exclusive_confirmed_owner(
        state, active, confirmed, pending, services_ready):
    assert _owner_through_state_node(
        state, active, confirmed, pending, services_ready) == 'stop'
