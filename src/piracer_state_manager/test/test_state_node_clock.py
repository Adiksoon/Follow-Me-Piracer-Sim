"""Clock-domain regression tests without a running ROS graph."""
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from piracer_state_manager.state_machine import SEARCH, StateMachine
from piracer_state_manager.state_node import StateManagerNode


def node(sim=True):
    clock = SimpleNamespace(nanoseconds=0)
    result = SimpleNamespace(
        machine=StateMachine(), _last_sim_time_ns=None, _sim_epoch_start_ns=0,
        get_parameter=lambda name: SimpleNamespace(value=sim),
        get_clock=lambda: SimpleNamespace(now=lambda: clock),
        get_logger=Mock(return_value=Mock()), _log_events=Mock(),
        _publish_changes=Mock(), _sync_behavior_services=Mock(),
    )
    for name in ("_freshness_now", "_on_detections", "_on_timer", "_on_enable"):
        setattr(result, name, MethodType(getattr(StateManagerNode, name), result))
    return result, clock


def detection(stamp_ns):
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)), detections=[])


def test_sim_slowdown_does_not_spend_wall_time_budget_but_sim_deadline_still_stops(monkeypatch):
    n, clock = node(); wall = [100.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: wall[0])
    clock.nanoseconds = 8_000_000_000; n._on_detections(detection(7_900_000_000))
    wall[0] += 2.4; clock.nanoseconds = 8_100_000_000
    n._on_detections(detection(7_900_000_000)); n._on_timer()
    assert n.machine.active and n.machine.last_receipt_time == 8.0
    clock.nanoseconds = 10_000_000_000; n._on_timer(); n._on_timer()
    assert not n.machine.active
    assert sum("NO_DATA" in call.args[0] for call in n._log_events.call_args_list) == 1


def test_physical_mode_keeps_monotonic_two_second_timeout(monkeypatch):
    n, _ = node(False); wall = [100.0]
    monkeypatch.setattr("piracer_state_manager.state_node.time.monotonic", lambda: wall[0])
    n._on_detections(detection(1_000_000_000)); assert n.machine.active
    wall[0] = 101.99; n._on_timer(); assert n.machine.active
    wall[0] = 102; n._on_timer(); assert not n.machine.active


def test_sim_zero_and_future_header_do_not_activate():
    n, clock = node()
    n._on_detections(detection(1_000_000_000)); assert not n.machine.active
    n._on_enable(SimpleNamespace(data=True), SimpleNamespace())
    assert not n.machine.active
    clock.nanoseconds = 1_000_000_000
    n._on_detections(detection(1_100_000_000)); assert not n.machine.active


def test_clock_regression_drops_current_and_old_epoch_data_then_accepts_new_epoch():
    n, clock = node(); clock.nanoseconds = 10_000_000_000
    n._on_detections(detection(9_900_000_000)); assert n.machine.active
    clock.nanoseconds = 2_000_000_000; n._on_detections(detection(1_900_000_000))
    assert not n.machine.active and n.machine.state == SEARCH
    assert n.machine.last_receipt_time is None and n.machine.last_stamp is None
    assert n.machine.data_paused and not n.machine.window
    clock.nanoseconds = 2_100_000_000
    n._on_detections(detection(9_900_000_000)); assert not n.machine.active
    n._on_detections(detection(1_900_000_000)); assert not n.machine.active
    n._on_detections(detection(2_050_000_000)); assert n.machine.active
    receipt = n.machine.last_receipt_time
    n._on_detections(detection(2_040_000_000)); assert n.machine.last_receipt_time == receipt


def test_manual_stop_survives_clock_regression_and_fresh_data():
    n, clock = node(); clock.nanoseconds = 10_000_000_000
    n._on_detections(detection(9_900_000_000))
    n._on_enable(SimpleNamespace(data=False), SimpleNamespace())
    clock.nanoseconds = 2_000_000_000; n._on_timer()
    clock.nanoseconds = 2_100_000_000; n._on_detections(detection(2_050_000_000))
    assert not n.machine.enabled and not n.machine.active


def test_launch_forwards_sim_time_without_changing_hardware_default():
    package = Path(__file__).parents[1]
    manager = (package / "launch/state_manager.launch.py").read_text()
    search = (package / "launch/search_state_manager.launch.py").read_text()
    assert 'DeclareLaunchArgument("use_sim_time", default_value="false")' in manager
    assert '"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)' in manager
    assert '"use_sim_time": LaunchConfiguration("use_sim_time")' in search
    assert '"follow_use_sim_time": LaunchConfiguration("use_sim_time")' in search
