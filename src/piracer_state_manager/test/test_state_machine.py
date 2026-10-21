import math
import pytest

from piracer_state_manager.state_machine import (
    ACQUIRE,
    FOLLOW,
    SEARCH,
    SearchEnableBinding,
    StateMachine,
    search_should_run,
)


def frame(machine, stamp, positive, now):
    return machine.observe(stamp, positive, now)


def test_initial_search_activates_on_first_valid_result():
    machine = StateMachine()
    assert machine.state == SEARCH and not machine.active
    assert frame(machine, 1, False, 0) == []
    assert machine.active and machine.state == SEARCH


def test_rolling_search_window_and_two_initial_hits():
    machine = StateMachine(acquire_hits=2, acquire_window=3)
    frame(machine, 1, True, 0)
    frame(machine, 2, False, .1)
    assert frame(machine, 3, True, .2) == ["SEARCH_TO_ACQUIRE"]
    assert machine.state == ACQUIRE
    machine = StateMachine()
    frame(machine, 1, True, 0)
    frame(machine, 2, False, .1)
    frame(machine, 3, False, .2)
    frame(machine, 4, True, .3)
    assert machine.state == SEARCH
    assert frame(machine, 5, True, .4) == ["SEARCH_TO_ACQUIRE"]


def test_transition_entry_frame_is_not_reused():
    machine = StateMachine()
    for i in range(1, 7):
        frame(machine, i, True, i / 10)
    assert machine.state == ACQUIRE
    assert frame(machine, 7, True, .7) == ["ACQUIRE_TO_FOLLOW"]
    assert machine.state == FOLLOW  # Five new positives after entry (frames 3-7).


def test_acquire_miss_resets_streak_and_sightings_do_not_extend_deadline():
    machine = StateMachine(detector_timeout_s=20)
    frame(machine, 1, True, 0)
    frame(machine, 2, True, .1)
    frame(machine, 3, True, .2)
    frame(machine, 4, False, .3)
    frame(machine, 5, True, 1.8)
    frame(machine, 6, False, 3.3)
    frame(machine, 7, True, 4.8)
    frame(machine, 8, False, 6.3)
    frame(machine, 9, True, 7.8)
    frame(machine, 10, False, 9.3)
    assert machine.tick(10.3) == ["ACQUIRE_TIMEOUT"]
    assert machine.state == SEARCH


def test_follow_requires_consecutive_misses_and_hit_resets():
    machine = StateMachine()
    for i in range(1, 8):
        frame(machine, i, True, i / 10)
    assert machine.state == FOLLOW
    frame(machine, 8, False, .8)
    frame(machine, 9, False, .9)
    frame(machine, 10, True, 1.)
    frame(machine, 11, False, 1.1)
    frame(machine, 12, False, 1.2)
    assert machine.state == FOLLOW
    assert frame(machine, 13, False, 1.3) == ["FOLLOW_TO_ACQUIRE"]
    assert machine.state == ACQUIRE


def test_acquire_expires_at_deadline_even_with_sightings_and_misses_do_not_exit():
    machine = StateMachine(detector_timeout_s=20)
    frame(machine, 1, True, 0)
    frame(machine, 2, True, .1)
    frame(machine, 3, False, .2)
    assert machine.state == ACQUIRE
    assert frame(machine, 4, True, 1.9) == []
    assert frame(machine, 5, False, 3.5) == []
    assert frame(machine, 6, True, 5.1) == []
    assert frame(machine, 7, False, 6.7) == []
    assert frame(machine, 8, True, 8.3) == []
    events = machine.tick(10.2)
    assert events == ["ACQUIRE_TIMEOUT"]
    assert machine.state == SEARCH


def test_detector_timeout_ignores_bad_stamps_and_recovers_with_reset_stamp():
    machine = StateMachine()
    frame(machine, (10, 20), True, 0)
    assert frame(machine, (10, 20), False, 1.9) == []
    assert machine.last_receipt_time == 0
    assert frame(machine, (0, 0), False, 1.9) == []
    assert machine.tick(2) == ["NO_DATA"]
    assert machine.state == SEARCH and not machine.active
    assert frame(machine, (1, 0), False, 2.1) == ["DATA_RECOVERED"]
    assert machine.active and machine.state == SEARCH


def test_manual_stop_stays_latched_through_data_recovery():
    machine = StateMachine()
    frame(machine, 1, True, 0)
    assert machine.set_enabled(False, .1) == ["MANUAL_STOP"]
    frame(machine, 2, True, .2)
    machine.tick(2.2)
    frame(machine, 1, True, 2.3)
    assert not machine.active and machine.state == SEARCH
    assert machine.set_enabled(True, 2.4) == ["MANUAL_ENABLE"]
    assert machine.active
    assert machine.state == SEARCH


def test_deadlines_are_applied_before_same_time_frame_and_frame_counted_once():
    machine = StateMachine(detector_timeout_s=20)
    frame(machine, 1, True, 0)
    frame(machine, 2, True, .1)
    frame(machine, 3, False, .2)
    assert machine.state == ACQUIRE
    events = frame(machine, 4, True, 10.2)
    assert events == ["ACQUIRE_TIMEOUT"]
    assert machine.state == SEARCH
    frame(machine, 5, True, 10.3)
    assert machine.state == ACQUIRE


def test_one_multi_box_message_is_one_positive_and_empty_is_negative():
    machine = StateMachine()
    frame(machine, 1, True, 0)  # nonempty array, regardless of box count
    assert machine.state == SEARCH
    frame(machine, 2, False, .1)  # empty array
    assert machine.state == SEARCH


def test_configuration_validation():
    StateMachine()
    for field, value in (("acquire_hits", 0), ("acquire_window", -1), ("follow_confirm_hits", 1.2), ("follow_lost_misses", True), ("detector_timeout_s", 0), ("acquire_timeout_s", math.inf)):
        with pytest.raises(ValueError):
            StateMachine(**{field: value})
    with pytest.raises(ValueError):
        StateMachine(acquire_hits=4, acquire_window=3)


def test_source_stamps_must_be_strictly_increasing_and_valid():
    machine = StateMachine()
    frame(machine, (12, 100), True, 0)
    frame(machine, (12, 99), True, .5)
    frame(machine, (12, 100), True, .6)
    assert machine.state == SEARCH
    assert machine.last_receipt_time == 0
    frame(machine, (12, 101), True, .7)
    assert machine.state == ACQUIRE


def test_search_is_requested_only_for_active_search_state():
    assert search_should_run(SEARCH, True)
    assert not search_should_run(SEARCH, False)
    assert not search_should_run(ACQUIRE, True)
    assert not search_should_run(FOLLOW, True)


def test_search_enable_request_waits_retries_and_applies_latest_state():
    binding = SearchEnableBinding()
    assert binding.update(False)
    assert binding.request_if_ready(False) is None
    assert binding.request_if_ready(True) is False
    assert binding.request_if_ready(True) is None  # one request in flight
    assert not binding.complete(False, False)  # response failure remains retryable
    assert binding.request_if_ready(True) is False
    assert binding.complete(False, True)
    assert binding.request_if_ready(True) is None  # confirmed until desired changes

    binding.update(True)
    assert binding.request_if_ready(True) is True
    binding.update(False)  # a late success must be followed by the latest stop request
    assert binding.request_if_ready(True) is None
    assert binding.complete(True, True)
    assert binding.request_if_ready(True) is False
    assert binding.complete(False, True)
    assert binding.request_if_ready(True) is None
