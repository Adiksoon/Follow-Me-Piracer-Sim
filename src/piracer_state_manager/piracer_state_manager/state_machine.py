"""ROS-independent SEARCH/ACQUIRE/FOLLOW state machine."""

import math
from collections import deque


SEARCH = "SEARCH"
ACQUIRE = "ACQUIRE"
FOLLOW = "FOLLOW"


def confirmed_command_owner(target, active, confirmed, pending):
    """Return an actuator owner only after an exclusive confirmed handoff."""
    if not active or pending is not None or not isinstance(target, str):
        return "stop"
    target_name = target.lower()
    if target_name not in (SEARCH.lower(), ACQUIRE.lower(), FOLLOW.lower()):
        return "stop"
    if confirmed.get(target_name) is not True:
        return "stop"
    if any(name != target_name and value is not False for name, value in confirmed.items()):
        return "stop"
    return target_name



class BehaviorEnableCoordinator:
    """Serialize SetBool handoffs between enabled behavior nodes."""

    def __init__(self, behavior_names):
        self.behavior_names = tuple(behavior_names)
        self.confirmed = {name: None for name in self.behavior_names}
        self.pending = None

    def next_request(self, target, services_ready):
        """Disable non-target behaviors before requesting the target enabled."""
        if self.pending is not None:
            return None
        if target is not None and target not in self.confirmed:
            raise ValueError(f"unknown behavior target: {target}")
        for name in self.behavior_names:
            needs_disable = self.confirmed[name] is None or (
                name != target and self.confirmed[name] is not False
            )
            if needs_disable:
                return (name, False) if services_ready.get(name, False) else None
        if target is not None and self.confirmed[target] is not True:
            return (target, True) if services_ready.get(target, False) else None
        return None

    def begin(self, behavior, enabled):
        request = (behavior, bool(enabled))
        if self.pending is not None or behavior not in self.confirmed:
            return False
        self.pending = request
        return True

    def complete(self, behavior, enabled, succeeded):
        request = (behavior, bool(enabled))
        if self.pending != request:
            return False
        self.pending = None
        if succeeded:
            self.confirmed[behavior] = bool(enabled)
            return True
        self.confirmed[behavior] = None
        return False


class StateMachine:
    def __init__(
        self,
        acquire_hits=2,
        acquire_window=3,
        follow_confirm_hits=5,
        follow_lost_misses=3,
        detector_timeout_s=2.0,
        acquire_timeout_s=30.0,
    ):
        for name, value in (
            ("acquire_hits", acquire_hits),
            ("acquire_window", acquire_window),
            ("follow_confirm_hits", follow_confirm_hits),
            ("follow_lost_misses", follow_lost_misses),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if acquire_hits > acquire_window:
            raise ValueError("acquire_hits must be <= acquire_window")
        durations = (("detector_timeout_s", detector_timeout_s), ("acquire_timeout_s", acquire_timeout_s))
        for name, value in durations:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite duration")
        self.acquire_hits = acquire_hits
        self.search_hits = deque(maxlen=acquire_window)
        self.follow_confirm_hits = follow_confirm_hits
        self.follow_lost_misses = follow_lost_misses
        self.detector_timeout_s = float(detector_timeout_s)
        self.acquire_timeout_s = float(acquire_timeout_s)
        self.enabled = True
        self.active = False
        self.data_paused = False
        self.state = SEARCH
        self.acquire_finished = False
        self.last_receipt_time = None
        self.last_stamp = None
        self.acquire_started = None
        self.hit_streak = 0
        self.miss_streak = 0
        self._confirmation_stamp = None

    def _reset_state(self, state=None, now=None):
        if state is None:
            state = SEARCH
        if state != ACQUIRE or self.state != ACQUIRE:
            self.acquire_finished = False
        self.state = state
        self.search_hits.clear()
        self.hit_streak = 0
        self.miss_streak = 0
        self._confirmation_stamp = self.last_stamp if state == ACQUIRE else None
        self.acquire_started = now if state == ACQUIRE else None

    def invalidate_detector_data(self):
        """Drop the previous clock epoch's evidence without releasing manual STOP."""
        self._reset_state()
        self.active = False
        self.data_paused = True
        self.last_receipt_time = None
        self.last_stamp = None

    def _timeout(self, now):
        return self.last_receipt_time is not None and now - self.last_receipt_time >= self.detector_timeout_s

    def _apply_deadlines(self, now):
        events = []
        if self._timeout(now):
            self._reset_state()
            self.active = False
            self.data_paused = True
            self.last_receipt_time = None
            self.last_stamp = None
            events.append("NO_DATA")
        if self.state == ACQUIRE and now - self.acquire_started >= self.acquire_timeout_s:
            self._reset_state()
            events.append("ACQUIRE_TIMEOUT")
        return events

    def tick(self, now):
        """Apply wall-clock deadlines; return event labels for one-time logging."""
        return self._apply_deadlines(float(now))

    def observe(self, stamp, positive, now, *, can_acquire=False,
                target_visible=False, hold_follow=False, target_stamp=None):
        """Process one Detection2DArray header stamp and target-filtered result."""
        now = float(now)
        events = self._apply_deadlines(now)
        if isinstance(stamp, tuple) and len(stamp) == 2:
            sec, nanosec = stamp
            invalid_parts = any(
                isinstance(part, bool) or not isinstance(part, int) for part in stamp
            )
            if invalid_parts or sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
                return events
            stamp = (sec, nanosec)
            if sec == 0 and nanosec == 0:
                return events
        elif isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp) and stamp > 0:
            stamp = float(stamp)
        else:
            return events
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return events
        recovering = self.data_paused
        self.last_stamp = stamp
        self.last_receipt_time = now
        if recovering:
            self.data_paused = False
            self._reset_state()
            events.append("DATA_RECOVERED")
        self.active = self.enabled
        if not self.enabled:
            return events
        if self.state == SEARCH:
            self.search_hits.append(bool(positive))
            if sum(self.search_hits) >= self.acquire_hits and can_acquire:
                self._reset_state(ACQUIRE, now)
                if target_stamp is not None:
                    self._confirmation_stamp = max(stamp, target_stamp)
                events.append("SEARCH_TO_ACQUIRE")
            return events

        if self.state == ACQUIRE:
            if not positive:
                self.invalidate_target_confirmation()
                self._confirmation_stamp = stamp
        elif self.state == FOLLOW:
            self.miss_streak = 0 if hold_follow or positive and target_visible else self.miss_streak + 1
            if self.miss_streak >= self.follow_lost_misses:
                self._reset_state(ACQUIRE if can_acquire else SEARCH, now)
                if can_acquire and target_stamp is not None:
                    self._confirmation_stamp = max(stamp, target_stamp)
                events.append("FOLLOW_TO_ACQUIRE" if can_acquire else "FOLLOW_TO_SEARCH")
        return events

    def confirm_target(self, stamp, now):
        """Count each new valid 3D observation once, separately from YOLO frames."""
        events = self.tick(now)
        if not self.active or not self.enabled or self.data_paused:
            return events
        if self.state == SEARCH and sum(self.search_hits) >= self.acquire_hits:
            self._reset_state(ACQUIRE, now)
            self._confirmation_stamp = max(self.last_stamp, stamp)
            return events + ["SEARCH_TO_ACQUIRE"]
        if self.state != ACQUIRE or stamp <= self._confirmation_stamp:
            return events
        self._confirmation_stamp = stamp
        self.hit_streak += 1
        if self.hit_streak >= self.follow_confirm_hits:
            self._reset_state(FOLLOW)
            events.append("ACQUIRE_TO_FOLLOW")
        return events

    def invalidate_target_confirmation(self):
        self.hit_streak = 0

    def finish_acquire(self, outcome="already_close"):
        """Normal completion releases to SEARCH; faults retain manual STOP."""
        if not self.active or self.state != ACQUIRE or self.acquire_finished:
            return []
        if outcome == "failed":
            return self.set_enabled(False)
        if outcome != "already_close":
            self._reset_state(SEARCH)
            return ["ACQUIRE_TO_SEARCH"]
        self.acquire_finished = True
        return ["ACQUIRE_FINISHED"]

    def set_enabled(self, enabled, now=None):
        """Latch manual enable/stop and restart at SEARCH."""
        self.enabled = bool(enabled)
        if self.enabled:
            self.acquire_finished = False
        self._reset_state()
        if not self.enabled:
            self.active = False
            return ["MANUAL_STOP"]
        now = self.last_receipt_time if now is None else float(now)
        self.active = self.last_receipt_time is not None and now is not None and not self._timeout(now)
        return ["MANUAL_ENABLE"]
