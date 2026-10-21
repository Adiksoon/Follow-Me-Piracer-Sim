"""Small, ROS-independent SEARCH/ACQUIRE/FOLLOW state machine."""

from collections import deque
import math


SEARCH = "SEARCH"
ACQUIRE = "ACQUIRE"
FOLLOW = "FOLLOW"


def search_should_run(state, active):
    """Return whether current manager output requests Search to run."""
    return bool(active) and state == SEARCH


class SearchEnableBinding:
    """Track one asynchronous SetBool request and retry until it succeeds."""

    def __init__(self):
        self.desired = None
        self.confirmed = None
        self.pending = None

    def update(self, desired):
        changed = self.desired is None or self.desired != bool(desired)
        self.desired = bool(desired)
        return changed

    def request_if_ready(self, service_ready):
        if not service_ready or self.pending is not None or self.desired == self.confirmed:
            return None
        self.pending = self.desired
        return self.pending

    def complete(self, requested, succeeded):
        if self.pending is None or requested != self.pending:
            return False
        self.pending = None
        if succeeded:
            self.confirmed = requested
            return True
        return False


class StateMachine:
    def __init__(
        self,
        acquire_hits=2,
        acquire_window=3,
        follow_confirm_hits=5,
        follow_lost_misses=3,
        detector_timeout_s=2.0,
        acquire_timeout_s=10.0,
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
        self.acquire_window = acquire_window
        self.follow_confirm_hits = follow_confirm_hits
        self.follow_lost_misses = follow_lost_misses
        self.detector_timeout_s = float(detector_timeout_s)
        self.acquire_timeout_s = float(acquire_timeout_s)
        self.enabled = True
        self.active = False
        self.data_paused = False
        self.state = SEARCH
        self.last_receipt_time = None
        self.last_stamp = None
        self.acquire_started = None
        self.window = deque(maxlen=acquire_window)
        self.hit_streak = 0
        self.miss_streak = 0

    def _reset_state(self, state=SEARCH, now=None):
        self.state = state
        self.window.clear()
        self.hit_streak = 0
        self.miss_streak = 0
        self.acquire_started = now if state == ACQUIRE else None

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

    def observe(self, stamp, positive, now):
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
            self.window.append(bool(positive))
            if sum(self.window) >= self.acquire_hits:
                self._reset_state(ACQUIRE, now)
                events.append("SEARCH_TO_ACQUIRE")
        elif self.state == ACQUIRE:
            self.hit_streak = self.hit_streak + 1 if positive else 0
            if self.hit_streak >= self.follow_confirm_hits:
                self._reset_state(FOLLOW)
                events.append("ACQUIRE_TO_FOLLOW")
        elif self.state == FOLLOW:
            self.miss_streak = 0 if positive else self.miss_streak + 1
            if self.miss_streak >= self.follow_lost_misses:
                self._reset_state(ACQUIRE, now)
                events.append("FOLLOW_TO_ACQUIRE")
        return events

    def set_enabled(self, enabled, now=None):
        """Latch manual enable/stop and restart behavioral state at SEARCH."""
        self.enabled = bool(enabled)
        self._reset_state()
        if not self.enabled:
            self.active = False
            return ["MANUAL_STOP"]
        now = self.last_receipt_time if now is None else float(now)
        self.active = self.last_receipt_time is not None and now is not None and not self._timeout(now)
        return ["MANUAL_ENABLE"]
