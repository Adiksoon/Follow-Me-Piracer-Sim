"""Dedicated TF reception and bounded OdomInfo handoff for Search."""

from collections import deque
from dataclasses import dataclass
from threading import Lock, Thread

from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import TransformListener


@dataclass(frozen=True)
class OdomInfoSample:
    """Small immutable handoff record; never retain the large ROS message."""

    sequence: int
    stamp_ns: int
    frame_id: str
    lost: bool
    lost_stamp_floor_ns: int = 0


class OdomInfoHandoff:
    """Bounded, thread-safe handoff from OdomInfo reception to Search decisions."""

    def __init__(self, capacity=10):
        self._lock = Lock()
        self._samples = deque(maxlen=capacity)
        self._sequence = 0
        self._pending_loss = None
        self._barrier_stamp_ns = None

    def add(self, stamp_ns, frame_id, lost):
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            stamp_ns = int(stamp_ns)
            sample = OdomInfoSample(
                sequence=sequence,
                stamp_ns=stamp_ns,
                frame_id=str(frame_id),
                lost=bool(lost),
            )
            if sample.lost:
                previous = self._pending_loss
                self._pending_loss = OdomInfoSample(
                    sequence=sequence,
                    stamp_ns=stamp_ns,
                    frame_id=sample.frame_id,
                    lost=True,
                    lost_stamp_floor_ns=max(
                        stamp_ns if stamp_ns > 0 else 0,
                        previous.lost_stamp_floor_ns if previous else 0,
                    ),
                )
                # A loss clears earlier good samples, while later good samples
                # remain available for recovery after Search sees this event.
                self._samples.clear()
                disposition = "lost"
            elif stamp_ns == 0:
                disposition = "dropped_zero_stamp"
            elif (self._barrier_stamp_ns is not None
                  and stamp_ns <= self._barrier_stamp_ns):
                disposition = "dropped_invalidation_barrier"
            else:
                self._samples.append(sample)
                disposition = "queued"
            return sample, disposition

    def snapshot(self):
        with self._lock:
            return tuple(self._samples), self._pending_loss, self._sequence

    def acknowledge_loss(self, sequence):
        with self._lock:
            if (self._pending_loss is not None
                    and self._pending_loss.sequence == sequence):
                self._pending_loss = None

    def invalidate(self, barrier_stamp_ns, *, after_sequence):
        with self._lock:
            self._barrier_stamp_ns = barrier_stamp_ns
            self._samples = deque(
                (sample for sample in self._samples
                 if sample.sequence > after_sequence
                 and (barrier_stamp_ns is None
                      or sample.stamp_ns > barrier_stamp_ns)),
                maxlen=self._samples.maxlen,
            )

    def reset(self):
        with self._lock:
            self._samples.clear()
            self._pending_loss = None
            self._barrier_stamp_ns = None
            return self._sequence

    def set_barrier(self, barrier_stamp_ns):
        with self._lock:
            self._barrier_stamp_ns = barrier_stamp_ns


class SearchTfReceiver:
    """Own a TF-only node/executor; share only the existing thread-safe buffer."""

    def __init__(self, buffer, error_reporter, *, context, namespace,
                 use_sim_time, odom_info_wakeup):
        self.executor = self.listener = self.thread = None
        self._closed = False
        self._error_reporter = error_reporter
        self._odom_info_wakeup = odom_info_wakeup
        self._odom_info_handoff = OdomInfoHandoff()
        # Do not inherit the launch's __node:=piracer_search remapping.
        self.node = Node(
            "piracer_search_tf_listener", context=context, namespace=namespace,
            use_global_arguments=False, enable_rosout=False,
            start_parameter_services=False,
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        try:
            self.executor = SingleThreadedExecutor(context=context)
            self.executor.add_node(self.node)
            self.listener = TransformListener(buffer, self.node, spin_thread=False)
            self.thread = Thread(
                target=self._spin, name="piracer-search-tf", daemon=False
            )
        except BaseException:
            self.close()
            raise

    def start(self):
        self.thread.start()

    def _spin(self):
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass  # The shared ROS context may already be shut down by SIGINT.

    @staticmethod
    def _stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def receive_odom_info(self, message):
        """Metadata utility called by SearchNode, outside the TF executor."""
        if self._closed:
            return None
        stamp_ns = self._stamp_ns(message)
        frame_id = message.header.frame_id
        lost = bool(message.lost)
        sample, disposition = self._odom_info_handoff.add(stamp_ns, frame_id, lost)
        if disposition in ("queued", "lost") and not self._closed:
            try:
                self._odom_info_wakeup.trigger()
            except Exception as exc:
                self._report_error("odom_info_wakeup", exc)
        return sample, disposition

    def odom_info_snapshot(self):
        return self._odom_info_handoff.snapshot()

    def acknowledge_odom_info_loss(self, sequence):
        self._odom_info_handoff.acknowledge_loss(sequence)

    def invalidate_odom_info(self, barrier_stamp_ns, *, after_sequence):
        self._odom_info_handoff.invalidate(
            barrier_stamp_ns, after_sequence=after_sequence
        )

    def reset_odom_info(self):
        return self._odom_info_handoff.reset()

    def set_odom_info_barrier(self, barrier_stamp_ns):
        self._odom_info_handoff.set_barrier(barrier_stamp_ns)

    def _report_error(self, operation, exc):
        if self._error_reporter is not None:
            try:
                self._error_reporter(operation, exc)
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            self.executor.shutdown()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join()
        if self.listener is not None:
            self.listener.unregister()
        self.node.executor = None
        self.node.destroy_node()
