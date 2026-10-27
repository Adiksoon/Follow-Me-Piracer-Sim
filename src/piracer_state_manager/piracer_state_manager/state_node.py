"""ROS adapter for the pure detector-driven state machine."""

import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray

from piracer_state_manager.state_machine import (
    ACQUIRE,
    SEARCH,
    BehaviorEnableCoordinator,
    StateMachine,
)


class StateManagerNode(Node):
    def __init__(self):
        super().__init__("piracer_state_manager")
        defaults = {
            "detection_topic": "/yolo/detections",
            "search_enable_service": "/search/enable",
            "acquire_enable_service": "/acquire/enable",
            "follow_enable_service": "/follow/enable",
            "acquire_hits": 2,
            "acquire_window": 3,
            "follow_confirm_hits": 5,
            "follow_lost_misses": 3,
            "detector_timeout_s": 2.0,
            "acquire_timeout_s": 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}
        state_machine_parameters = {
            name: value
            for name, value in values.items()
            if name not in {
                "detection_topic",
                "search_enable_service",
                "acquire_enable_service",
                "follow_enable_service",
            }
        }
        self.machine = StateMachine(**state_machine_parameters)
        self._last_sim_time_ns = None
        self._sim_epoch_start_ns = 0
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.run_pub = self.create_publisher(Bool, "/piracer/state/run", output_qos)
        self._published_run = None
        self._publish_changes(force=True)
        self.behavior_coordinator = BehaviorEnableCoordinator(
            ("search", "acquire", "follow")
        )
        self.behavior_clients = {
            "search": self.create_client(SetBool, values["search_enable_service"]),
            "acquire": self.create_client(SetBool, values["acquire_enable_service"]),
            "follow": self.create_client(SetBool, values["follow_enable_service"]),
        }
        self._handoff_failure_logged = False
        self._handoff_started_monotonic = None
        self._handoff_future = None
        self._handoff_wait_logged = False
        self._blocking_service_logged = None
        self.subscription = self.create_subscription(
            Detection2DArray, values["detection_topic"], self._on_detections, qos_profile_sensor_data
        )
        self.service = self.create_service(SetBool, "/piracer/state/enable", self._on_enable)
        steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(0.1, self._on_timer, clock=steady_clock)
        self._sync_behavior_services()

    def _publish_changes(self, force=False):
        if force or self.machine.active != self._published_run:
            msg = Bool()
            msg.data = self.machine.active
            self.run_pub.publish(msg)
            self._published_run = self.machine.active

    def _log_events(self, events):
        for event in events:
            if event in ("SEARCH_TO_ACQUIRE", "ACQUIRE_TO_FOLLOW", "FOLLOW_TO_ACQUIRE"):
                self.get_logger().info(f"State transition: {event}")
            elif event == "NO_DATA":
                self.get_logger().warning("NO_DATA: detector stream timed out; paused in SEARCH")
            elif event == "DATA_RECOVERED":
                self.get_logger().info("Detector data recovered; resumed in SEARCH")
            elif event == "MANUAL_STOP":
                self.get_logger().info("MANUAL_STOP: state manager disabled")
            elif event == "MANUAL_ENABLE":
                self.get_logger().info("Manual enable: restarted in SEARCH")
            elif event == "ACQUIRE_TIMEOUT":
                self.get_logger().info("ACQUIRE attempt timed out; returned to SEARCH")

    def _freshness_now(self):
        """Use simulation time for simulation deadlines, monotonic time on hardware."""
        if not self.get_parameter("use_sim_time").value:
            return time.monotonic()
        now_ns = self.get_clock().now().nanoseconds
        previous_ns = self._last_sim_time_ns
        self._last_sim_time_ns = now_ns
        if previous_ns is not None and now_ns < previous_ns:
            self.machine.invalidate_detector_data()
            self._sim_epoch_start_ns = now_ns
            self.get_logger().warning("Simulation clock moved backwards; detector data invalidated")
            return None
        return now_ns / 1_000_000_000 if now_ns > 0 else None

    def _on_detections(self, message):
        now = self._freshness_now()
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        events = []
        if now is not None:
            valid_epoch = True
            if self.get_parameter("use_sim_time").value:
                stamp_ns = stamp[0] * 1_000_000_000 + stamp[1]
                valid_epoch = 0 < stamp_ns and self._sim_epoch_start_ns <= stamp_ns <= self._last_sim_time_ns
            events = (self.machine.observe(stamp, bool(message.detections), now)
                      if valid_epoch else self.machine.tick(now))
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_timer(self):
        now = self._freshness_now()
        events = [] if now is None else self.machine.tick(now)
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_enable(self, request, response):
        now = self._freshness_now()
        events = self.machine.set_enabled(request.data, 0.0 if now is None else now)
        response.success = True
        response.message = "enabled" if request.data else "stopped"
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()
        return response

    def _sync_behavior_services(self):
        if not self.machine.active:
            target = None
        elif self.machine.state == SEARCH:
            target = "search"
        elif self.machine.state == ACQUIRE:
            target = "acquire"
        else:
            target = None
        services_ready = {
            name: client.service_is_ready()
            for name, client in self.behavior_clients.items()
        }
        requested = self.behavior_coordinator.next_request(target, services_ready)
        if requested is None:
            pending = self.behavior_coordinator.pending
            if pending is not None and self._handoff_started_monotonic is not None:
                elapsed = time.monotonic() - self._handoff_started_monotonic
                if (
                    elapsed >= 5.0
                    and self.get_parameter("use_sim_time").value
                    and pending == ("search", False)
                    and all(value is None for value in self.behavior_coordinator.confirmed.values())
                    and self._handoff_future is not None
                    and not self._handoff_future.done()
                ):
                    # TYMCZASOWO: tylko na czas testowania SEARCH w symulacji; usunac po dopracowaniu startu stanow.
                    old_future = self._handoff_future
                    self._handoff_future = None
                    self.behavior_clients["search"].remove_pending_request(old_future)
                    self.behavior_coordinator.complete("search", False, False)
                    old_future.cancel()
                    self._handoff_started_monotonic = None
                    self._handoff_wait_logged = False
                    self._handoff_failure_logged = False
                    self.get_logger().warning(
                        "Timed out initial Search disable in simulation; retrying"
                    )
                    return
                if elapsed >= 5.0 and not self._handoff_wait_logged:
                    done = None if self._handoff_future is None else self._handoff_future.done()
                    self.get_logger().warning(
                        f"Behavior pending: request={pending}, elapsed_wall_s={elapsed:.3f}, future_done={done}"
                    )
                    self._handoff_wait_logged = True
            elif pending is None:
                # next_request is pure: ask which request would run if services were ready.
                wanted = self.behavior_coordinator.next_request(
                    target, {name: True for name in self.behavior_clients}
                )
                blocked = None if wanted is None else wanted[0]
                if blocked != self._blocking_service_logged:
                    if blocked is not None:
                        self.get_logger().info(
                            f"Behavior waiting for service: {self.behavior_clients[blocked].srv_name}"
                        )
                    self._blocking_service_logged = blocked
            return
        behavior, enabled = requested
        if not self.behavior_coordinator.begin(behavior, enabled):
            return
        request = SetBool.Request()
        request.data = enabled
        self._handoff_started_monotonic = time.monotonic()
        self._handoff_future = None
        self._handoff_wait_logged = False
        self._blocking_service_logged = None
        self.get_logger().info(
            f"Behavior request: service={self.behavior_clients[behavior].srv_name}, "
            f"enabled={enabled}, monotonic_s={self._handoff_started_monotonic:.6f}"
        )
        try:
            future = self.behavior_clients[behavior].call_async(request)
        except Exception as error:
            self.get_logger().error(f"Behavior request dispatch raised: {error}")
            raise
        self._handoff_future = future
        future.add_done_callback(
            lambda completed, sent=behavior, value=enabled: self._on_behavior_enable_result(
                sent, value, completed
            )
        )

    def _on_behavior_enable_result(self, behavior, enabled, future):
        if future is not self._handoff_future:
            return
        try:
            response = future.result()
        except Exception as exc:  # Service failures remain retryable on the next timer tick.
            succeeded = False
            detail = str(exc)
        else:
            succeeded = response.success
            detail = response.message
        accepted = self.behavior_coordinator.complete(behavior, enabled, succeeded)
        if succeeded and accepted:
            self._handoff_failure_logged = False
            self.get_logger().info(
                f"{behavior.capitalize()} {'enabled' if enabled else 'disabled'} by behavior state"
            )
            self._sync_behavior_services()
        elif not self._handoff_failure_logged:
            self._handoff_failure_logged = True
            self.get_logger().warning(
                f"{behavior} enable service did not accept {enabled}: {detail}; will retry"
            )


def main(args=None):
    rclpy.init(args=args)
    node = StateManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
