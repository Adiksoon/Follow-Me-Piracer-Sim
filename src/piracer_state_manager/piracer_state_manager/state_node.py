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

    def _on_detections(self, message):
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        events = self.machine.observe(stamp, bool(message.detections), time.monotonic())
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_timer(self):
        events = self.machine.tick(time.monotonic())
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_enable(self, request, response):
        events = self.machine.set_enabled(request.data, time.monotonic())
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
            return
        behavior, enabled = requested
        if not self.behavior_coordinator.begin(behavior, enabled):
            return
        request = SetBool.Request()
        request.data = enabled
        future = self.behavior_clients[behavior].call_async(request)
        future.add_done_callback(
            lambda completed, sent=behavior, value=enabled: self._on_behavior_enable_result(
                sent, value, completed
            )
        )

    def _on_behavior_enable_result(self, behavior, enabled, future):
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
