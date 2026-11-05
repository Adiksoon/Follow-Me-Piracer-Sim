"""ROS adapter for the pure detector-driven state machine."""

import json
import math
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray

from piracer_state_manager.state_machine import (
    ACQUIRE,
    FOLLOW,
    SEARCH,
    BehaviorEnableCoordinator,
    StateMachine,
    confirmed_command_owner,
)


class StateManagerNode(Node):
    def __init__(self):
        super().__init__("piracer_state_manager")
        defaults = {
            "detection_topic": "/yolo/detections",
            "target_point_topic": "/piracer/follow/target_point",
            "search_enable_service": "/search/enable",
            "acquire_enable_service": "/acquire/enable",
            "acquire_result_topic": "/piracer/acquire/result",
            "follow_enable_service": "/follow/enable",
            "acquire_hits": 2,
            "acquire_window": 3,
            "follow_confirm_hits": 5,
            "follow_lost_misses": 3,
            "detector_timeout_s": 2.0,
            "acquire_timeout_s": 30.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}
        state_machine_parameters = {
            name: value
            for name, value in values.items()
            if name not in {
                "detection_topic",
                "target_point_topic",
                "search_enable_service",
                "acquire_enable_service",
                "acquire_result_topic",
                "follow_enable_service",
            }
        }
        self.machine = StateMachine(**state_machine_parameters)
        self._last_sim_time_ns = None
        self._sim_epoch_start_ns = 0
        self._target_stamp = self._target_received = None
        self._follow_status = self._follow_status_received = None
        self._follow_enabled_wall = None
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.run_pub = self.create_publisher(Bool, "/piracer/state/run", output_qos)
        self.command_owner_pub = self.create_publisher(
            String, "/piracer/state/command_owner", output_qos
        )
        self._command_owner = None
        self._publish_command_owner("stop")
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
        self._behavior_enable_dispatched = False
        self._acquire_attempt_id = None
        self._acquire_results_pending = {}
        self.result_subscription = self.create_subscription(
            String, values["acquire_result_topic"], self._on_acquire_result, 10
        )
        self.subscription = self.create_subscription(
            Detection2DArray, values["detection_topic"], self._on_detections, qos_profile_sensor_data
        )
        self.point_subscription = self.create_subscription(
            PointStamped, values["target_point_topic"], self._on_target_point, 1
        )
        self.follow_status_subscription = self.create_subscription(
            String, "/piracer/follow/status", self._on_follow_status, output_qos
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

    def _publish_command_owner(self, owner):
        if owner == self._command_owner:
            return
        message = String()
        message.data = owner
        self.command_owner_pub.publish(message)
        self._command_owner = owner

    def _log_events(self, events):
        for event in events:
            if event in ("SEARCH_TO_ACQUIRE", "ACQUIRE_TO_FOLLOW", "FOLLOW_TO_ACQUIRE",
                         "FOLLOW_TO_SEARCH", "ACQUIRE_TO_SEARCH"):
                self.get_logger().info(f"State transition: {event}")
            elif event == "NO_DATA":
                self.get_logger().warning("NO_DATA: detector stream timed out; paused in SEARCH")
            elif event == "DATA_RECOVERED":
                self.get_logger().info("Detector data recovered; resumed in SEARCH")
            elif event == "MANUAL_STOP":
                self.get_logger().info("MANUAL_STOP: state manager disabled")
            elif event == "MANUAL_ENABLE":
                self.get_logger().info("Manual enable: restarted in SEARCH")
            elif event == "ACQUIRE_FINISHED":
                self.get_logger().info("ACQUIRE finished; disabled for this entry")
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
            self._acquire_attempt_id = None
            self._acquire_results_pending.clear()
            self._target_stamp = self._target_received = None
            self._follow_status = self._follow_status_received = None
            self._sim_epoch_start_ns = now_ns
            self.get_logger().warning("Simulation clock moved backwards; detector data invalidated")
            return None
        return now_ns / 1_000_000_000 if now_ns > 0 else None

    def _point_fresh(self, limit=0.5):
        if self._target_stamp is None or self._target_received is None:
            return False
        sec, ns = self._target_stamp
        age = (self.get_clock().now().nanoseconds - (sec * 1_000_000_000 + ns)) / 1e9
        return (0 <= age <= limit
                and 0 <= time.monotonic() - self._target_received <= limit)

    def _on_target_point(self, message):
        now = self._freshness_now()
        if now is None:
            return
        events = self.machine.tick(now)
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        stamp_ns = stamp[0] * 1_000_000_000 + stamp[1]
        age_ns = self.get_clock().now().nanoseconds - stamp_ns
        valid = (stamp[0] >= 0 and 0 <= stamp[1] < 1_000_000_000
                 and stamp_ns > 0 and stamp_ns >= self._sim_epoch_start_ns
                 and 0 <= age_ns <= 500_000_000 and bool(message.header.frame_id)
                 and all(math.isfinite(v) for v in (message.point.x, message.point.y, message.point.z)))
        if valid and (self._target_stamp is None or stamp > self._target_stamp):
            if not self._point_fresh():
                self.machine.invalidate_target_confirmation()
            self._target_stamp, self._target_received = stamp, time.monotonic()
            events += self.machine.confirm_target(stamp, now)
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_follow_status(self, message):
        if message.data not in {"DISABLED", "DIRECT", "AVOIDING", "FAULT"}:
            return
        self._follow_status, self._follow_status_received = message.data, time.monotonic()
        self._check_follow_health()
        self._publish_changes()
        self._sync_behavior_services()

    def _check_follow_health(self):
        if self.behavior_coordinator.confirmed["follow"] is not True:
            return
        receipt = self._follow_status_received
        start = self._follow_enabled_wall if receipt is None else receipt
        if self._follow_status == "FAULT" or (start is not None and time.monotonic() - start > 0.5):
            if self.machine.enabled:
                self.get_logger().error("FOLLOW fault or missing status; manual enable required")
                self._log_events(self.machine.set_enabled(False))

    def _follow_avoiding(self):
        return (self._follow_status == "AVOIDING" and self._follow_status_received is not None
                and 0 <= time.monotonic() - self._follow_status_received <= 0.5
                and self.behavior_coordinator.confirmed["follow"] is True)

    def _on_detections(self, message):
        now = self._freshness_now()
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        events = []
        if now is not None:
            self._check_follow_health()
            valid_epoch = True
            if self.get_parameter("use_sim_time").value:
                stamp_ns = stamp[0] * 1_000_000_000 + stamp[1]
                valid_epoch = 0 < stamp_ns and self._sim_epoch_start_ns <= stamp_ns <= self._last_sim_time_ns
            events = (self.machine.observe(
                stamp, bool(message.detections), now,
                can_acquire=self._point_fresh(3.0), target_visible=self._point_fresh(),
                hold_follow=self._follow_avoiding(), target_stamp=self._target_stamp)
                      if valid_epoch else self.machine.tick(now))
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_timer(self):
        now = self._freshness_now()
        events = [] if now is None else self.machine.tick(now)
        if not self._point_fresh():
            self.machine.invalidate_target_confirmation()
        self._check_follow_health()
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()

    def _on_enable(self, request, response):
        now = self._freshness_now()
        events = self.machine.set_enabled(request.data, 0.0 if now is None else now)
        self._target_stamp = self._target_received = None
        if not request.data:
            self._acquire_attempt_id = None
            self._acquire_results_pending.clear()
        response.success = True
        response.message = "enabled" if request.data else "stopped"
        self._log_events(events)
        self._publish_changes()
        self._sync_behavior_services()
        return response

    def _on_acquire_result(self, message):
        try:
            result = json.loads(message.data)
            if (not isinstance(result, dict) or not isinstance(result.get("attempt_id"), str)
                    or result.get("outcome") not in {"exhausted", "already_close", "blocked", "failed", "canceled"}):
                return
        except (ValueError, TypeError):
            return
        if self.behavior_coordinator.pending == ("acquire", True):
            # The result may arrive before the corresponding enable response.
            self._acquire_results_pending[result["attempt_id"]] = message
            return
        if result["attempt_id"] != self._acquire_attempt_id:
            return
        events = (self.machine.set_enabled(False) if result["outcome"] == "failed"
                  else self.machine.finish_acquire(result["outcome"]))
        if events:
            self.get_logger().info(f"Acquire result: {message.data}")
            self._log_events(events)
            self._publish_changes()
            self._sync_behavior_services()


    def _sync_behavior_services(self):
        if not self.machine.active:
            target = None
        elif self.machine.state == SEARCH:
            target = "search"
        elif self.machine.state == ACQUIRE:
            target = None if self.machine.acquire_finished else "acquire"
        elif self.machine.state == FOLLOW:
            target = "follow"
        else:
            target = None
        publish_owner = getattr(self, "_publish_command_owner", None)
        if publish_owner is not None:
            publish_owner(confirmed_command_owner(
                target,
                self.machine.active,
                self.behavior_coordinator.confirmed,
                self.behavior_coordinator.pending,
            ))
        services_ready = {
            name: client.service_is_ready()
            for name, client in self.behavior_clients.items()
        }
        requested = self.behavior_coordinator.next_request(target, services_ready)
        if requested is None:
            pending = self.behavior_coordinator.pending
            if pending is not None and self._handoff_started_monotonic is not None:
                elapsed = time.monotonic() - self._handoff_started_monotonic
                confirmed = self.behavior_coordinator.confirmed
                initial_search_disable = (
                    pending == ("search", False)
                    and all(value is None for value in confirmed.values())
                )
                initial_follow_disable = (
                    pending == ("follow", False)
                    and confirmed.get("search") is False
                    and confirmed.get("acquire") is False
                    and confirmed.get("follow") is None
                )
                initial_acquire_disable = (
                    pending == ("acquire", False)
                    and confirmed.get("search") is False
                    and confirmed.get("acquire") is None
                    and confirmed.get("follow") is None
                )
                if (
                    elapsed >= 5.0
                    and self.get_parameter("use_sim_time").value
                    and not self._behavior_enable_dispatched
                    and (initial_search_disable or initial_acquire_disable or initial_follow_disable)
                    and self._handoff_future is not None
                    and not self._handoff_future.done()
                ):
                    old_future = self._handoff_future
                    self._handoff_future = None
                    behavior = pending[0]
                    self.behavior_clients[behavior].remove_pending_request(old_future)
                    self.behavior_coordinator.complete(behavior, False, False)
                    old_future.cancel()
                    self._handoff_started_monotonic = None
                    self._handoff_wait_logged = False
                    self._handoff_failure_logged = False
                    self.get_logger().warning(
                        f"Timed out initial {behavior.capitalize()} disable in simulation; retrying"
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
        if behavior == "acquire" and enabled:
            self._acquire_attempt_id = None
            self._acquire_results_pending.clear()
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
        if enabled:
            self._behavior_enable_dispatched = True
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
        if behavior == "follow":
            if enabled and succeeded and accepted:
                self._follow_status = self._follow_status_received = None
                self._follow_enabled_wall = time.monotonic()
        if behavior in ("follow", "acquire") and not enabled and "fault_latched=true" in detail:
            # Fault results and service responses can arrive in either order.
            self._log_events(self.machine.set_enabled(False))
            self._publish_changes()
        if behavior == "acquire" and enabled:
            self._acquire_attempt_id = detail.partition("attempt_id=")[2] if succeeded and accepted else None
            pending = self._acquire_results_pending.get(self._acquire_attempt_id)
            self._acquire_results_pending.clear()
        else:
            pending = None
        if succeeded and accepted:
            self._handoff_failure_logged = False
            self.get_logger().info(
                f"{behavior.capitalize()} {'enabled' if enabled else 'disabled'} by behavior state"
            )
            if pending is not None:
                self._on_acquire_result(pending)
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
