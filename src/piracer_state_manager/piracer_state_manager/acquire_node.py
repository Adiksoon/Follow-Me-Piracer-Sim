"""A bounded sequence of validated Nav2 paths toward a remembered suitcase."""

import copy
from collections import deque
import json
import math
from threading import Thread
import time
import uuid

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.srv import IsPathValid
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rtabmap_msgs.msg import OdomInfo
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from rclpy.utilities import try_shutdown
import tf2_geometry_msgs  # Registers PointStamped with tf2.

from piracer_state_manager.acquire_geometry import observation_goal, path_length


class AcquireTfReceiver:
    """Receive TF on a dedicated node and executor, independently of Acquire callbacks."""

    def __init__(self, buffer, *, context, namespace, use_sim_time):
        self.executor = self.listener = self.thread = None
        self._closed = False
        self.node = Node(
            "piracer_acquire_tf_listener", context=context, namespace=namespace,
            use_global_arguments=False, enable_rosout=False,
            start_parameter_services=False,
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        try:
            self.executor = SingleThreadedExecutor(context=context)
            self.executor.add_node(self.node)
            self.listener = TransformListener(buffer, self.node, spin_thread=False)
            self.thread = Thread(
                target=self._spin, name="piracer-acquire-tf", daemon=False
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
            pass

    def close(self):
        if self._closed:
            return
        if self.executor is not None:
            self.executor.shutdown()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join()
        if self.listener is not None:
            self.listener.unregister()
        self.node.executor = None
        self.node.destroy_node()
        self._closed = True


class AcquireNode(Node):
    def __init__(self):
        super().__init__("piracer_acquire")
        defaults = {
            "enable_service": "/acquire/enable", "point_topic": "/piracer/follow/target_point",
            "odom_info_topic": "/visual_vo/odom_info_lite", "odom_topic": "/visual_odom",
            "scan_topic": "/piracer/scan", "planner_action": "/compute_path_to_pose",
            "controller_action": "/follow_path", "path_valid_service": "/is_path_valid",
            "result_topic": "/piracer/acquire/result",
            "command_topic": "/piracer/ackermann_cmd_vel",
            "goal_checker_id": "acquire_goal_checker",
            "map_frame": "map", "odom_frame": "vo_odom", "base_frame": "base_link",
            "standoff_m": 2.2, "max_goal_distance_m": 0.5, "max_path_length_m": 1.0,
            "max_travel_distance_m": 1.2, "stop_margin_m": 0.15,
            "max_maneuver_time_s": 30.0, "observation_max_age_s": 3.0,
            "max_target_shift_m": 0.25,
            "localization_max_age_s": 1.0, "scan_max_age_s": 1.0, "plan_max_age_s": 2.0,
            "start_tolerance_m": 0.10, "start_yaw_tolerance_rad": 0.3, "end_tolerance_m": 0.15,
            "stop_linear_mps": 0.02, "stop_angular_radps": 0.05,
            "stop_confirmation_s": 2.0, "stop_position_tolerance_m": 0.01,
            "tf_future_lead_s": 0.2,
            "max_maneuvers": 3, "turn_radius_m": 0.75, "turn_angle_rad": 0.3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        for name, value in defaults.items():
            if isinstance(value, float):
                actual = float(self._param(name))
                if not math.isfinite(actual) or actual <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        if self._param("stop_margin_m") >= self._param("max_travel_distance_m"):
            raise ValueError("stop margin must be less than travel limit")
        count = self._param("max_maneuvers")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 3:
            raise ValueError("max_maneuvers must be an integer from 1 to 3")
        if self._param("turn_radius_m") < 0.5 or self._param("turn_angle_rad") > 0.4:
            raise ValueError("turn radius must be >= Smac's 0.5 m and turn angle <= 0.4 rad")

        self._enabled = False
        self._epoch = 0
        self._attempted = False
        self._finished = False
        self._fault_latched = False
        self._sequence = None
        self._attempt_id = None
        self._operation = None
        self._validation = None
        self._target = None
        self._target_received = None
        self._target_turn_direction = 1
        self._odom_info_history = deque(maxlen=10)
        self._localization_barrier = None
        self._last_accepted_info_stamp = None
        self._last_accepted_map_tf_stamp = None
        self._localization_available = False
        self._localization_reason = None
        self._odom_stamp = self._odom_received = self._odom_velocity = None
        self._last_command = None
        self._scan_stamp = self._scan_received = self._scan_frame = None
        self._last_now = None
        self._wait_reason = None
        self._tf = Buffer()
        self._tf_receiver = AcquireTfReceiver(
            self._tf, context=self.context, namespace=self.get_namespace(),
            use_sim_time=bool(self.get_parameter("use_sim_time").value),
        )
        self._planner = ActionClient(self, ComputePathToPose, self._param("planner_action"))
        self._controller = ActionClient(self, FollowPath, self._param("controller_action"))
        self._validator = self.create_client(IsPathValid, self._param("path_valid_service"))
        self._result_pub = self.create_publisher(String, self._param("result_topic"), 10)
        self._point_sub = self.create_subscription(PointStamped, self._param("point_topic"), self._on_point, 10)
        self._info_sub = self.create_subscription(OdomInfo, self._param("odom_info_topic"), self._on_info, qos_profile_sensor_data)
        self._odom_sub = self.create_subscription(Odometry, self._param("odom_topic"), self._on_odom, qos_profile_sensor_data)
        self._command_sub = self.create_subscription(Twist, self._param("command_topic"), self._on_command, 10)
        self._scan_sub = self.create_subscription(LaserScan, self._param("scan_topic"), self._on_scan, qos_profile_sensor_data)
        self._service = self.create_service(SetBool, self._param("enable_service"), self._set_enabled)
        self._timer = self.create_timer(0.1, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _param(self, name):
        return self.get_parameter(name).value


    @staticmethod
    def _stamp(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _now(self):
        now = self.get_clock().now().nanoseconds
        if self._last_now is not None and now < self._last_now:
            self._target = self._target_received = None
            self._odom_info_history.clear()
            self._localization_barrier = None
            self._last_accepted_info_stamp = None
            self._last_accepted_map_tf_stamp = None
            self._localization_available = False
            self._localization_reason = "ROS clock moved backwards"
            self._odom_stamp = self._odom_received = self._odom_velocity = None
            self._last_command = None
            self._scan_stamp = self._scan_received = self._scan_frame = None
            self._validation = None
            self._tf.clear()
            if self._operation is not None and self._operation.get("terminal_follow"):
                self._reset_terminal_stop_window(self._operation)
            self._cancel("ROS clock moved backwards")
            self.get_logger().warning("Acquire invalidated data after clock regression")
        self._last_now = now
        return now

    @staticmethod
    def _fresh(stamp, receipt, now, limit, future_lead_s=0.0):
        if stamp is None or receipt is None or stamp <= 0 or now <= 0:
            return False
        return (-future_lead_s * 1e9 <= now - stamp <= limit * 1e9
                and 0 <= time.monotonic() - receipt <= limit)

    def _wait(self, reason):
        if reason != self._wait_reason:
            self.get_logger().info(f"Acquire waiting: {reason}")
            self._wait_reason = reason

    def _on_info(self, message):
        now = self._now()
        if message.lost:
            stamp = self._stamp(message.header.stamp)
            boundary = stamp if 0 < stamp <= now else None
            self._invalidate_localization("visual localization lost", now, boundary)
            self._target = self._target_received = None
            self._validation = None
            return
        stamp = self._stamp(message.header.stamp)
        if (stamp > 0 and (self._localization_barrier is None
                           or stamp > self._localization_barrier)):
            self._odom_info_history.append(stamp)

    def _invalidate_localization(self, reason, now, lost_stamp=None):
        if self._operation is not None and self._operation.get("terminal_follow"):
            self._reset_terminal_stop_window(self._operation)
        was_available = self._localization_available
        boundaries = [
            self._localization_barrier,
            self._last_accepted_info_stamp,
            lost_stamp,
            max(self._odom_info_history, default=None),
        ]
        if was_available and now > 0:
            boundaries.append(now)
        self._localization_barrier = max(
            (stamp for stamp in boundaries if stamp is not None), default=None
        )
        self._localization_available = False
        self._odom_info_history.clear()
        if was_available or self._localization_reason != reason:
            self.get_logger().warning(f"Acquire localization invalidated: {reason}")
        self._localization_reason = reason
        self._cancel(reason)

    def _on_odom(self, message):
        now = self._now()
        stamp = self._stamp(message.header.stamp)
        v = message.twist.twist
        velocity = (v.linear.x, v.linear.y, v.angular.z)
        operation = self._operation
        if (message.header.frame_id != self._param("odom_frame")
                or message.child_frame_id != self._param("base_frame")
                or stamp <= 0 or not all(math.isfinite(x) for x in velocity)):
            if operation is not None and operation.get("terminal_follow"):
                self._reset_terminal_stop_window(operation)
            self._odom_stamp = self._odom_received = self._odom_velocity = None
            self._cancel("invalid odometry")
            return
        if self._odom_stamp is not None and stamp <= self._odom_stamp:
            if operation is not None and operation.get("terminal_follow"):
                self._reset_terminal_stop_window(operation)
            return
        received = time.monotonic()
        self._odom_stamp, self._odom_received, self._odom_velocity = stamp, received, velocity
        if operation is not None and operation.get("terminal_follow"):
            try:
                odom_tf, odom_pose = self._robot_tf(self._param("odom_frame"), now)
                self._track_travel(odom_tf, odom_pose)
                tf_stamp = self._stamp(odom_tf.header.stamp)
                valid_tf = operation["terminal_stamp"] < tf_stamp
            except (TransformException, ValueError, TypeError):
                valid_tf = False
                odom_tf = None
                odom_pose = None
            command_stopped = (self._last_command is not None
                               and self._last_command[0] >= operation["started"]
                               and self._last_command[1])
            localized = self._localization_fresh(now)
            odom_fresh = (self._odom_velocity is not None and self._fresh(
                self._odom_stamp, self._odom_received, now,
                float(self._param("localization_max_age_s")),
                float(self._param("tf_future_lead_s"))))
            if (stamp <= operation["terminal_stamp"] or not valid_tf or not command_stopped
                    or not localized or not odom_fresh):
                self._reset_terminal_stop_window(operation)
            elif self._terminal_stop_sample(operation, stamp, tf_stamp, odom_pose, received):
                self._after_stop(operation)

    def _on_command(self, message):
        self._now()
        values = (message.linear.x, message.linear.y, message.linear.z,
                  message.angular.x, message.angular.y, message.angular.z)
        finite = all(math.isfinite(v) for v in values)
        self._last_command = (time.monotonic(), finite and all(abs(v) <= 1e-6 for v in values))
        if (self._operation is not None and self._operation.get("terminal_follow")
                and not self._last_command[1]):
            self._reset_terminal_stop_window(self._operation)
        if not finite:
            self._cancel("non-finite drive command")

    def _reset_terminal_stop_window(self, operation, tf_stamp=None, pose=None):
        window = operation.get("stop_window")
        if window is not None:
            if tf_stamp is None or tf_stamp <= window["last_tf_stamp"]:
                tf_stamp = window["last_tf_stamp"]
                pose = window.get("last_pose")
        if tf_stamp is not None and tf_stamp > 0:
            previous_floor = operation.get("stop_min_tf_stamp", 0)
            if tf_stamp >= previous_floor:
                operation["stop_min_tf_stamp"] = tf_stamp
                if pose is not None:
                    operation["stop_min_tf_pose"] = pose
        operation["stop_window"] = None

    def _terminal_stop_sample(self, operation, odom_stamp, tf_stamp, pose, wall):
        """Confirm terminal stop from one fresh, advancing odometry/TF sample."""
        limit_s = float(self._param("localization_max_age_s"))
        duration_s = float(self._param("stop_confirmation_s"))
        xy_limit = float(self._param("stop_position_tolerance_m"))
        linear_limit = float(self._param("stop_linear_mps"))
        angular_limit = float(self._param("stop_angular_radps"))
        if not all(math.isfinite(value) for value in pose):
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False
        window = operation.get("stop_window")
        if window is None:
            minimum_tf = operation.get("stop_min_tf_stamp")
            if minimum_tf is not None and tf_stamp <= minimum_tf:
                if (tf_stamp == minimum_tf and pose == operation.get("stop_min_tf_pose")):
                    return False
                self._reset_terminal_stop_window(operation, tf_stamp, pose)
                return False
            operation.pop("stop_min_tf_stamp", None)
            operation.pop("stop_min_tf_pose", None)
            operation["stop_window"] = {
                "start_odom_stamp": odom_stamp, "last_odom_stamp": odom_stamp,
                "start_tf_stamp": tf_stamp, "last_tf_stamp": tf_stamp,
                "start_wall": wall, "last_wall": wall,
                "anchor_xy": pose[:2], "last_yaw": pose[2],
                "last_pose": pose,
                "signed_yaw": 0.0,
                "yaw_variation": 0.0, "samples": 1,
            }
            return False

        odom_gap = odom_stamp - window["last_odom_stamp"]
        tf_gap = tf_stamp - window["last_tf_stamp"]
        wall_gap = wall - window["last_wall"]
        if (odom_gap <= 0 or tf_gap < 0 or wall_gap <= 0
                or odom_gap > limit_s * 1e9 or tf_gap > limit_s * 1e9
                or wall_gap > limit_s):
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False
        if tf_gap == 0:
            if pose == window["last_pose"]:
                return False
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False

        moved = math.hypot(pose[0] - window["anchor_xy"][0],
                           pose[1] - window["anchor_xy"][1])
        yaw_step = math.atan2(math.sin(pose[2] - window["last_yaw"]),
                              math.cos(pose[2] - window["last_yaw"]))
        variation = window["yaw_variation"] + abs(yaw_step)
        if moved > xy_limit:
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False
        fixed_angular_budget = angular_limit * duration_s
        if variation > fixed_angular_budget:
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False

        window["last_odom_stamp"] = odom_stamp
        window["last_tf_stamp"] = tf_stamp
        window["last_wall"] = wall
        window["last_pose"] = pose
        window["last_yaw"] = pose[2]
        window["signed_yaw"] += yaw_step
        window["yaw_variation"] = variation
        window["samples"] += 1
        elapsed_ros = (tf_stamp - window["start_tf_stamp"]) / 1e9
        elapsed_wall = wall - window["start_wall"]
        elapsed_odom = (odom_stamp - window["start_odom_stamp"]) / 1e9
        if elapsed_ros <= 0 or elapsed_wall <= 0:
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False
        if (window["samples"] < 3 or elapsed_ros < duration_s
                or elapsed_odom < duration_s or elapsed_wall < duration_s):
            return False
        linear_rate = moved / elapsed_ros
        angular_rate = abs(window["signed_yaw"]) / elapsed_ros
        if linear_rate > linear_limit or angular_rate > angular_limit:
            self._reset_terminal_stop_window(operation, tf_stamp, pose)
            return False

        return True

    def _on_scan(self, message):
        self._now()
        stamp = self._stamp(message.header.stamp)
        valid = (bool(message.header.frame_id) and math.isfinite(message.range_min)
                 and math.isfinite(message.range_max) and 0 <= message.range_min < message.range_max
                 and any((math.isfinite(r) and message.range_min <= r <= message.range_max)
                         or r == float("inf") for r in message.ranges))
        if not valid or stamp <= 0:
            self._scan_stamp = self._scan_received = self._scan_frame = None
            self._cancel("invalid scan")
        elif self._scan_stamp is None or stamp > self._scan_stamp:
            self._scan_stamp, self._scan_received = stamp, time.monotonic()
            self._scan_frame = message.header.frame_id

    def _on_point(self, message):
        now = self._now()
        stamp = self._stamp(message.header.stamp)
        if (not message.header.frame_id or stamp <= 0 or now <= 0 or stamp > now
                or now - stamp > float(self._param("observation_max_age_s")) * 1e9
                or not all(math.isfinite(v) for v in (message.point.x, message.point.y, message.point.z))):
            self._wait("invalid or stale suitcase observation")
            return
        if self._target is not None and stamp <= self._stamp(self._target.header.stamp):
            return
        if not self._localization_fresh(now):
            self._wait("localization unavailable for observation")
            return
        try:
            target = self._tf.transform(message, str(self._param("odom_frame")))
            point_in_base = self._tf.transform(message, str(self._param("base_frame")))
        except (TransformException, ValueError, TypeError) as exc:
            self._wait(f"observation TF unavailable: {exc}")
            return
        if not all(math.isfinite(v) for p in (target.point, point_in_base.point)
                   for v in (p.x, p.y, p.z)):
            self._wait("invalid transformed suitcase point")
            return
        target.header.stamp = message.header.stamp
        self._target, self._target_received = target, time.monotonic()
        self._target_turn_direction = -1 if point_in_base.point.y < 0 else 1
        self._wait_reason = None

    def _localization_fresh(self, now):
        now = int(now)
        max_age_ns = round(float(self._param("localization_max_age_s")) * 1e9)
        eligible = [
            stamp for stamp in self._odom_info_history
            if 0 < stamp <= now
            and (self._localization_barrier is None or stamp > self._localization_barrier)
            and (self._last_accepted_info_stamp is None
                 or stamp >= self._last_accepted_info_stamp)
        ]
        if not eligible:
            if self._localization_available:
                self._invalidate_localization("no current OdomInfo", now)
            return False

        info_stamp = max(eligible)
        if now - info_stamp > max_age_ns:
            self._invalidate_localization("OdomInfo is stale", now)
            return False

        try:
            transform, _ = self._robot_tf(self._param("map_frame"), now)
        except (TransformException, ValueError, TypeError) as exc:
            self._invalidate_localization(f"full TF unavailable: {exc}", now)
            return False

        validation_now = self._now()
        if validation_now < now:
            return False
        tf_stamp = self._stamp(transform.header.stamp)
        current_eligible = [
            stamp for stamp in self._odom_info_history
            if 0 < stamp <= validation_now
            and (self._localization_barrier is None
                 or stamp > self._localization_barrier)
            and (self._last_accepted_info_stamp is None
                 or stamp >= self._last_accepted_info_stamp)
        ]
        reason = None
        if info_stamp not in current_eligible or info_stamp != max(current_eligible, default=None):
            reason = "OdomInfo changed during localization validation"
        elif validation_now - info_stamp > max_age_ns:
            reason = "OdomInfo is stale"
        elif tf_stamp <= 0:
            reason = "full TF stamp is zero"
        elif tf_stamp > validation_now + round(float(self._param("tf_future_lead_s")) * 1e9):
            reason = "full TF stamp is in the future"
        elif validation_now - tf_stamp > max_age_ns:
            reason = "full TF is stale"
        elif (self._localization_barrier is not None
              and tf_stamp <= self._localization_barrier):
            reason = "full TF does not exceed localization barrier"
        elif (self._last_accepted_info_stamp is not None
              and info_stamp < self._last_accepted_info_stamp):
            reason = "OdomInfo timestamp regressed"
        elif (self._last_accepted_map_tf_stamp is not None
              and tf_stamp < self._last_accepted_map_tf_stamp):
            reason = "full TF timestamp regressed"
        if reason is not None:
            self._invalidate_localization(reason, validation_now)
            return False

        self._last_accepted_info_stamp = info_stamp
        self._last_accepted_map_tf_stamp = tf_stamp
        self._localization_available = True
        self._localization_reason = None
        return True

    def _stationary(self, now):
        if (not self._localization_fresh(now)
                or not self._fresh(self._odom_stamp, self._odom_received, now,
                                   float(self._param("localization_max_age_s")),
                                   float(self._param("tf_future_lead_s")))
                or self._odom_velocity is None):
            return False
        vx, vy, wz = self._odom_velocity
        return (math.hypot(vx, vy) <= float(self._param("stop_linear_mps"))
                and abs(wz) <= float(self._param("stop_angular_radps")))

    def _observation_usable(self, observation, now):
        stamp, receipt, x, y = observation
        remembered = (self._sequence is not None
                      and observation == self._sequence["observation"]
                      and self._sequence_failure() is None)
        return (self._target is not None
                and (remembered or self._fresh(stamp, receipt, now, float(self._param("observation_max_age_s"))))
                and math.hypot(self._target.point.x - x, self._target.point.y - y)
                <= float(self._param("max_target_shift_m")))

    def _sequence_failure(self):
        sequence = self._sequence
        if sequence is None:
            return None
        if sequence.get("failure"):
            return sequence["failure"]
        if time.monotonic() - sequence["started"] >= float(self._param("max_maneuver_time_s")):
            return "sequence time limit"
        if sequence["traveled"] >= float(self._param("max_travel_distance_m")) - float(self._param("stop_margin_m")):
            return "sequence travel limit with braking margin"
        return None

    def _track_travel(self, odom_tf, odom_pose):
        sequence = self._sequence
        if sequence is None:
            return
        stamp = self._stamp(odom_tf.header.stamp)
        if stamp < sequence["last_tf_stamp"]:
            sequence["failure"] = "odometry TF timestamp regressed"
        elif stamp > sequence["last_tf_stamp"]:
            sequence["traveled"] += math.dist(sequence["last_xy"], odom_pose[:2])
            sequence["last_xy"], sequence["last_tf_stamp"] = odom_pose[:2], stamp

    def _finish(self, outcome, reason):
        if self._finished or self._operation is not None:
            return
        self._fault_latched = self._fault_latched or outcome == "failed"
        if self._fault_latched:
            outcome = "failed"
        self._finished = self._attempted = True
        self._validation = None
        sequence = self._sequence
        result = {"attempt_id": self._attempt_id, "outcome": outcome, "reason": reason,
                  "completed_maneuvers": 0 if sequence is None else sequence["completed"],
                  "travel_m": 0.0 if sequence is None else sequence["traveled"]}
        message = String()
        message.data = json.dumps(result)
        self._result_pub.publish(message)
        self.get_logger().info(f"Acquire finished: {message.data}")

    def _after_stop(self, operation):
        self._operation = None
        if (operation["terminal_status"] == GoalStatus.STATUS_SUCCEEDED
                and operation["cancel_reason"] is None and self._enabled
                and not operation.get("fault", False)
                and operation["epoch"] == self._epoch):
            self._sequence["completed"] += 1
            failure = self._sequence_failure()
            if failure:
                self._finish("failed" if self._sequence.get("failure") else "blocked", failure)
            elif self._sequence["completed"] >= int(self._param("max_maneuvers")):
                self._finish("exhausted", "observation maneuvers completed")
            else:
                self._attempted = False
                self._wait_reason = None
        else:
            canceled = (operation["cancel_reason"] is not None
                        or operation["terminal_status"] == GoalStatus.STATUS_CANCELED
                        or not self._enabled)
            reason = operation["cancel_reason"] or ("Acquire disabled" if not self._enabled else "Nav2 follow failed")
            self._finish("failed" if operation.get("fault", False) or not canceled else "canceled", reason)

    def _robot_tf(self, frame, now):
        tf = self._tf.lookup_transform(str(frame), str(self._param("base_frame")), Time())
        stamp = self._stamp(tf.header.stamp)
        age = now - stamp
        if (stamp <= 0 or age > float(self._param("localization_max_age_s")) * 1e9
                or age < -float(self._param("tf_future_lead_s")) * 1e9
                or (self._localization_barrier is not None and stamp <= self._localization_barrier)):
            raise ValueError("robot TF stale or in the future")
        t, q = tf.transform.translation, tf.transform.rotation
        values = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite robot TF")
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1e-9:
            raise ValueError("invalid robot orientation")
        x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        return tf, (t.x, t.y, math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

    def _ready(self, now):
        if not self._localization_fresh(now):
            raise ValueError("localization unavailable or stale")
        if not self._fresh(self._scan_stamp, self._scan_received, now,
                           float(self._param("scan_max_age_s")),
                           float(self._param("tf_future_lead_s"))):
            raise ValueError("obstacle scan unavailable or stale")
        self._tf.lookup_transform(str(self._param("base_frame")), self._scan_frame,
                                  Time(nanoseconds=self._scan_stamp))
        map_tf, map_pose = self._robot_tf(self._param("map_frame"), now)
        odom_tf, odom_pose = self._robot_tf(self._param("odom_frame"), now)
        return map_tf, map_pose, odom_tf, odom_pose

    def _set_enabled(self, request, response):
        self._now()  # Observe a clock regression before acknowledging ownership release.
        if not request.data:
            if self._enabled:
                self._epoch += 1
            self._enabled = False
            self._validation = None
            self._cancel("Acquire disabled", fault=False)
            if self._attempt_id is not None and self._operation is None:
                self._finish("canceled", "Acquire disabled")
            response.success = self._operation is None
            response.message = "Acquire disabled" if response.success else "Waiting for terminal result and stop"
            response.message += f"; fault_latched={str(self._fault_latched).lower()}"
            return response
        if self._operation is not None:
            response.success = False
            response.message = "Acquire still owns a Nav2 operation"
            return response
        if not self._enabled:
            self._epoch += 1
            self._attempted = False
            self._finished = False
            self._fault_latched = False
            self._sequence = None
            self._attempt_id = uuid.uuid4().hex
        self._enabled = True
        response.success = True
        response.message = f"Acquire enabled; attempt_id={self._attempt_id}"
        return response

    def _tick(self):
        now = self._now()
        op = self._operation
        if op is not None and op.get("terminal_follow"):
            try:
                tf, pose = self._robot_tf(self._param("odom_frame"), now)
                self._track_travel(tf, pose)
            except (TransformException, ValueError, TypeError):
                pass  # Keep ownership until post-result stationary odometry.
            return
        if not self._enabled and (op is None or op["kind"] != "follow"):
            return
        if self._finished:
            return
        try:
            state = self._ready(now)
        except (TransformException, ValueError, TypeError) as exc:
            self._cancel(str(exc))
            self._wait(str(exc))
            if self._sequence is not None and op is None:
                self._finish("failed", str(exc))
            return
        self._track_travel(state[2], state[3])
        failure = self._sequence_failure()
        if (failure is None and self._sequence is not None
                and not self._observation_usable(self._sequence["observation"], now)):
            failure = "remembered suitcase position invalidated or changed"
        if failure:
            fault = bool(self._sequence and self._sequence.get("failure"))
            self._cancel(failure, fault=fault)
            if op is None:
                self._finish("failed" if fault else "blocked", failure)
            return
        if op is not None:
            return
        if self._validation is not None or self._attempted:
            return
        if self._sequence is None and (self._target is None or not self._fresh(self._stamp(self._target.header.stamp),
                self._target_received, now, float(self._param("observation_max_age_s")))):
            self._wait("no fresh remembered suitcase point")
            return
        if not self._stationary(now):
            self._wait("robot is not confirmed stationary")
            return
        if not self._planner.server_is_ready() or not self._controller.server_is_ready() or not self._validator.service_is_ready():
            self._wait("Nav2 planner, controller or path validator unavailable")
            return
        try:
            map_tf, robot = state[0], state[1]
            if self._sequence is None:
                self._sequence = {"target": copy.deepcopy(self._target),
                    "turn_direction": self._target_turn_direction,
                    "observation": (self._stamp(self._target.header.stamp), self._target_received,
                                    self._target.point.x, self._target.point.y),
                    "started": time.monotonic(), "traveled": 0.0, "completed": 0,
                    "last_xy": state[3][:2], "last_tf_stamp": self._stamp(state[2].header.stamp)}
            point = self._sequence["target"]
            remaining = int(self._param("max_maneuvers")) - self._sequence["completed"]
            travel_budget = (float(self._param("max_travel_distance_m"))
                             - float(self._param("stop_margin_m")) - self._sequence["traveled"])
            step_cap = min(float(self._param("max_goal_distance_m")), travel_budget / remaining)
            candidate_odom = observation_goal(state[3], (point.point.x, point.point.y),
                float(self._param("standoff_m")), step_cap,
                self._sequence["completed"], int(self._param("max_maneuvers")),
                float(self._param("turn_radius_m")), float(self._param("turn_angle_rad")),
                self._sequence["turn_direction"])
            if candidate_odom is None:
                outcome = "already_close" if self._sequence["completed"] == 0 else "exhausted"
                self._finish(outcome, "no forward maneuver fits observation clearance")
                return
            goal = PoseStamped()
            goal.header.frame_id = str(self._param("odom_frame"))
            goal.header.stamp = map_tf.header.stamp
            goal.pose.position.x, goal.pose.position.y = candidate_odom[:2]
            goal.pose.orientation.z = math.sin(candidate_odom[2]/2)
            goal.pose.orientation.w = math.cos(candidate_odom[2]/2)
            goal = self._tf.transform(goal, str(self._param("map_frame")))
            q = goal.pose.orientation
            candidate = (goal.pose.position.x, goal.pose.position.y,
                         math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z)))
        except (TransformException, ValueError, TypeError) as exc:
            self._finish("failed", f"target transform or goal unavailable: {exc}")
            return
        start = PoseStamped()
        start.header.frame_id = str(self._param("map_frame"))
        start.header.stamp = map_tf.header.stamp
        start.pose.position.x, start.pose.position.y = robot[:2]
        start.pose.orientation = map_tf.transform.rotation
        goal.header.stamp = self.get_clock().now().to_msg()
        request = ComputePathToPose.Goal()
        request.start, request.goal, request.use_start, request.planner_id = start, goal, True, "Hybrid"
        self._attempted = True
        self._send_action("plan", self._planner, request, candidate,
                          observation=self._sequence["observation"])

    def _send_action(self, kind, client, goal, candidate=None, observation=None):
        op = {"kind": kind, "epoch": self._epoch, "handle": None, "cancel_reason": None,
              "cancel_requested": False, "candidate": candidate, "observation": observation}
        if kind == "follow":
            op["started"] = time.monotonic()
        self._operation = op
        try:
            client.send_goal_async(goal).add_done_callback(lambda future: self._goal_response(op, future))
        except Exception as exc:
            self.get_logger().error(f"Acquire {kind} dispatch outcome unknown: {exc}")

    def _goal_response(self, op, future):
        if self._operation is not op:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Acquire {op['kind']} acceptance unknown: {exc}")
            return
        if not handle.accepted:
            self.get_logger().warning(f"Acquire {op['kind']} goal rejected")
            self._operation = None
            self._finish("failed" if op.get("fault", False) or not op["cancel_reason"] else "canceled",
                         f"Nav2 {op['kind']} rejected")
            return
        op["handle"] = handle
        self.get_logger().info(f"Acquire {op['kind']} goal accepted")
        try:
            handle.get_result_async().add_done_callback(lambda result: self._goal_result(op, result))
        except Exception as exc:
            self.get_logger().error(f"Acquire result monitor unavailable: {exc}")
            self._cancel("result monitor unavailable")
            return
        if op["cancel_reason"] is not None or not self._enabled or op["epoch"] != self._epoch:
            self._cancel(op["cancel_reason"] or "Acquire disabled", fault=op.get("fault", False))

    def _cancel(self, reason, fault=True):
        op = self._operation
        if fault and (op is not None or self._enabled and self._sequence is not None):
            self._fault_latched = True
        if op is None:
            return
        if fault:
            op["fault"] = True
        if op.get("terminal_follow"):
            return
        if op["cancel_reason"] is None:
            op["cancel_reason"] = reason
            self.get_logger().warning(f"Acquire cancel {op['kind']}: {reason}")
        if op["handle"] is not None and not op["cancel_requested"]:
            op["cancel_requested"] = True
            try:
                op["handle"].cancel_goal_async().add_done_callback(lambda future: self._cancel_response(op, future))
            except Exception as exc:
                self.get_logger().error(f"Acquire cancellation outcome unknown: {exc}")

    def _cancel_response(self, op, future):
        if self._operation is not op:
            return
        try:
            response = future.result()
            goal_id = bytes(op["handle"].goal_id.uuid)
            accepted = (response.return_code == CancelGoal.Response.ERROR_NONE
                        and any(bytes(info.goal_id.uuid) == goal_id for info in response.goals_canceling))
        except Exception as exc:
            self.get_logger().error(f"Acquire cancellation response unknown: {exc}")
            return
        if not accepted:
            self.get_logger().warning("Acquire cancellation not acknowledged; awaiting terminal result")
        else:
            self.get_logger().info("Acquire cancellation accepted; awaiting terminal result")

    def _goal_result(self, op, future):
        if self._operation is not op:
            return
        try:
            wrapped = future.result()
            status = wrapped.status
        except Exception as exc:
            self.get_logger().error(f"Acquire {op['kind']} result unknown: {exc}")
            self._cancel("result unknown")
            return
        names = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted",
                 GoalStatus.STATUS_CANCELED: "canceled"}
        if status not in names:
            self._cancel("nonterminal action result")
            return
        self.get_logger().info(f"Acquire {op['kind']} {names[status]}")
        if op["kind"] == "follow":
            op["terminal_follow"] = True
            op["terminal_status"] = status
            op["terminal_stamp"] = self._now()
            op["stop_window"] = None
            return
        self._operation = None
        if (status != GoalStatus.STATUS_SUCCEEDED or op["cancel_reason"] is not None
                or not self._enabled or op["epoch"] != self._epoch):
            self._finish("failed" if op.get("fault", False) else
                         "canceled" if op["cancel_reason"] or not self._enabled else "failed",
                         op["cancel_reason"] or "Nav2 planning failed")
            return
        try:
            now = self._now()
            state = self._ready(now)
        except (TransformException, ValueError, TypeError) as exc:
            self._finish("failed", f"plan readiness lost: {exc}")
            return
        try:
            if not self._observation_usable(op["observation"], now):
                raise ValueError("suitcase observation became stale or target moved")
            self._check_path(wrapped.result.path, op["candidate"], state[1], now, time.monotonic())
        except (TransformException, ValueError, TypeError, AttributeError) as exc:
            self._wait(f"plan rejected: {exc}")
            self._finish("blocked", f"plan rejected: {exc}")
            return
        if not self._validator.service_is_ready():
            self._wait("path validator unavailable")
            self._finish("blocked", "path validator unavailable")
            return
        validation = {"path": wrapped.result.path, "candidate": op["candidate"],
                      "epoch": self._epoch, "received": time.monotonic(), "observation": op["observation"]}
        self._validation = validation
        request = IsPathValid.Request()
        request.path = validation["path"]
        try:
            self._validator.call_async(request).add_done_callback(lambda result: self._validated(validation, result))
        except Exception as exc:
            self._validation = None
            self._wait(f"path validation dispatch failed: {exc}")
            self._finish("blocked", f"path validation dispatch failed: {exc}")

    def _check_path(self, path, candidate, robot, now, received):
        length = path_length(path, str(self._param("map_frame")))
        stamp = self._stamp(path.header.stamp)
        if not self._fresh(stamp, received, now, float(self._param("plan_max_age_s"))):
            raise ValueError("path is stale or has an invalid stamp")
        for pose in path.poses:
            pose_stamp = self._stamp(pose.header.stamp)
            if pose_stamp and not self._fresh(pose_stamp, received, now, float(self._param("plan_max_age_s"))):
                raise ValueError("path contains a stale or future pose stamp")
        if length <= 0 or length > float(self._param("max_path_length_m")):
            raise ValueError(f"path length {length:.3f} m exceeds limit or is zero")
        if self._sequence is not None:
            remaining = (float(self._param("max_travel_distance_m"))
                         - float(self._param("stop_margin_m")) - self._sequence["traveled"])
            if length > remaining:
                raise ValueError("path exceeds remaining sequence travel budget")
        start, end = path.poses[0].pose.position, path.poses[-1].pose.position
        if math.hypot(start.x - robot[0], start.y - robot[1]) > float(self._param("start_tolerance_m")):
            raise ValueError("path start no longer matches robot")
        q = path.poses[0].pose.orientation
        start_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        yaw_error = math.atan2(math.sin(start_yaw - robot[2]), math.cos(start_yaw - robot[2]))
        if abs(yaw_error) > float(self._param("start_yaw_tolerance_rad")):
            raise ValueError("path start orientation no longer matches robot")
        if math.hypot(end.x - candidate[0], end.y - candidate[1]) > float(self._param("end_tolerance_m")):
            raise ValueError("path end does not match requested goal")
        return length

    def _validated(self, validation, future):
        if self._validation is not validation:
            return
        self._validation = None
        if not self._enabled or validation["epoch"] != self._epoch or self._operation is not None:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._finish("failed", f"path validation result unavailable: {exc}")
            return
        if not response.is_valid or getattr(response, "invalid_pose_indices", []):
            self._finish("blocked", "Nav2 reports invalid path")
            return
        try:
            now = self._now()
            state = self._ready(now)
        except Exception as exc:
            self._finish("failed", f"validated path readiness lost: {exc}")
            return
        try:
            self._track_travel(state[2], state[3])
            if self._sequence is not None and self._sequence.get("failure"):
                self._finish("failed", self._sequence["failure"])
                return
            failure = self._sequence_failure()
            if failure:
                raise ValueError(failure)
            if not self._observation_usable(validation["observation"], now):
                raise ValueError("suitcase observation became stale or target moved")
            if not self._stationary(now):
                raise ValueError("robot is no longer stationary")
            length = self._check_path(validation["path"], validation["candidate"], state[1], now, validation["received"])
            if not self._controller.server_is_ready():
                raise ValueError("Nav2 controller unavailable")
        except Exception as exc:
            self._wait(f"validated path cannot start: {exc}")
            self._finish("blocked", f"validated path cannot start: {exc}")
            return
        request = FollowPath.Goal()
        request.path = validation["path"]
        request.controller_id = "FollowPath"
        request.goal_checker_id = str(self._param("goal_checker_id"))
        self.get_logger().info(f"Acquire following validated path length={length:.3f} m frame={request.path.header.frame_id}")
        self._send_action("follow", self._controller, request)


def main(args=None):
    rclpy.init(args=args)
    node = AcquireNode()
    try:
        node._tf_receiver.start()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._cancel("Acquire node shutdown")
        node._tf_receiver.close()
        node.destroy_node()
        try_shutdown()


if __name__ == "__main__":
    main()
