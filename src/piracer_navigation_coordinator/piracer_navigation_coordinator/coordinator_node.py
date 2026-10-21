"""Relay Search and Follow goals through one downstream Nav2 action client."""

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node


class _OwnerSession:
    def __init__(self, owner):
        self.owner = owner
        self.upstream_handle = None
        self.downstream_handle = None
        self.cancel_requested = False
        self.cancel_sent = False
        self.terminal_result_received = False


class NavigationCoordinator(Node):
    def __init__(self):
        super().__init__("piracer_navigation_coordinator")
        self._owner = None
        self._nav_action = ActionClient(
            self, NavigateToPose, "/navigate_to_pose"
        )
        self._search_server = ActionServer(
            self,
            NavigateToPose,
            "/navigation_coordinator/search",
            execute_callback=lambda handle: self._execute_goal("search", handle),
            goal_callback=lambda goal: self._accept_goal("search", goal),
            handle_accepted_callback=lambda handle: self._start_goal("search", handle),
            cancel_callback=self._cancel_goal,
        )
        self._follow_server = ActionServer(
            self,
            NavigateToPose,
            "/navigation_coordinator/follow",
            execute_callback=lambda handle: self._execute_goal("follow", handle),
            goal_callback=lambda goal: self._accept_goal("follow", goal),
            handle_accepted_callback=lambda handle: self._start_goal("follow", handle),
            cancel_callback=self._cancel_goal,
        )

    def _accept_goal(self, owner, _goal):
        if self._owner is not None:
            self.get_logger().warning(
                f"Rejecting {owner} goal; {self._owner.owner} currently owns navigation"
            )
            return GoalResponse.REJECT
        if not self._nav_action.server_is_ready():
            self.get_logger().warning("Rejecting goal; Nav2 NavigateToPose is not ready")
            return GoalResponse.REJECT
        self._owner = _OwnerSession(owner)
        return GoalResponse.ACCEPT

    def _start_goal(self, owner, upstream_handle):
        session = self._owner
        if session is None or session.owner != owner:
            self.get_logger().error("Accepted action goal has no matching owner session")
            upstream_handle.execute()
            return
        session.upstream_handle = upstream_handle
        upstream_handle.execute()

    def _cancel_goal(self, upstream_handle):
        session = self._owner
        if session is None or session.upstream_handle is not upstream_handle:
            return CancelResponse.REJECT
        session.cancel_requested = True
        self._request_downstream_cancel(session)
        return CancelResponse.ACCEPT

    async def _execute_goal(self, owner, upstream_handle):
        session = self._owner
        if (
            session is None
            or session.owner != owner
            or session.upstream_handle is not upstream_handle
        ):
            upstream_handle.abort()
            return NavigateToPose.Result()

        if not self._nav_action.server_is_ready():
            self.get_logger().error("Nav2 NavigateToPose became unavailable before forwarding")
            upstream_handle.abort()
            self._release_unforwarded_goal(session)
            return NavigateToPose.Result()

        try:
            downstream_handle = await self._nav_action.send_goal_async(
                upstream_handle.request,
                feedback_callback=lambda feedback: self._forward_feedback(
                    session, feedback
                ),
            )
        except Exception as exc:
            self.get_logger().error(f"Could not submit downstream goal: {exc}")
            upstream_handle.abort()
            # Keep ownership when submission outcome is uncertain.
            return NavigateToPose.Result()

        if not downstream_handle.accepted:
            self.get_logger().warning("Nav2 rejected the forwarded goal")
            upstream_handle.abort()
            self._release_unforwarded_goal(session)
            return NavigateToPose.Result()

        session.downstream_handle = downstream_handle
        if session.cancel_requested:
            self._request_downstream_cancel(session)

        try:
            downstream_result = await downstream_handle.get_result_async()
        except Exception as exc:
            self.get_logger().error(f"Could not receive downstream result: {exc}")
            upstream_handle.abort()
            # Keep the owner because the downstream goal may still be active.
            return NavigateToPose.Result()

        session.terminal_result_received = True
        if session.cancel_requested:
            # An accepted upstream cancel leaves its goal handle CANCELING.
            # Finish that transition only after Nav2 reports a terminal result.
            upstream_handle.canceled()
        elif downstream_result.status == GoalStatus.STATUS_SUCCEEDED:
            upstream_handle.succeed()
        elif downstream_result.status == GoalStatus.STATUS_CANCELED:
            upstream_handle.canceled()
        else:
            upstream_handle.abort()
        result = downstream_result.result
        self._release_terminal_goal(session)
        return result

    def _forward_feedback(self, session, feedback_message):
        if self._owner is session and session.upstream_handle is not None:
            session.upstream_handle.publish_feedback(feedback_message.feedback)

    def _request_downstream_cancel(self, session):
        if session.downstream_handle is None or session.cancel_sent:
            return
        session.cancel_sent = True
        session.downstream_handle.cancel_goal_async()

    def _release_unforwarded_goal(self, session):
        """Release after Nav2 confirms no downstream goal was accepted."""
        if self._owner is session:
            self._owner = None

    def _release_terminal_goal(self, session):
        if session.terminal_result_received and self._owner is session:
            self._owner = None


def main(args=None):
    rclpy.init(args=args)
    node = NavigationCoordinator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
