import rclpy
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from piracer_ground_truth.pose_conversion import pose_stamped_from_entity_response


class GroundTruthPublisher(Node):
    def __init__(self):
        super().__init__('piracer_ground_truth_publisher')
        self.declare_parameter('service', '/piracer_ground_truth/get_entity_state')
        self.declare_parameter('reference_frame', 'world')
        self.declare_parameter('poll_rate_hz', 10.0)
        self.declare_parameter('robot_model', 'piracer')
        self.declare_parameter('suitcase_model', 'suitcase_target')
        self.declare_parameter('robot_topic', '/piracer/ground_truth/pose')
        self.declare_parameter('suitcase_topic', '/suitcase/ground_truth/pose')

        service = self.get_parameter('service').value
        self._frame = self.get_parameter('reference_frame').value
        self._client = self.create_client(GetEntityState, service)
        self._pending = set()

        self._targets = {}
        for model_parameter, topic_parameter in (
            ('robot_model', 'robot_topic'),
            ('suitcase_model', 'suitcase_topic'),
        ):
            model = self.get_parameter(model_parameter).value
            topic = self.get_parameter(topic_parameter).value
            self._targets[model] = self.create_publisher(PoseStamped, topic, 10)

        rate = float(self.get_parameter('poll_rate_hz').value)
        if rate <= 0.0:
            raise ValueError('poll_rate_hz must be greater than zero')
        self._timer = self.create_timer(1.0 / rate, self._poll_models)

    def _poll_models(self):
        if not self._client.service_is_ready():
            return

        for model in self._targets:
            if model in self._pending:
                continue
            request = GetEntityState.Request()
            request.name = model
            request.reference_frame = self._frame
            future = self._client.call_async(request)
            self._pending.add(model)
            future.add_done_callback(
                lambda completed, entity=model: self._handle_response(entity, completed)
            )

    def _handle_response(self, model, future):
        self._pending.discard(model)
        try:
            response = future.result()
            message = pose_stamped_from_entity_response(response, self._frame)
        except Exception as error:
            self.get_logger().warning(f'No ground truth sample for {model}: {error}')
            return
        self._targets[model].publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
