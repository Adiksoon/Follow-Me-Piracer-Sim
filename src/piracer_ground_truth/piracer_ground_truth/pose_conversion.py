from geometry_msgs.msg import PoseStamped


def pose_stamped_from_entity_response(response, expected_frame='world'):
    """Copy a successful GetEntityState sample without changing its pose time."""
    if not response.success:
        raise ValueError('Gazebo GetEntityState returned success=false')

    response_frame = response.header.frame_id
    if response_frame and response_frame != expected_frame:
        raise ValueError(
            f'Gazebo returned frame {response_frame!r}, expected {expected_frame!r}'
        )

    message = PoseStamped()
    message.header.stamp = response.header.stamp
    message.header.frame_id = expected_frame
    message.pose = response.state.pose
    return message
