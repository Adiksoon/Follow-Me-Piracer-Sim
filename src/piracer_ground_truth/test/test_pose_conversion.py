import unittest
from types import SimpleNamespace

from geometry_msgs.msg import Pose
from std_msgs.msg import Header

from piracer_ground_truth.pose_conversion import pose_stamped_from_entity_response


class PoseConversionTest(unittest.TestCase):
    def test_preserves_gazebo_stamp_and_pose_in_world_frame(self):
        header = Header(frame_id='world')
        header.stamp.sec = 12
        header.stamp.nanosec = 345
        pose = Pose()
        pose.position.x = 1.25
        response = SimpleNamespace(
            success=True,
            header=header,
            state=SimpleNamespace(pose=pose),
        )

        message = pose_stamped_from_entity_response(response)

        self.assertEqual(message.header.frame_id, 'world')
        self.assertEqual(message.header.stamp.sec, 12)
        self.assertEqual(message.header.stamp.nanosec, 345)
        self.assertAlmostEqual(message.pose.position.x, 1.25)

    def test_rejects_a_different_reference_frame(self):
        response = SimpleNamespace(
            success=True,
            header=Header(frame_id='map'),
            state=SimpleNamespace(pose=Pose()),
        )

        with self.assertRaises(ValueError):
            pose_stamped_from_entity_response(response)

    def test_rejects_unsuccessful_gazebo_response(self):
        response = SimpleNamespace(success=False)

        with self.assertRaises(ValueError):
            pose_stamped_from_entity_response(response)


if __name__ == '__main__':
    unittest.main()
