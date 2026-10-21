import math
from types import SimpleNamespace

import pytest

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from tf2_ros import LookupException

from piracer_navigation.initial_free_zone import apply_initial_free_zone
from piracer_navigation.start_zone_map import StartZoneMap


def make_grid(width, height, resolution=1.0, origin=(0.0, 0.0), yaw=0.0, data=None):
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.header.stamp.sec = 12
    grid.info.width = width
    grid.info.height = height
    grid.info.resolution = resolution
    grid.info.origin.position.x, grid.info.origin.position.y = origin
    grid.info.origin.orientation.z = math.sin(yaw / 2.0)
    grid.info.origin.orientation.w = math.cos(yaw / 2.0)
    grid.data = data or [-1] * (width * height)
    return grid


def test_rotated_translated_origin_uses_cell_centers_and_preserves_metadata():
    grid = make_grid(2, 2, origin=(10.0, 20.0), yaw=math.pi / 2)
    original = list(grid.data)
    # Cell (0, 0) center is (9.5, 20.5) after translating the rotated grid.
    result = apply_initial_free_zone(grid, 9.5, 20.5, 0.01)
    assert list(result.data) == [0, -1, -1, -1]
    assert list(grid.data) == original
    assert result.header == grid.header
    assert result.info == grid.info


def test_radius_boundary_is_inclusive_and_only_unknown_cells_change():
    grid = make_grid(3, 1, data=[-1, 25, 0])
    result = apply_initial_free_zone(grid, 1.0, 0.5, 0.5)
    assert list(result.data) == [0, 25, 0]


def test_non_unit_resolution_and_fixed_center_preserve_later_obstacle():
    grid = make_grid(4, 1, resolution=0.5)
    first = apply_initial_free_zone(grid, 0.5, 0.25, 0.5)
    assert list(first.data) == [0, 0, -1, -1]
    later = make_grid(4, 1, resolution=0.5, data=[-1, 100, -1, -1])
    result = apply_initial_free_zone(later, 0.5, 0.25, 0.5)
    assert list(result.data) == [0, 100, -1, -1]


@pytest.mark.parametrize(
    "grid, center, radius",
    [
        (make_grid(0, 1), (0.0, 0.0), 1.0),
        (make_grid(1, 1, resolution=0.0), (0.0, 0.0), 1.0),
            (make_grid(2, 1, data=[-1]), (0.0, 0.0), 1.0),
        (make_grid(1, 1), (float("nan"), 0.0), 1.0),
    ],
)
def test_invalid_grid_or_zone_is_rejected(grid, center, radius):
    with pytest.raises(ValueError):
        apply_initial_free_zone(grid, center[0], center[1], radius)


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class FakeClock:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def now(self):
        return SimpleNamespace(nanoseconds=self.nanoseconds)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def make_transform(x, y, stamp_ns):
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "base_link"
    transform.header.stamp.sec, transform.header.stamp.nanosec = divmod(
        stamp_ns, 1_000_000_000
    )
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.w = 1.0
    return transform


class FakeBuffer:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def lookup_transform(self, *_args):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_startup_waits_for_clock_and_fresh_tf_then_uses_latest_map_once():
    node = StartZoneMap()
    try:
        for endpoint in (node._publisher, node._subscription):
            assert endpoint.qos_profile.reliability == ReliabilityPolicy.RELIABLE
            assert endpoint.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
            assert endpoint.qos_profile.depth == 1
        publisher = FakePublisher()
        node._publisher = publisher
        node._latest_map = make_grid(2, 1)
        clock = FakeClock(0)
        node.get_clock = lambda: clock
        node._tf_buffer = FakeBuffer(
            [
                LookupException("not available yet"),
                make_transform(0.5, 0.5, 99_000_000_000),
                make_transform(0.5, 0.5, 99_900_000_000),
            ]
        )

        node._retry_initialization()  # Simulation clock has not started.
        assert node._tf_buffer.calls == 0
        assert not publisher.messages

        clock.nanoseconds = 100_000_000_000
        node._retry_initialization()  # No TF yet.
        node._retry_initialization()  # Stale TF is rejected.
        assert node._center is None
        assert not publisher.messages

        node._retry_initialization()  # Fresh TF captures the fixed center.
        assert node._center == (0.5, 0.5)
        assert list(publisher.messages[-1].data) == [0, 0]

        # Subsequent robot motion does not relocate the zone; a new obstacle wins.
        node._tf_buffer = FakeBuffer([make_transform(20.0, 20.0, 100_000_000_000)])
        node._on_map(make_grid(2, 1, data=[100, -1]))
        assert node._center == (0.5, 0.5)
        assert list(publisher.messages[-1].data) == [100, 0]
    finally:
        node.destroy_node()


def test_disabled_mode_passes_map_through_without_tf():
    node = StartZoneMap()
    try:
        node._enabled = False
        publisher = FakePublisher()
        node._publisher = publisher
        node._tf_buffer = FakeBuffer([AssertionError("TF must not be queried")])
        message = make_grid(1, 1)
        node._on_map(message)
        assert publisher.messages == [message]
        assert node._tf_buffer.calls == 0
        assert node._center is None
    finally:
        node.destroy_node()


def test_new_node_starts_without_previous_session_center():
    first = StartZoneMap()
    second = StartZoneMap()
    try:
        first._center = (5.0, 6.0)
        assert second._center is None
    finally:
        first.destroy_node()
        second.destroy_node()
