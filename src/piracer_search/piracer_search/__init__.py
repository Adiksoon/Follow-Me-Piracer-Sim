"""ROS-independent PiRacer frontier search."""

from .search_algorithm import evaluate_search, world_to_cell

__all__ = ["evaluate_search", "world_to_cell"]
