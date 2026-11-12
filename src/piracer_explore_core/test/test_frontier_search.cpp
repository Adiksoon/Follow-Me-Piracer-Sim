#include <memory>

#include "explore/frontier_search.h"
#include "gtest/gtest.h"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"

TEST(FrontierSearchTest, UsesUpstreamClusterCentroidAndCost)
{
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(
    12, 12, 1.0, 0.0, 0.0, nav2_costmap_2d::NO_INFORMATION);
  for (unsigned int y = 1; y <= 5; ++y) {
    for (unsigned int x = 1; x <= 5; ++x) {
      costmap->setCost(x, y, nav2_costmap_2d::FREE_SPACE);
    }
  }

  frontier_exploration::FrontierSearch search(
    costmap.get(), 3.0, 1.0, 0.0, rclcpp::get_logger("upstream_frontier_test"));
  geometry_msgs::msg::Point robot;
  robot.x = 2.5;
  robot.y = 2.5;
  const auto frontiers = search.searchFrom(robot);

  ASSERT_EQ(frontiers.size(), 1u);
  EXPECT_EQ(frontiers.front().size, 20u);
  EXPECT_NEAR(frontiers.front().centroid.x, 3.475, 1e-9);
  EXPECT_NEAR(frontiers.front().centroid.y, 3.375, 1e-9);
  EXPECT_NEAR(frontiers.front().min_distance, 2.0, 1e-9);
  EXPECT_NEAR(frontiers.front().cost, -14.0, 1e-9);
}
