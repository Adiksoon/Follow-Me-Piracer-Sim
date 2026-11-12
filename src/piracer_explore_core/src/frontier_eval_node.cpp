#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "explore/frontier_search.h"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/string.hpp"

class FrontierEvalNode final : public rclcpp::Node
{
public:
  FrontierEvalNode() : Node("piracer_explore_frontier_core")
  {
    map_topic_ = declare_parameter<std::string>("map_topic", "/nav2/map");
    pose_topic_ = declare_parameter<std::string>("pose_topic", "/piracer/search/robot_pose");
    result_topic_ = declare_parameter<std::string>("result_topic", "/piracer/search/frontiers");
    potential_scale_ = declare_parameter<double>("potential_scale", 3.0);
    gain_scale_ = declare_parameter<double>("gain_scale", 1.0);
    min_frontier_size_ = declare_parameter<double>("min_frontier_size", 0.5);
    result_pub_ = create_publisher<std_msgs::msg::String>(result_topic_, 10);
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, rclcpp::QoS(1).transient_local().reliable(),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) { map_ = msg; });
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      pose_topic_, rclcpp::QoS(10),
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { evaluate(*msg); });
  }

private:
  void evaluate(const geometry_msgs::msg::PoseStamped & pose)
  {
    if (!map_ || map_->info.width == 0 || map_->info.height == 0 ||
      map_->data.size() != static_cast<size_t>(map_->info.width) * map_->info.height)
    {
      return;
    }
    const auto & origin = map_->info.origin;
    const auto & q = origin.orientation;
    const double origin_yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    if (std::abs(origin_yaw) > 1e-6) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "m-explore frontier adapter requires an axis-aligned OccupancyGrid origin");
      return;
    }
    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(
      map_->info.width, map_->info.height, map_->info.resolution,
      origin.position.x, origin.position.y, nav2_costmap_2d::NO_INFORMATION);
    for (size_t i = 0; i < map_->data.size(); ++i) {
      const int8_t value = map_->data[i];
      const auto cost = value == 0 ? nav2_costmap_2d::FREE_SPACE :
        value == 100 ? nav2_costmap_2d::LETHAL_OBSTACLE :
        nav2_costmap_2d::NO_INFORMATION;
      costmap->setCost(static_cast<unsigned int>(i % map_->info.width),
        static_cast<unsigned int>(i / map_->info.width), cost);
    }
    frontier_exploration::FrontierSearch search(
      costmap.get(), potential_scale_, gain_scale_, min_frontier_size_, get_logger());
    geometry_msgs::msg::Point robot;
    robot.x = pose.pose.position.x;
    robot.y = pose.pose.position.y;
    const auto frontiers = search.searchFrom(robot);

    std::ostringstream json;
    json << std::setprecision(12);
    std::vector<std::string> encoded;
    for (size_t i = 0; i < frontiers.size(); ++i) {
      const auto & f = frontiers[i];
      std::ostringstream item;
      item << std::setprecision(12)
        << "{\"id\":\"frontier-" << i << "\","
        << "\"cluster_id\":\"frontier-" << i << "\","
        << "\"cluster_size\":" << f.size << ","
        << "\"score\":" << -f.cost << ","
        << "\"cost\":" << f.cost << ","
        << "\"min_distance\":" << f.min_distance << ","
        << "\"pose_world\":[" << f.centroid.x << "," << f.centroid.y << ",0],"
        << "\"frontiers_world\":[";
      for (size_t j = 0; j < f.points.size(); ++j) {
        if (j) {item << ",";}
        item << "[" << f.points[j].x << "," << f.points[j].y << "]";
      }
      item << "]}";
      encoded.push_back(item.str());
    }
    json << "{\"frame\":\"" << map_->header.frame_id << "\","
      << "\"map_stamp_ns\":" << stamp_ns(map_->header.stamp) << ","
      << "\"pose_stamp_ns\":" << stamp_ns(pose.header.stamp) << ","
      << "\"robot_xy\":[" << robot.x << "," << robot.y << "],"
      << "\"status\":\"" << (frontiers.empty() ? "NO_FRONTIERS" : "RECOMMENDATION_AVAILABLE")
      << "\",\"clustered\":true,\"candidates\":[";
    for (size_t i = 0; i < encoded.size(); ++i) {
      if (i) {json << ",";}
      json << encoded[i];
    }
    json << "],\"ranking\":[";
    for (size_t i = 0; i < frontiers.size(); ++i) {
      if (i) {json << ",";}
      json << "\"frontier-" << i << "\"";
    }
    json << "]}";
    std_msgs::msg::String message;
    message.data = json.str();
    result_pub_->publish(message);
  }

  static int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
  }

  std::string map_topic_, pose_topic_, result_topic_;
  double potential_scale_, gain_scale_, min_frontier_size_;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrontierEvalNode>());
  rclcpp::shutdown();
  return 0;
}
