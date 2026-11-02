#include <cmath>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class AckermannTwistAdapter final : public rclcpp::Node
{
public:
  AckermannTwistAdapter()
  : Node("ackermann_twist_adapter")
  {
    wheelbase_m_ = declare_parameter<double>("wheelbase_m", 0.155);
    command_timeout_s_ = declare_parameter<double>("command_timeout_s", 0.5);
    require_behavior_owner_ = declare_parameter<bool>("require_behavior_owner", false);
    if (!std::isfinite(command_timeout_s_) || command_timeout_s_ <= 0.0) {
      throw std::invalid_argument("command_timeout_s must be finite and greater than zero");
    }

    command_publisher_ = create_publisher<geometry_msgs::msg::Twist>(
      "ackermann_cmd_vel", 10);
    command_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10,
      [this](geometry_msgs::msg::Twist::ConstSharedPtr input)
      {
        if (!require_behavior_owner_ || command_owner_ == "search") {
          accept_command(*input);
        }
      });
    follow_command_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel_follow", 10,
      [this](geometry_msgs::msg::Twist::ConstSharedPtr input)
      {
        if (require_behavior_owner_ && command_owner_ == "follow") {
          accept_command(*input);
        }
      });
    command_owner_subscription_ = create_subscription<std_msgs::msg::String>(
      "/piracer/state/command_owner",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](std_msgs::msg::String::ConstSharedPtr input)
      {
        if (!require_behavior_owner_) {
          return;
        }
        if (have_received_owner_ && input->data == command_owner_) {
          return;
        }
        command_owner_ = input->data;
        have_received_owner_ = true;
        have_received_command_ = false;
        command_timed_out_ = false;
        command_publisher_->publish(geometry_msgs::msg::Twist{});
      });
    timeout_timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      [this]()
      {
        if (!have_received_command_ || command_timed_out_ ||
          std::chrono::steady_clock::now() - last_command_time_ <=
          std::chrono::duration<double>(command_timeout_s_))
        {
          return;
        }

        command_publisher_->publish(geometry_msgs::msg::Twist{});
        command_timed_out_ = true;
        RCLCPP_WARN(
          get_logger(), "No cmd_vel received for %.2f s; publishing zero command",
          command_timeout_s_);
      });
  }

private:
  void accept_command(const geometry_msgs::msg::Twist & input)
  {
    last_command_time_ = std::chrono::steady_clock::now();
    have_received_command_ = true;
    command_timed_out_ = false;

    geometry_msgs::msg::Twist output;
    output.linear.x = input.linear.x;

    const double speed_mps = input.linear.x;
    if (std::abs(speed_mps) > 1.0e-3) {
      // AckermannDrive applies the direction sign internally, so divide by
      // absolute speed to preserve the usual yaw-rate convention in reverse.
      output.angular.z = std::atan(
        wheelbase_m_ * input.angular.z / std::abs(speed_mps));
    }

    command_publisher_->publish(output);
  }

  double wheelbase_m_{0.155};
  double command_timeout_s_{0.5};
  bool require_behavior_owner_{false};
  bool have_received_owner_{false};
  std::string command_owner_{"stop"};
  std::chrono::steady_clock::time_point last_command_time_{
    std::chrono::steady_clock::now()};
  bool have_received_command_{false};
  bool command_timed_out_{false};
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr command_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr follow_command_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_owner_subscription_;
  rclcpp::TimerBase::SharedPtr timeout_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AckermannTwistAdapter>());
  rclcpp::shutdown();
  return 0;
}
