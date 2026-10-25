#include <cmath>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

class AckermannTwistAdapter final : public rclcpp::Node
{
public:
  AckermannTwistAdapter()
  : Node("ackermann_twist_adapter")
  {
    wheelbase_m_ = declare_parameter<double>("wheelbase_m", 0.155);
    command_publisher_ = create_publisher<geometry_msgs::msg::Twist>(
      "ackermann_cmd_vel", 10);
    command_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10,
      [this](geometry_msgs::msg::Twist::ConstSharedPtr input)
      {
        geometry_msgs::msg::Twist output;
        output.linear.x = input->linear.x;

        const double speed_mps = input->linear.x;
        if (std::abs(speed_mps) > 1.0e-3) {
          // AckermannDrive applies the direction sign internally, so divide by
          // absolute speed to preserve the usual yaw-rate convention in reverse.
          output.angular.z = std::atan(
            wheelbase_m_ * input->angular.z / std::abs(speed_mps));
        }

        command_publisher_->publish(output);
      });
  }

private:
  double wheelbase_m_{0.155};
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr command_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AckermannTwistAdapter>());
  rclcpp::shutdown();
  return 0;
}
