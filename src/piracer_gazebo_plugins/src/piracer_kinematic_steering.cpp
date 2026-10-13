#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

namespace
{
double Clamp(const double value, const double limit)
{
  return std::max(-limit, std::min(value, limit));
}

double MoveToward(const double current, const double target, const double max_delta)
{
  return current + Clamp(target - current, max_delta);
}

class PiracerKinematicSteering final : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = std::move(model);
    left_joint_ = model_->GetJoint(sdf->Get<std::string>("left_steering_joint"));
    right_joint_ = model_->GetJoint(sdf->Get<std::string>("right_steering_joint"));
    wheelbase_m_ = sdf->Get<double>("wheelbase_m");
    track_m_ = sdf->Get<double>("front_track_m");
    max_angle_rad_ = sdf->Get<double>("max_steering_angle_rad");
    max_rate_radps_ = sdf->Get<double>("max_steering_rate_radps");
    if (sdf->HasElement("trace_path")) {
      trace_path_ = sdf->Get<std::string>("trace_path");
    }

    if (!left_joint_ || !right_joint_ || wheelbase_m_ <= 0.0 || track_m_ <= 0.0 ||
        max_angle_rad_ <= 0.0 || max_rate_radps_ <= 0.0)
    {
      gzerr << "PiracerKinematicSteering: invalid joint or steering configuration\n";
      return;
    }

    if (!trace_path_.empty()) {
      trace_out_.open(trace_path_, std::ios::out | std::ios::trunc);
      if (!trace_out_) {
        gzerr << "PiracerKinematicSteering: cannot open adapter trace "
              << trace_path_ << "\n";
        return;
      }
      trace_out_ << std::setprecision(17);
      trace_out_.flush();
    }

    node_ = gazebo_ros::Node::Get(sdf);
    if (!node_)
    {
      gzerr << "PiracerKinematicSteering: failed to create ROS node\n";
      return;
    }

    command_subscription_ = node_->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", rclcpp::QoS(10),
      [this](geometry_msgs::msg::Twist::ConstSharedPtr message)
      {
        std::lock_guard<std::mutex> lock(command_mutex_);
        linear_speed_mps_ = message->linear.x;
        yaw_rate_radps_ = message->angular.z;
        ++command_sequence_;
      });

    desired_left_steering_rad_ = left_joint_->Position(0);
    desired_right_steering_rad_ = right_joint_->Position(0);
    last_sim_time_s_ = 0.0;
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&PiracerKinematicSteering::OnUpdate, this, std::placeholders::_1));
  }

private:
  void OnUpdate(const gazebo::common::UpdateInfo &info)
  {
    const double now_s = info.simTime.Double();
    const double dt_s = now_s - last_sim_time_s_;
    last_sim_time_s_ = now_s;
    if (!(dt_s > 0.0) || dt_s > 0.05) {
      return;
    }

    double speed_mps;
    double yaw_rate_radps;
    std::uint64_t command_sequence;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      speed_mps = linear_speed_mps_;
      yaw_rate_radps = yaw_rate_radps_;
      command_sequence = command_sequence_;
    }

    double left_target_unclamped = 0.0;
    double right_target_unclamped = 0.0;
    if (std::isfinite(speed_mps) && std::isfinite(yaw_rate_radps) &&
        std::abs(speed_mps) > 1e-6 && std::abs(yaw_rate_radps) > 1e-9)
    {
      const double turn_radius_m = speed_mps / yaw_rate_radps;
      left_target_unclamped = std::atan(wheelbase_m_ / (turn_radius_m - track_m_ / 2.0));
      right_target_unclamped = std::atan(wheelbase_m_ / (turn_radius_m + track_m_ / 2.0));
    }
    const double left_target = Clamp(left_target_unclamped, max_angle_rad_);
    const double right_target = Clamp(right_target_unclamped, max_angle_rad_);

    const double max_delta_rad = max_rate_radps_ * dt_s;
    const double left_current = left_joint_->Position(0);
    const double right_current = right_joint_->Position(0);
    const double left_next = MoveToward(desired_left_steering_rad_, left_target, max_delta_rad);
    const double right_next = MoveToward(desired_right_steering_rad_, right_target, max_delta_rad);
    desired_left_steering_rad_ = left_next;
    desired_right_steering_rad_ = right_next;
    bool left_set_called = false;
    bool right_set_called = false;
    bool left_set_result = false;
    bool right_set_result = false;

    // This is a kinematic steering abstraction, not force-based servo dynamics.
    if (std::abs(left_next - left_current) > 1e-12) {
      left_set_result = left_joint_->SetPosition(0, left_next, true);
      left_set_called = true;
    }
    if (std::abs(right_next - right_current) > 1e-12) {
      right_set_result = right_joint_->SetPosition(0, right_next, true);
      right_set_called = true;
    }

    const double left_after = left_joint_->Position(0);
    const double right_after = right_joint_->Position(0);
    const double left_rate_after = left_joint_->GetVelocity(0);
    const double right_rate_after = right_joint_->GetVelocity(0);
    if (trace_out_.is_open()) {
      trace_out_ << "{\"update\":" << ++update_index_
        << ",\"sim_time_s\":" << now_s
        << ",\"dt_s\":" << dt_s
        << ",\"command_sequence\":" << command_sequence
        << ",\"cmd_linear_x_mps\":" << speed_mps
        << ",\"cmd_yaw_rate_radps\":" << yaw_rate_radps
        << ",\"left_target_unclamped_rad\":" << left_target_unclamped
        << ",\"left_target_rad\":" << left_target
        << ",\"left_position_before_rad\":" << left_current
        << ",\"left_desired_position_rad\":" << desired_left_steering_rad_
        << ",\"left_requested_position_rad\":" << left_next
        << ",\"left_set_position_called\":" << (left_set_called ? "true" : "false")
        << ",\"left_set_position_result\":";
    if (left_set_called) trace_out_ << (left_set_result ? "true" : "false");
    else trace_out_ << "null";
    trace_out_
        << ",\"left_position_after_rad\":" << left_after
        << ",\"left_rate_after_radps\":" << left_rate_after
        << ",\"right_target_unclamped_rad\":" << right_target_unclamped
        << ",\"right_target_rad\":" << right_target
        << ",\"right_position_before_rad\":" << right_current
        << ",\"right_desired_position_rad\":" << desired_right_steering_rad_
        << ",\"right_requested_position_rad\":" << right_next
        << ",\"right_set_position_called\":" << (right_set_called ? "true" : "false")
        << ",\"right_set_position_result\":";
    if (right_set_called) trace_out_ << (right_set_result ? "true" : "false");
    else trace_out_ << "null";
    trace_out_
        << ",\"right_position_after_rad\":" << right_after
        << ",\"right_rate_after_radps\":" << right_rate_after << "}\n";
      trace_out_.flush();
    }
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::JointPtr left_joint_;
  gazebo::physics::JointPtr right_joint_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
  gazebo::event::ConnectionPtr update_connection_;
  std::mutex command_mutex_;
  std::ofstream trace_out_;
  std::string trace_path_;
  std::uint64_t command_sequence_{0};
  std::uint64_t update_index_{0};
  double linear_speed_mps_{0.0};
  double yaw_rate_radps_{0.0};
  double wheelbase_m_{0.155};
  double track_m_{0.148};
  double max_angle_rad_{0.04};
  double max_rate_radps_{0.10};
  double last_sim_time_s_{0.0};
  double desired_left_steering_rad_{0.0};
  double desired_right_steering_rad_{0.0};
};
}  // namespace

GZ_REGISTER_MODEL_PLUGIN(PiracerKinematicSteering)
