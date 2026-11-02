cmake_minimum_required(VERSION 3.8)
project(piracer_gazebo_plugins)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(ackermann_twist_adapter
  src/ackermann_twist_adapter.cpp
)
target_compile_features(ackermann_twist_adapter PUBLIC cxx_std_17)
ament_target_dependencies(ackermann_twist_adapter
  geometry_msgs
  rclcpp
  std_msgs
)

install(TARGETS ackermann_twist_adapter
  DESTINATION lib/${PROJECT_NAME}
)
ament_export_dependencies(geometry_msgs rclcpp std_msgs)
ament_package()
