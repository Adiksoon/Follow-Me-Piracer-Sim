cmake_minimum_required(VERSION 3.8)
project(piracer_gazebo_plugins)

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_library(piracer_kinematic_steering SHARED
  src/piracer_kinematic_steering.cpp
)
target_compile_features(piracer_kinematic_steering PUBLIC cxx_std_17)
target_include_directories(piracer_kinematic_steering PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
)
target_link_libraries(piracer_kinematic_steering ${GAZEBO_LIBRARIES})
ament_target_dependencies(piracer_kinematic_steering
  gazebo_dev
  gazebo_ros
  geometry_msgs
  rclcpp
)

install(TARGETS piracer_kinematic_steering
  LIBRARY DESTINATION lib
)
ament_export_dependencies(gazebo_dev gazebo_ros geometry_msgs rclcpp)
ament_package()
