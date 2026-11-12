cmake_minimum_required(VERSION 3.8)
project(piracer_explore_core)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

set(UPSTREAM_ROOT third_party/m-explore-ros2)
add_executable(frontier_eval_node
  src/frontier_eval_node.cpp
  ${UPSTREAM_ROOT}/src/frontier_search.cpp)
target_include_directories(frontier_eval_node PRIVATE
  ${UPSTREAM_ROOT}/include)
ament_target_dependencies(frontier_eval_node
  builtin_interfaces geometry_msgs nav2_costmap_2d nav_msgs rclcpp std_msgs)
install(TARGETS frontier_eval_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY third_party/m-explore-ros2 DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_frontier_search
    test/test_frontier_search.cpp
    ${UPSTREAM_ROOT}/src/frontier_search.cpp)
  target_include_directories(test_frontier_search PRIVATE ${UPSTREAM_ROOT}/include)
  ament_target_dependencies(test_frontier_search geometry_msgs nav2_costmap_2d rclcpp)
endif()

ament_package()
