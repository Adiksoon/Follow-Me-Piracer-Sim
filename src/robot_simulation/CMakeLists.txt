cmake_minimum_required(VERSION 3.8)
project(robot_simulation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config worlds models DESTINATION share/${PROJECT_NAME})
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/nvidia-rendering.dsv")

ament_package()
