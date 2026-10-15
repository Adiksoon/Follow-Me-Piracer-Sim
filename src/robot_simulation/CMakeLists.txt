cmake_minimum_required(VERSION 3.8)
project(robot_simulation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config worlds models DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/suitcase_dataset_generator.py DESTINATION lib/${PROJECT_NAME})

ament_package()
