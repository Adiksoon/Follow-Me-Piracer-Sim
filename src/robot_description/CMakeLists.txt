cmake_minimum_required(VERSION 3.8)
project(robot_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes third_party_licenses
  DESTINATION share/robot_description
)
install(FILES README.md DESTINATION share/robot_description)

ament_package()