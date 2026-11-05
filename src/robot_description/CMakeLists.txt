cmake_minimum_required(VERSION 3.8)
project(robot_description)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_plugins REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(piracer_ackermann_drive SHARED src/piracer_ackermann_drive.cpp)
target_compile_features(piracer_ackermann_drive PRIVATE cxx_std_17)
target_include_directories(piracer_ackermann_drive PRIVATE
  ${GAZEBO_INCLUDE_DIRS} ${gazebo_plugins_INCLUDE_DIRS})
target_link_libraries(piracer_ackermann_drive ${GAZEBO_LIBRARIES})
ament_target_dependencies(piracer_ackermann_drive
  gazebo_ros nav_msgs std_msgs tf2_geometry_msgs tf2_ros)
install(TARGETS piracer_ackermann_drive LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_ackermann_stop test/test_ackermann_stop.cpp)
  target_compile_features(test_ackermann_stop PRIVATE cxx_std_17)
  add_test(NAME ackermann_stop COMMAND test_ackermann_stop)
endif()

install(DIRECTORY urdf meshes third_party_licenses
  DESTINATION share/robot_description
)
install(FILES README.md DESTINATION share/robot_description)

ament_package()
