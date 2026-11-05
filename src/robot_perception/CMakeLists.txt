cmake_minimum_required(VERSION 3.8)
project(robot_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/yolo11n_node
  scripts/suitcase_point_node
  scripts/obstacle_scan_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES LICENSE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_detection_pipeline test/test_detection_pipeline.py)
  ament_add_pytest_test(test_rgbd_geometry test/test_rgbd_geometry.py)
  ament_add_pytest_test(test_target_point test/test_target_point.py)
  ament_add_pytest_test(test_yolo_timing test/test_yolo_timing.py)
  ament_add_pytest_test(test_ros_messages test/test_ros_messages.py)
  ament_add_pytest_test(test_opencv_dnn test/test_opencv_dnn.py)
  ament_add_pytest_test(test_obstacle_scan test/test_obstacle_scan.py)
endif()

ament_package()
