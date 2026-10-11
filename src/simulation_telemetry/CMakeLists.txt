cmake_minimum_required(VERSION 3.8)
project(simulation_telemetry)

find_package(ament_cmake REQUIRED)

install(PROGRAMS scripts/telemetry_recorder DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
