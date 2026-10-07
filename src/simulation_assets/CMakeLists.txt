cmake_minimum_required(VERSION 3.8)
project(simulation_assets)

find_package(ament_cmake REQUIRED)

install(DIRECTORY assets DESTINATION share/simulation_assets)
install(FILES README.md DESTINATION share/simulation_assets)

ament_package()