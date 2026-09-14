cmake_minimum_required(VERSION 3.5)
project(cleanrobot_amr_navigation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  params
  maps
  rviz2
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/cleanrobot_laser_self_filter.py
  scripts/cleanrobot_patrol_trash_scenario.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
