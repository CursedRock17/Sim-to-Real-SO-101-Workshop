cmake_minimum_required(VERSION 3.22)
project(so101_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config meshes DESTINATION share/${PROJECT_NAME})

ament_package()
