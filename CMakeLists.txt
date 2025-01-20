cmake_minimum_required(VERSION 3.8)
project(tracker_prediction VERSION 0.0.1)

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(objects_msgs REQUIRED)

ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME})

#Устанавливаем исполняемый файл
install(PROGRAMS
  tracker/inference.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()
