cmake_minimum_required(VERSION 3.8)
project(tracker_prediction)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)

find_package(std_msgs REQUIRED)

find_package(diagnostic_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(objects_msgs REQUIRED)

add_executable(tracker_node
              src/inference.cpp
              src/trajectory.cpp
              src/tracker.cpp
              src/utils.cpp)

target_include_directories(tracker_node PUBLIC
                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                          $<INSTALL_INTERFACE:include>
                          ${EIGEN3_INCLUDE_DIR})

target_compile_features(tracker_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(tracker_node
                          "rclcpp"
                          "Eigen3"
                          "tf2_ros"
                          "diagnostic_updater"
                          "diagnostic_msgs"
                          "objects_msgs"
                          "geometry_msgs")

install(TARGETS tracker_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  evaluation/evaluate_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
