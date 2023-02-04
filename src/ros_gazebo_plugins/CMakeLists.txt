cmake_minimum_required(VERSION 3.8)
project(ros_gazebo_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# Things below are added by me:
find_package(gazebo REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
# include_directories(${std_msgs_INCLUDE_DIRS})


# Added by me --------------------------------->
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

# list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")
# <--------------------------------- Added by me 

# Added by me --------------------------------->
add_library(ros_collision_detection SHARED src/CollisionDetectionPlugin.cc)
# add_library(ros_collision_detection SHARED src/CollisionDetectionPlugin.cc)
ament_target_dependencies(ros_collision_detection gazebo rclcpp gazebo_msgs gazebo_ros) # Added by me
# target_link_libraries(ros_collision_detection ${GAZEBO_LIBRARIES})
# <--------------------------------- Added by me 


target_include_directories(ros_collision_detection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(ros_collision_detection PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS ros_collision_detection
  DESTINATION lib/${PROJECT_NAME})


# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# add_executable(ros_collision_detection_plugin src/ros_collision_detection_plugin.cpp)
# ament_target_dependencies(ros_collision_detection_plugin rclcpp) # Added by me
# add_executable(ros_collision_detection_plugin src/CollisionDetectionPlugin.cc)
# target_include_directories(ros_collision_detection_plugin PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_compile_features(ros_collision_detection_plugin PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# install(TARGETS ros_collision_detection_plugin
#   DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
