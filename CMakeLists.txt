cmake_minimum_required(VERSION 3.8)
project(onnxruntime_controller)

find_package(ros2_control_cmake REQUIRED)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(onnxruntime_vendor REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(rosidl_typesupport_introspection_cpp REQUIRED)

generate_parameter_library(onnxruntime_controller_parameters
  src/onnxruntime_controller_parameters.yaml
)

add_library(${PROJECT_NAME} SHARED
  src/onnxruntime_controller.cpp
  src/typed_interface.cpp
)
target_compile_features(onnxruntime_controller PUBLIC cxx_std_17)
target_include_directories(onnxruntime_controller PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(onnxruntime_controller PUBLIC
                      onnxruntime_controller_parameters
                      controller_interface::controller_interface
                      hardware_interface::hardware_interface
                      pluginlib::pluginlib
                      rclcpp::rclcpp
                      rclcpp_lifecycle::rclcpp_lifecycle
                      ${onnxruntime_vendor_LIBRARIES})

pluginlib_export_plugin_description_file(
  controller_interface onnxruntime_controller_plugins.xml)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY config launch urdf models
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS onnxruntime_controller onnxruntime_controller_parameters
  EXPORT export_onnxruntime_controller
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

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

ament_export_targets(export_onnxruntime_controller HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
