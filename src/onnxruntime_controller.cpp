#include "onnxruntime_controller/onnxruntime_controller.hpp"
#include <algorithm>
#include <controller_interface/chainable_controller_interface.hpp>
#include <controller_interface/controller_interface_base.hpp>
#include <cstddef>
#include <cstdint>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <iterator>
#include <onnxruntime_cxx_api.h>
#include <rosidl_typesupport_introspection_cpp/field_types.hpp>
#include <vector>

namespace onnxruntime_controller {

ONNXRuntimeController::ONNXRuntimeController()
    : env_(nullptr), session_(nullptr), allocator_(nullptr),
      io_binding_(nullptr) {}

std::tuple<std::vector<std::string>, controller_interface::CallbackReturn>
ONNXRuntimeController::process_interface(std::string interface_name,
                                         std::string interface_type,
                                         bool is_reference_interface) {
  // Invalid configuration
  if (interface_name.empty() || interface_type.empty()) {
    return {{}, controller_interface::CallbackReturn::ERROR};
  }

  // If interface name is a valid joint interface, return joint-specific
  // interfaces
  if (std::find(valid_joint_interfaces.begin(), valid_joint_interfaces.end(),
                interface_name) != valid_joint_interfaces.end()) {
    std::vector<std::string> interfaces;
    for (auto &joint_name : joint_names_) {
      interfaces.push_back(joint_name + "/" + interface_name);
    }
    return {interfaces, controller_interface::CallbackReturn::SUCCESS};
  }

  // If interface type is float64, return a single interface
  if (interface_type == "float64") {
    // Requires interface name to include prefix
    if (std::count(interface_name.begin(), interface_name.end(), "/") != 1) {
      return {{}, controller_interface::CallbackReturn::ERROR};
    }

    return {{interface_name}, controller_interface::CallbackReturn::SUCCESS};
  }

  const rosidl_message_type_support_t ts =
      ::rosidl_get_zero_initialized_message_type_support_handle();
  ::get_message_typesupport_handle(&ts, interface_type.c_str());

  if (!ts.data) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to get message typesupport for interface type: %s",
                 interface_type.c_str());
    return {{}, controller_interface::CallbackReturn::ERROR};
  }

  auto members =
      static_cast<const rosidl_typesupport_introspection_cpp::MessageMembers *>(
          ts.data);

  std::vector<std::string> interface_names;
  std::vector<int32_t> interface_offsets;

  for (size_t i = 0; i < members->member_count_; ++i) {
    auto member = members->members_[i];

    if (member.is_array_ ||
        member.type_id_ ==
            rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE ||
        member.type_id_ ==
            rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Array, string or message field type not supported for "
                   "interface: %s/%s",
                   interface_name.c_str(), member.name_);
      return {{}, controller_interface::CallbackReturn::ERROR};
    } else {
      interface_names.push_back(interface_name + "." +
                                std::string(member.name_));
      interface_offsets.push_back(member.offset_);
    }
  }

  if (is_reference_interface) {
    int64_t reference_index_start = references_interface_names_.size();

    auto subscription_callback =
        [this, interface_offsets, interface_names, reference_index_start](
            std::shared_ptr<rclcpp::SerializedMessage> msg) {
          std::vector<double> references = *references_rt_.readFromNonRT();
          for (size_t i = 0; i < interface_offsets.size(); ++i) {
            // Get pointer to field value
            auto ptr =
                msg->get_rcl_serialized_message().buffer + interface_offsets[i];

            // TODO: Make this work for non-double types
            // Assign to references vector
            references[reference_index_start + i] =
                static_cast<double>(*((double *)ptr));
          }
          references_rt_.writeFromNonRT(references);
        };

    subscriptions_.push_back(get_node()->create_generic_subscription(
        "~/" + interface_name, interface_type, rclcpp::SystemDefaultsQoS(),
        subscription_callback));
  }

  return {interface_names, controller_interface::CallbackReturn::SUCCESS};
}

controller_interface::CallbackReturn
ONNXRuntimeController::validate_interface_name(std::string &interface_name) {
  auto slash_count =
      std::count(interface_name.begin(), interface_name.end(), "/");
  auto period_count =
      std::count(interface_name.begin(), interface_name.end(), ".");

  if (slash_count > 1 || period_count > 1) {
    RCLCPP_ERROR(get_node()->get_logger(), "Invalid interface name: %s",
                 interface_name.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Exception thrown during init stage with message: %s \n",
                 e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Validate parameters

  if (params_.joint_names.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joint names provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  joint_names_ = params_.joint_names;

  auto [actions_interface_names, ret] =
      process_interface(params_.actions_interface, "", false);

  if (ret != controller_interface::CallbackReturn::SUCCESS) {
    return ret;
  }

  actions_interface_ = params_.actions_interface;
  actions_interface_names_ = actions_interface_names;
  num_actions_ = actions_interface_names_.size();

  // Specify command, state, and reference interfaces

  for (size_t i = 0; i < params_.observations_interfaces.size(); ++i) {
    auto ret = validate_interface_name(params_.observations_interfaces[i]);
    if (ret != controller_interface::CallbackReturn::SUCCESS) {
      return ret;
    }

    auto [interfaces, ret2] =
        process_interface(params_.observations_interfaces[i],
                          params_.observations_types[i], false);
    if (ret2 != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Observations interface invalid: %s",
                   params_.observations_interfaces[i].c_str());
      return ret2;
    }
    if (interfaces.empty()) {
      continue;
    }
    observations_interface_names_.insert(observations_interface_names_.end(),
                                         interfaces.begin(), interfaces.end());
  }

  num_observations_ = observations_interface_names_.size();

  for (size_t i = 0; i < params_.reference_interfaces.size(); ++i) {
    auto ret = validate_interface_name(params_.reference_interfaces[i]);
    if (ret != controller_interface::CallbackReturn::SUCCESS) {
      return ret;
    }

    auto [interfaces, ret2] = process_interface(
        params_.reference_interfaces[i], params_.reference_types[i], true);
    if (ret2 != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(), "Reference interface invalid: %s",
                   params_.reference_interfaces[i].c_str());
      return ret2;
    }
    if (interfaces.empty()) {
      continue;
    }
    references_interface_names_.insert(references_interface_names_.end(),
                                       interfaces.begin(), interfaces.end());
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ONNXRuntimeController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = actions_interface_names_;
  return config;
}

controller_interface::InterfaceConfiguration
ONNXRuntimeController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = observations_interface_names_;
  return config;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_configure(
    const rclcpp_lifecycle::State & /* previous_state */) {

  actions_.resize(num_actions_, std::numeric_limits<double>::quiet_NaN());
  observations_.resize(num_observations_,
                       std::numeric_limits<double>::quiet_NaN());

  // Configure tensors
  actions_shape_ = {static_cast<int64_t>(num_observations_)};
  actions_tensor_ = Ort::Value::CreateTensor<double>(
      allocator_, actions_shape_.data(), actions_shape_.size());
  observations_shape_ = {static_cast<int64_t>(num_observations_)};
  observations_tensor_ = Ort::Value::CreateTensor<double>(
      allocator_, observations_shape_.data(), observations_shape_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::CommandInterface>
ONNXRuntimeController::on_export_reference_interfaces() {
  // TODO: Switch to using shared pointers to interfaces w/
  // on_export_reference_interfaces_list in future ros2_control version

  std::vector<hardware_interface::CommandInterface> interfaces;

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto &joint_name = joint_names_[i];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_name, actions_interface_, &actions_[i]));
#pragma GCC diagnostic pop
  }

  references_.resize(references_interface_names_.size(),
                     std::numeric_limits<double>::quiet_NaN());

  for (size_t i = 0; i < references_interface_names_.size(); ++i) {
    const auto &interface_name = references_interface_names_[i];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    interfaces.emplace_back(hardware_interface::CommandInterface(
        get_node()->get_name(), interface_name, &references_[i]));
#pragma GCC diagnostic pop
  }
  return interfaces;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_cleanup(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
ONNXRuntimeController::on_error(const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::SUCCESS;
}

bool ONNXRuntimeController::on_set_chained_mode(bool /*chained*/) { return true; }

controller_interface::return_type
ONNXRuntimeController::update_reference_from_subscribers(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  return controller_interface::return_type::OK;
}

controller_interface::return_type
ONNXRuntimeController::update_and_write_commands(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  return controller_interface::return_type::OK;
}

} // namespace onnxruntime_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(onnxruntime_controller::ONNXRuntimeController,
                       controller_interface::ChainableControllerInterface)
