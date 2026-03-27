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
#include <limits>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <rcutils/allocator.h>
#include <rmw/rmw.h>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <rosidl_typesupport_introspection_cpp/field_types.hpp>
#include <string>
#include <vector>

namespace onnxruntime_controller {

ONNXRuntimeController::ONNXRuntimeController() {}

std::tuple<std::vector<std::string>, controller_interface::CallbackReturn>
ONNXRuntimeController::process_interface(std::string interface_name,
                                         std::string interface_type,
                                         bool is_reference_interface) {
  RCLCPP_INFO(get_node()->get_logger(), "Processing interface: %s, type: %s",
              interface_name.c_str(), interface_type.c_str());

  // Invalid configuration
  if (interface_name.empty() || interface_type.empty()) {
    return {std::vector<std::string>(),
            controller_interface::CallbackReturn::ERROR};
  }

  // Handle last action interface
  if (interface_name == HW_IF_LAST_ACTION) {
    if (is_reference_interface) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Last action interface cannot be a reference interface");
      return {std::vector<std::string>(),
              controller_interface::CallbackReturn::ERROR};
    }

    auto curr = observations_.size();
    for (auto i = 0U; i < num_actions_; ++i) {
      last_actions_indices_.push_back(curr + i);
    }
    observations_.resize(observations_.size() + num_actions_,
                         std::numeric_limits<double>::quiet_NaN());
    return {std::vector<std::string>(),
            controller_interface::CallbackReturn::SUCCESS};
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
    return {{interface_name}, controller_interface::CallbackReturn::SUCCESS};
  }

  if (is_reference_interface) {
    auto interface = std::make_shared<TypedSubscriptionInterface>(
        get_node(), interface_name, interface_type, params_.reference_timeout,
        reference_interfaces_);
    references_.emplace_back(interface);
    return {interface->get_interface_names(),
            controller_interface::CallbackReturn::SUCCESS};
  }

  auto interface = std::make_shared<TypedInterface>(get_node(), interface_name,
                                                    interface_type);
  return {interface->get_interface_names(),
          controller_interface::CallbackReturn::SUCCESS};
}

controller_interface::CallbackReturn
ONNXRuntimeController::validate_interface_name(std::string &interface_name) {
  auto slash_count =
      std::count(interface_name.begin(), interface_name.end(), '/');
  auto period_count =
      std::count(interface_name.begin(), interface_name.end(), '.');

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

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ONNXRuntimeController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = action_interface_names_;
  return config;
}

controller_interface::InterfaceConfiguration
ONNXRuntimeController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = observation_interface_names_;
  return config;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_configure(
    const rclcpp_lifecycle::State & /* previous_state */) {
  if (params_.joint_names.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joint names provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.observation_interfaces.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "No observation interfaces provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.observation_types.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No observation types provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.observation_interfaces.size() !=
      params_.observation_types.size()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Observation interfaces and types must have the same size.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.reference_interfaces.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No reference interfaces provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.reference_types.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No reference types provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.reference_interfaces.size() != params_.reference_types.size()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Reference interfaces and types must have the same size.");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (params_.model_path.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No model path provided.");
    return controller_interface::CallbackReturn::ERROR;
  }

  model_path_ = params_.model_path;
  joint_names_ = params_.joint_names;

  auto [actions_interface_names, ret] =
      process_interface(params_.actions_interface, "float64", false);

  if (ret != controller_interface::CallbackReturn::SUCCESS) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to configure actions interface.");
    return ret;
  }

  actions_interface_ = params_.actions_interface;
  action_interface_names_ = actions_interface_names;
  num_actions_ = action_interface_names_.size();
  actions_.resize(num_actions_, std::numeric_limits<double>::quiet_NaN());

  // Specify state and reference interfaces
  for (auto i = 0U; i < params_.reference_interfaces.size(); ++i) {
    auto ret = validate_interface_name(params_.reference_interfaces[i]);
    if (ret != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to configure reference interface: %s",
                   params_.reference_interfaces[i].c_str());
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
    reference_interface_names_.insert(reference_interface_names_.end(),
                                      interfaces.begin(), interfaces.end());
  }

  for (auto i = 0U; i < reference_interface_names_.size(); ++i) {
    RCLCPP_INFO(get_node()->get_logger(), "Reference interface: %s",
                reference_interface_names_[i].c_str());
  }

  for (auto i = 0U; i < params_.observation_interfaces.size(); ++i) {
    auto ret = validate_interface_name(params_.observation_interfaces[i]);
    if (ret != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to configure observations interface: %s",
                   params_.observation_interfaces[i].c_str());
      return ret;
    }

    auto [interfaces, ret2] = process_interface(
        params_.observation_interfaces[i], params_.observation_types[i], false);
    if (ret2 != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Observations interface invalid: %s",
                   params_.observation_interfaces[i].c_str());
      return ret2;
    }
    for (const auto &interface : interfaces) {
      if (std::find(reference_interface_names_.begin(),
                    reference_interface_names_.end(),
                    interface) != reference_interface_names_.end()) {
        reference_indices_.push_back(observations_.size());
        observations_.resize(observations_.size() + 1,
                             std::numeric_limits<double>::quiet_NaN());
      } else {
        state_indices_.push_back(observations_.size());
        observations_.resize(observations_.size() + 1,
                             std::numeric_limits<double>::quiet_NaN());
        observation_interface_names_.push_back(interface);
      }
    }
  }

  num_observations_ = observations_.size();

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING);
  session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(),
                                            Ort::SessionOptions{});
  allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

  // Configure tensors
  actions_shape_ = {static_cast<int64_t>(num_actions_)};
  actions_tensor_ = Ort::Value::CreateTensor<double>(
      allocator_->GetInfo(), actions_.data(), num_actions_,
      actions_shape_.data(), actions_shape_.size());
  observations_shape_ = {static_cast<int64_t>(num_observations_)};
  observations_tensor_ = Ort::Value::CreateTensor<double>(
      allocator_->GetInfo(), observations_.data(), num_observations_,
      observations_shape_.data(), observations_shape_.size());

  io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
  io_binding_->BindInput("observations", observations_tensor_);
  io_binding_->BindOutput("actions", actions_tensor_);

  return controller_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::CommandInterface>
ONNXRuntimeController::on_export_reference_interfaces() {
  std::vector<hardware_interface::CommandInterface> interfaces;

  RCLCPP_INFO(get_node()->get_logger(), "Exporting %lu reference interfaces",
              reference_interfaces_.size());

  for (auto &reference : references_) {
    auto reference_interfaces = reference->get_interfaces();
    for (auto &ref_interface : reference_interfaces) {
      interfaces.push_back(std::move(ref_interface));
    }
  }

  return interfaces;
}

controller_interface::CallbackReturn ONNXRuntimeController::on_activate(
    const rclcpp_lifecycle::State &previous_state) {
  return controller_interface::ChainableControllerInterface::on_activate(
      previous_state);
}

controller_interface::CallbackReturn ONNXRuntimeController::on_deactivate(
    const rclcpp_lifecycle::State &previous_state) {
  return controller_interface::ChainableControllerInterface::on_deactivate(
      previous_state);
}

controller_interface::CallbackReturn ONNXRuntimeController::on_cleanup(
    const rclcpp_lifecycle::State &previous_state) {
  return controller_interface::ChainableControllerInterface::on_cleanup(
      previous_state);
}

controller_interface::CallbackReturn
ONNXRuntimeController::on_error(const rclcpp_lifecycle::State &previous_state) {
  return controller_interface::ChainableControllerInterface::on_error(
      previous_state);
}

bool ONNXRuntimeController::on_set_chained_mode(bool /*chained*/) {
  return false;
}

controller_interface::return_type
ONNXRuntimeController::update_reference_from_subscribers(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  for (auto &reference : references_) {
    reference->update_from_subscriber();
  }
  return controller_interface::return_type::OK;
}

controller_interface::return_type
ONNXRuntimeController::update_and_write_commands(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  for (auto i = 0U; i < reference_indices_.size(); ++i) {
    observations_[reference_indices_[i]] = reference_interfaces_[i];
  }

  for (auto i = 0U; i < state_indices_.size(); ++i) {
    auto state_op = state_interfaces_[i].get_optional();
    if (state_op.has_value()) {
      observations_[state_indices_[i]] = state_op.value();
    }
  }

  for (auto i = 0U; i < last_actions_indices_.size(); ++i) {
    observations_[last_actions_indices_[i]] = actions_[i];
  }

  session_->Run(Ort::RunOptions{}, *io_binding_);

  bool set_actions = true;
  for (auto i = 0U; i < actions_.size(); ++i) {
    set_actions &= command_interfaces_[i].set_value(actions_[i]);
  }

  if (!set_actions) {
    RCLCPP_DEBUG_EXPRESSION(
        get_node()->get_logger(), !set_actions,
        "Unable to set an actions command interface. :(");
  }

  return controller_interface::return_type::OK;
}

} // namespace onnxruntime_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(onnxruntime_controller::ONNXRuntimeController,
                       controller_interface::ChainableControllerInterface)
