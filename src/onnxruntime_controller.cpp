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

ONNXRuntimeController::ONNXRuntimeController() : session_(nullptr) {}

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
    observations_.resize(observations_.size() + num_actions_, 0.0f);
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
          params_.observation_types.size() ||
      (params_.observation_interfaces.size() !=
           params_.observation_scales.size() &&
       !params_.observation_scales.empty())) {
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

  if (params_.action_offsets.size() != params_.joint_names.size() &&
      !params_.action_offsets.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Action offsets and joint names must have the same size.");
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
  actions_.resize(num_actions_, 0.0);
  clip_actions_ = params_.clip_actions == 0.0
                      ? std::numeric_limits<float>::infinity()
                      : params_.clip_actions;
  action_offsets_ = params_.action_offsets.empty()
                        ? std::vector<double>(num_actions_, 0.0)
                        : params_.action_offsets;
  actions_scale_ = params_.actions_scale;

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
      std::vector<std::string>::iterator reference_index =
          std::find(reference_interface_names_.begin(),
                    reference_interface_names_.end(), interface);
      if (reference_index != reference_interface_names_.end()) {
        reference_indices_[reference_index -
                           reference_interface_names_.begin()] =
            observations_.size();
        observations_.resize(observations_.size() + 1, 0.0f);
      } else {
        state_indices_.push_back(observations_.size());
        observations_.resize(observations_.size() + 1, 0.0f);
        observation_interface_names_.push_back(interface);
      }
      observation_scales_.push_back(params_.observation_scales.empty()
                                        ? 1.0
                                        : params_.observation_scales[i]);
    }
  }

  num_observations_ = observations_.size();

  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;

  Ort::SessionOptions options;
  options.AppendExecutionProvider_CUDA(cuda_options);

  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
  session_ = Ort::Session(env_, model_path_.c_str(), options);
  allocator_ = Ort::AllocatorWithDefaultOptions();

  // Configure tensors
  actions_shape_ = {1, static_cast<int64_t>(num_actions_)};
  actions_tensor_ = Ort::Value::CreateTensor<float>(
      allocator_.GetInfo(), actions_.data(), num_actions_,
      actions_shape_.data(), actions_shape_.size());
  observations_shape_ = {1, static_cast<int64_t>(num_observations_)};
  observations_tensor_ = Ort::Value::CreateTensor<float>(
      allocator_.GetInfo(), observations_.data(), num_observations_,
      observations_shape_.data(), observations_shape_.size());

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
ONNXRuntimeController::update_reference_from_subscribers() {
  for (auto &reference : references_) {
    reference->update_from_subscriber();
  }
  return controller_interface::return_type::OK;
}

controller_interface::return_type
ONNXRuntimeController::update_and_write_commands(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  for (auto [r_idx, o_idx] : reference_indices_) {
    observations_[o_idx] =
        reference_interfaces_[r_idx] * observation_scales_[o_idx];
  }

  for (auto i = 0U; i < state_indices_.size(); ++i) {
    auto index = state_indices_[i];
    observations_[index] = state_interfaces_[i].get_value() * observation_scales_[index];
  }

  for (auto i = 0U; i < last_actions_indices_.size(); ++i) {
    auto index = last_actions_indices_[i];
    observations_[index] = std::isfinite(actions_[i])
                               ? actions_[i] * observation_scales_[index]
                               : 0.0f;
  }

  session_.Run(Ort::RunOptions{}, input_names_, &observations_tensor_, 1,
               output_names_, &actions_tensor_, 1);

  for (auto i = 0U; i < actions_.size(); ++i) {
    RCLCPP_INFO(get_node()->get_logger(), "Action %u: %f", i, actions_[i]);
  }

  // bool actions_set = true;
  for (auto i = 0U; i < actions_.size(); ++i) {
    actions_[i] = std::clamp(actions_[i], -clip_actions_, clip_actions_);
    command_interfaces_[i].set_value(
        (actions_[i] * actions_scale_) + action_offsets_[i]);
  }

  // if (!actions_set) {
  //   RCLCPP_DEBUG_EXPRESSION(get_node()->get_logger(), !actions_set,
  //                           "Unable to set an actions command interface. :(");
  // }

  return controller_interface::return_type::OK;
}

} // namespace onnxruntime_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(onnxruntime_controller::ONNXRuntimeController,
                       controller_interface::ChainableControllerInterface)
