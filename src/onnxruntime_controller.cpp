#include "onnxruntime_controller/onnxruntime_controller.hpp"
#include <controller_interface/chainable_controller_interface.hpp>
#include <controller_interface/controller_interface_base.hpp>
#include <cstddef>
#include <cstdint>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <iterator>
#include <rosidl_typesupport_introspection_cpp/field_types.hpp>
#include <vector>

namespace onnxruntime_controller {

std::tuple<std::vector<std::string>, controller_interface::CallbackReturn>
ONNXRuntimeController::process_interface(std::string interface_name,
                                         std::string interface_type,
                                         bool is_reference_interface) {
  // Invalid configuration
  if (interface_name.empty() || interface_type.empty()) {
    return {{}, controller_interface::CallbackReturn::ERROR};
  }

  if (interface_name == HW_IF_LAST_ACTION) {
    using_last_actions_ = true;
    actions_begin_ = initial_interfaces_.end();
    initial_interfaces_.resize(initial_interfaces_.size() + num_actions_);
    actions_end_ = interfaces_rt_.readFromNonRT()->end() + num_actions_;
    return {{}, controller_interface::CallbackReturn::SUCCESS};
  }

  if (interface_type == "float64") {
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
  std::vector<int32_t> interface_sizes;
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
    std::vector<int64_t> observation_indexes;

    for (auto &name : interface_names) {
      auto it = std::find(observations_interfaces_.begin(),
                          observations_interfaces_.end(), name);
      int64_t index = it != observations_interfaces_.end()
                          ? it - observations_interfaces_.begin()
                          : -1;

      // If the index is after the last actions interfaces, shift it to the
      // corresponding observation index
      if (actions_begin_ != actions_end_ &&
          index > actions_begin_ - initial_interfaces_.begin()) {
        index += num_actions_;
      }

      observation_indexes.push_back(index);
    }

    auto subscription_callback =
        [this, interface_sizes, interface_offsets, interface_names,
         observation_indexes](std::shared_ptr<rclcpp::SerializedMessage> msg) {
          std::vector<double> observations = *interfaces_rt_.readFromNonRT();
          for (size_t i = 0; i < interface_sizes.size(); ++i) {
            auto index = observation_indexes[i];
            if (index != -1) {
              // TODO: Make this work for non-double types
              observations[index] =
                  static_cast<double>(msg->get_rcl_serialized_message()
                                          .buffer[interface_offsets[i]]);
            }
          }
          interfaces_rt_.writeFromNonRT(observations);
        };

    subscriptions_.push_back(get_node()->create_generic_subscription(
        "~/" + interface_name, interface_type, rclcpp::SystemDefaultsQoS(),
        subscription_callback));
  }

  return {interface_names, controller_interface::CallbackReturn::SUCCESS};
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

  if (std::find(valid_action_interfaces.begin(), valid_action_interfaces.end(),
                params_.actions_interface) == valid_action_interfaces.end()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Actions interface not valid: %s",
                 params_.actions_interface.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Specify command, state, and reference interfaces

  for (const auto &joint_name : params_.joint_names) {
    actions_interfaces_.push_back(joint_name + "/" + params_.actions_interface);
  }

  num_actions_ = actions_interfaces_.size();

  for (size_t i = 0; i < params_.observations_interfaces.size(); ++i) {
    auto [interfaces, res] =
        process_interface(params_.observations_interfaces[i],
                          params_.observations_types[i], false);
    if (res != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Observations interface invalid: %s",
                   params_.observations_interfaces[i].c_str());
      return res;
    }
    if (interfaces.empty()) {
      continue;
    }
    observations_interfaces_.insert(observations_interfaces_.end(),
                                    interfaces.begin(), interfaces.end());
    initial_interfaces_.resize(initial_interfaces_.size() + interfaces.size());
    std::fill(initial_interfaces_.end() - interfaces.size(), initial_interfaces_.end(), 0.0);
  }

  num_observations_ = observations_interfaces_.size();

  // Adds actions interfaces to the end of initial interfaces
  if (!using_last_actions_) {
    actions_begin_ = initial_interfaces_.end();
    initial_interfaces_.resize(initial_interfaces_.size() + num_actions_);
    actions_end_ = interfaces_rt_.readFromNonRT()->end() + num_actions_;
  }

  for (size_t i = 0; i < params_.reference_interfaces.size(); ++i) {
    auto [interfaces, res] = process_interface(
        params_.reference_interfaces[i], params_.reference_types[i], true);
    if (res != controller_interface::CallbackReturn::SUCCESS) {
      RCLCPP_ERROR(get_node()->get_logger(), "Reference interface invalid: %s",
                   params_.reference_interfaces[i].c_str());
      return res;
    }
    if (interfaces.empty()) {
      continue;
    }
    reference_interfaces_.insert(reference_interfaces_.end(),
                                 interfaces.begin(), interfaces.end());
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ONNXRuntimeController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = actions_interfaces_;
  return config;
}

controller_interface::InterfaceConfiguration
ONNXRuntimeController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = observations_interfaces_;
  return config;
}

void ONNXRuntimeController::assign_interfaces(
    std::vector<hardware_interface::LoanedCommandInterface>
        &&command_interfaces,
    std::vector<hardware_interface::LoanedStateInterface> &&state_interfaces) {
  controller_interface::ChainableControllerInterface::assign_interfaces(
      std::move(command_interfaces), std::move(state_interfaces));

  auto command_it = command_interfaces.begin();
  auto state_it = state_interfaces.begin();

  for (auto it = initial_interfaces_.begin(); it != initial_interfaces_.end(); ++it) {
    if (it >= actions_begin_ && it < actions_end_) {
      command_it->
    } else {

    }
  }
}

controller_interface::CallbackReturn ONNXRuntimeController::on_configure(
    const rclcpp_lifecycle::State & /* previous_state */) {
  // Reserve space for interfaces buffer
  interfaces_rt_.initRT(initial_interfaces_);

  // Configure tensors
  actions_shape_ = {static_cast<int64_t>(num_observations_)};
  actions_ = Ort::Value::CreateTensor<double>(allocator_, actions_shape_.data(),
                                              actions_shape_.size());
  observations_shape_ = {static_cast<int64_t>(num_observations_)};
  observations_ = Ort::Value::CreateTensor<double>(
      allocator_, observations_shape_.data(), observations_shape_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::CommandInterface> ONNXRuntimeController::on_export_reference_interfaces() {
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (const auto &interface_name : reference_interfaces_) {
    bool in_observations = std::find(observations_interfaces_.begin(),
                                     observations_interfaces_.end(), interface_name) !=
                           observations_interfaces_.end();

    auto interface = hardware_interface::CommandInterface(
        interface_name, hardware_interface::HW_IF_VELOCITY, in_observations ?  : &unused_reference_interface_);

    interfaces.emplace_back();
  }
  return interfaces;
}

} // namespace onnxruntime_controller
