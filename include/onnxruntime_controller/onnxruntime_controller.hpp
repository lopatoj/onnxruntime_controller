#ifndef ONNXRUNTIME_CONTROLLER__ONNXRUNTIME_CONTROLLER_HPP_
#define ONNXRUNTIME_CONTROLLER__ONNXRUNTIME_CONTROLLER_HPP_

#include <controller_interface/controller_interface_base.hpp>
#include <map>
#include <memory>
#include <rclcpp/generic_subscription.hpp>
#include <string>
#include <vector>

#include "controller_interface/chainable_controller_interface.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "rosidl_runtime_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"

#include "onnxruntime_cxx_api.h"

#include "onnxruntime_controller/onnxruntime_controller_parameters.hpp"

namespace onnxruntime_controller {
/// Constant defining last action interface name
constexpr char HW_IF_LAST_ACTION[] = "last_action";

const std::map<std::string, std::string> message_to_hw_if = {
    {"geometry_msgs/msg/Pose", hardware_interface::HW_IF_POSITION},
    {"geometry_msgs/msg/Pose2D", hardware_interface::HW_IF_POSITION},
    {"geometry_msgs/msg/Twist", hardware_interface::HW_IF_VELOCITY},
    {"geometry_msgs/msg/Accel", hardware_interface::HW_IF_ACCELERATION},
    {"geometry_msgs/msg/Wrench", hardware_interface::HW_IF_EFFORT},
    {"geometry_msgs/msg/Point", hardware_interface::HW_IF_POSITION},
    {"geometry_msgs/msg/Quaternion", hardware_interface::HW_IF_POSITION},
};

const std::array<std::string, 5> valid_joint_interfaces = {
    hardware_interface::HW_IF_POSITION, hardware_interface::HW_IF_VELOCITY,
    hardware_interface::HW_IF_EFFORT, hardware_interface::HW_IF_ACCELERATION,
    HW_IF_LAST_ACTION};

class ONNXRuntimeController
    : public controller_interface::ChainableControllerInterface {
public:
  ONNXRuntimeController();

  controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_error(const rclcpp_lifecycle::State &previous_state) override;

protected:
  std::vector<hardware_interface::CommandInterface>
  on_export_reference_interfaces() override;
  bool on_set_chained_mode(bool chained) override;

  controller_interface::return_type
  update_reference_from_subscribers(const rclcpp::Time &time,
                                    const rclcpp::Duration &period) override;

  controller_interface::return_type
  update_and_write_commands(const rclcpp::Time &time,
                            const rclcpp::Duration &period) override;

private:
  /**
   * @brief Processes an interface (either reference or action) and returns the
   *        corresponding interface field names.
   *
   * @param interface_name The name of the interface to process.
   * @param interface_type The type of the interface (e.g., "position",
   * "velocity").
   * @param is_reference_interface Whether the interface is a reference
   * interface, to create subscribers.
   * @return A tuple containing the interface names and the callback return
   * status.
   */
  std::tuple<std::vector<std::string>, controller_interface::CallbackReturn>
  process_interface(std::string interface_name, std::string interface_type,
                    bool is_reference_interface);

  controller_interface::CallbackReturn
  validate_interface_name(std::string &interface_name);

  std::shared_ptr<onnxruntime_controller::ParamListener> param_listener_;
  onnxruntime_controller::Params params_;

  std::string actions_interface_;
  std::vector<std::string> joint_names_;

  size_t num_actions_ = 0;
  size_t num_observations_ = 0;

  std::vector<std::string> observations_interface_names_;
  std::vector<std::string> actions_interface_names_;
  std::vector<std::string> references_interface_names_;

  std::vector<double> observations_;
  std::vector<double> actions_;
  std::vector<double> references_;

  realtime_tools::RealtimeBuffer<std::vector<double>> observations_rt_;
  realtime_tools::RealtimeBuffer<std::vector<double>> actions_rt_;
  realtime_tools::RealtimeBuffer<std::vector<double>> references_rt_;

  std::vector<rclcpp::GenericSubscription::SharedPtr> subscriptions_;

  Ort::Env env_;
  Ort::Session session_;
  Ort::Allocator allocator_;
  Ort::IoBinding io_binding_;

  Ort::Value observations_tensor_;
  std::array<int64_t, 1> observations_shape_;

  Ort::Value actions_tensor_;
  std::array<int64_t, 1> actions_shape_;
};

} // namespace onnxruntime_controller

#endif // ONNXRUNTIME_CONTROLLER__ONNXRUNTIME_CONTROLLER_HPP_
