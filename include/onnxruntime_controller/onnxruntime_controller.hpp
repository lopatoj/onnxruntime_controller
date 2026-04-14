#ifndef ONNXRUNTIME_CONTROLLER__ONNXRUNTIME_CONTROLLER_HPP_
#define ONNXRUNTIME_CONTROLLER__ONNXRUNTIME_CONTROLLER_HPP_

#include <controller_interface/controller_interface_base.hpp>
#include <cstddef>
#include <map>
#include <memory>
#include <rclcpp/generic_subscription.hpp>
#include <rclcpp/serialized_message.hpp>
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

#include "onnxruntime_controller/typed_interface.hpp"

namespace onnxruntime_controller {
/// Constant defining last action interface name
constexpr char HW_IF_LAST_ACTION[] = "last_action";

const std::array<std::string, 4> valid_joint_interfaces = {
    "position", "velocity", "effort", "acceleration"};

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
  on_export_reference_interfaces();

  bool on_set_chained_mode(bool chained) override;

  controller_interface::return_type
  update_reference_from_subscribers() override;

  controller_interface::return_type
  update_and_write_commands(const rclcpp::Time &time,
                            const rclcpp::Duration &period) override;

private:
  /**
   * @brief Processes an interface (either reference, action, or observation)
   * and returns the corresponding interface field names.
   *
   * @param interface_name The name of the interface to process.
   * @param interface_type The type of the interface (e.g., "position",
   * "velocity").
   * @param is_reference_interface Whether the interface is a reference
   * interface, to create a subscriber via TypedSubscriptionInterface.
   * @return A tuple containing the interface names and the callback return
   * status.
   */
  std::tuple<std::vector<std::string>, controller_interface::CallbackReturn>
  process_interface(std::string interface_name, std::string interface_type,
                    bool is_reference_interface);

  /**
   * @brief Validates an interface name for correctness.
   *
   * @param interface_name The name of the interface to validate.
   * @return A callback return status indicating success or failure.
   */
  controller_interface::CallbackReturn
  validate_interface_name(std::string &interface_name);

  std::shared_ptr<onnxruntime_controller::ParamListener> param_listener_;
  onnxruntime_controller::Params params_;

  std::vector<std::string> observation_interface_names_;
  std::vector<std::string> action_interface_names_;
  std::vector<std::string> reference_interface_names_;

  size_t num_observations_ = 0;
  size_t num_actions_ = 0;

  std::vector<std::shared_ptr<TypedSubscriptionInterface>> references_;

  std::string model_path_;
  Ort::Env env_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  const char *input_names_[1]{"obs"};
  const char *output_names_[1]{"actions"};

  std::vector<float> observations_;
  Ort::Value observations_tensor_;
  std::array<int64_t, 2> observations_shape_;

  std::map<size_t, size_t> reference_indices_;
  std::vector<size_t> last_actions_indices_;
  std::vector<size_t> state_indices_;

  std::vector<double> observation_scales_;
  std::vector<double> observation_offsets_;

  std::vector<float> actions_;
  Ort::Value actions_tensor_;
  std::array<int64_t, 2> actions_shape_;

  std::string actions_interface_;
  std::vector<std::string> joint_names_;
  double actions_scale_;
  float clip_actions_;
  std::vector<double> action_offsets_;
};

} // namespace onnxruntime_controller

#endif // ONNXRUNTIME_CONTROLLER__ONNXRUNTIME_CONTROLLER_HPP_
