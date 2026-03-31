#include <controller_interface/controller_interface_base.hpp>
#include <hardware_interface/handle.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <rosidl_runtime_c/message_type_support_struct.h>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <rosidl_typesupport_introspection_cpp/field_types.hpp>
#include <string>
#include <vector>

namespace onnxruntime_controller {
class TypedInterface {
public:
  TypedInterface(std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node,
                 std::string name, std::string type);

  const std::vector<std::string> &get_interface_names() const;

  controller_interface::CallbackReturn status() const;

protected:
  rosidl_message_type_support_t *
  get_type_support_handle(std::string ts_library, std::string type,
                          std::shared_ptr<rcpputils::SharedLibrary> &library);

  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;

  rosidl_message_type_support_t ts_members_;
  const rosidl_typesupport_introspection_cpp::MessageMembers *members_;
  std::shared_ptr<rcpputils::SharedLibrary> library_members_;

  std::string name_;
  std::vector<std::string> field_names_;
  std::vector<std::string> interface_names_;
  std::vector<int64_t> interface_offsets_;

  controller_interface::CallbackReturn status_ =
      controller_interface::CallbackReturn::SUCCESS;
};

class TypedSubscriptionInterface : public TypedInterface {
public:
  TypedSubscriptionInterface(
      std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node, std::string name,
      std::string type, double timeout, std::vector<double> &values);
  ~TypedSubscriptionInterface();

  void reset();

  controller_interface::return_type update_from_subscriber();

  std::vector<hardware_interface::CommandInterface> get_interfaces();

protected:
  void callback(const std::shared_ptr<rclcpp::SerializedMessage> &msg);

private:
  void cleanup();

  void *msg_;
  rcutils_allocator_t allocator_;
  rclcpp::Duration timeout_;

  std::unique_ptr<rclcpp::SerializationBase> serializer_;
  const rosidl_message_type_support_t *ts_serializer_;
  std::shared_ptr<rcpputils::SharedLibrary> library_serializer_;

  rclcpp::GenericSubscription::SharedPtr sub_;
  realtime_tools::RealtimeBuffer<std::vector<double>> values_rt_;
  std::vector<double> &values_;
  size_t start_index_;
};
} // namespace onnxruntime_controller
