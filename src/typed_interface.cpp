#include "onnxruntime_controller/typed_interface.hpp"
#include "rclcpp/logging.hpp"
#include <cmath>
#include <controller_interface/controller_interface_base.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <rcl/types.h>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/serialization.hpp>
#include <rcpputils/find_library.hpp>
#include <rcpputils/shared_library.hpp>

#include <rosidl_typesupport_introspection_cpp/message_introspection.hpp>
#include <utility>

namespace onnxruntime_controller {

rosidl_message_type_support_t *TypedInterface::get_type_support_handle(
    std::string ts_library, std::string type,
    std::shared_ptr<rcpputils::SharedLibrary> &library) {
  auto package_name = type.substr(0, type.find('/'));
  auto message_name = type.substr(type.rfind('/') + 1);
  auto library_path =
      rcpputils::find_library_path(package_name + "__" + ts_library);
  library = std::make_shared<rcpputils::SharedLibrary>(library_path);
  auto symbol_name = ts_library + "__get_message_type_support_handle__" +
                     package_name + "__msg__" + message_name;
  auto get_ts_handle =
      reinterpret_cast<rosidl_message_type_support_t *(*)(void)>(
          library->get_symbol(symbol_name.c_str()));
  return get_ts_handle();
}

TypedInterface::TypedInterface(
    std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node, std::string name,
    std::string type)
    : node_(node),
      ts_members_(::rosidl_get_zero_initialized_message_type_support_handle()),
      members_(nullptr), name_(name) {
  ts_members_ = *get_type_support_handle("rosidl_typesupport_introspection_cpp",
                                         type, library_members_);

  members_ =
      static_cast<const rosidl_typesupport_introspection_cpp::MessageMembers *>(
          ts_members_.data);

  for (size_t i = 0; i < members_->member_count_; ++i) {
    auto member = members_->members_[i];

    if (member.is_array_ ||
        member.type_id_ ==
            rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE ||
        member.type_id_ ==
            rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING) {
      RCLCPP_ERROR(node_->get_logger(),
                   "Array, string or message field type not supported for "
                   "interface: %s.%s",
                   name.c_str(), member.name_);
      status_ = controller_interface::CallbackReturn::ERROR;
      return;
    } else {
      RCLCPP_ERROR(node_->get_logger(), "interface: %s.%s", name.c_str(),
                   member.name_);
      field_names_.push_back(std::string(member.name_));
      interface_names_.push_back(name + "." + std::string(member.name_));
      interface_offsets_.push_back(member.offset_);
    }
  }
}

const std::vector<std::string> &TypedInterface::get_interface_names() const {
  return interface_names_;
}

TypedSubscriptionInterface::TypedSubscriptionInterface(
    std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node, std::string name,
    std::string type, double timeout, std::vector<double> &values)
    : TypedInterface(node, name, type),
      timeout_(rclcpp::Duration::from_nanoseconds(timeout)), values_(values) {
  if (status_ != controller_interface::CallbackReturn::SUCCESS) {
    return;
  }

  ts_serializer_ = get_type_support_handle("rosidl_typesupport_cpp", type,
                                           library_serializer_);
  serializer_ = std::make_unique<rclcpp::SerializationBase>(ts_serializer_);

  allocator_ = rcutils_get_default_allocator();
  msg_ = allocator_.allocate(members_->size_of_, allocator_.state);

  if (!msg_) {
    status_ = controller_interface::CallbackReturn::ERROR;
    return;
  }

  members_->init_function(msg_, rosidl_runtime_cpp::MessageInitialization::ALL);

  sub_ = node_->create_generic_subscription(
      "~/" + name, type, rclcpp::SystemDefaultsQoS(),
      std::bind(&TypedSubscriptionInterface::callback, this,
                std::placeholders::_1));

  start_index_ = values_.size();
  values.resize(start_index_ + interface_names_.size(),
                std::numeric_limits<double>::quiet_NaN());
}

TypedSubscriptionInterface::~TypedSubscriptionInterface() { cleanup(); }

void TypedSubscriptionInterface::callback(
    const std::shared_ptr<rclcpp::SerializedMessage> &msg) {
  serializer_->deserialize_message(msg.get(), msg_);

  std::vector<double> values;
  values.resize(interface_offsets_.size(),
                std::numeric_limits<double>::quiet_NaN());
  uint8_t *base_ptr = static_cast<uint8_t *>(msg_);
  for (size_t i = 0; i < interface_offsets_.size(); ++i) {
    values[i] = *reinterpret_cast<double *>(base_ptr + interface_offsets_[i]);
  }

  values_rt_.writeFromNonRT(values);
}

std::vector<hardware_interface::CommandInterface>
TypedSubscriptionInterface::get_interfaces() {
  // TODO: Switch to using shared pointers to interfaces w/
  // on_export_reference_interfaces_list in future ros2_control version

  std::vector<hardware_interface::CommandInterface> interfaces;

  for (auto i = 0U; i < interface_names_.size(); ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    interfaces.emplace_back(node_->get_name(), interface_names_[i],
                            &values_[start_index_ + i]);
#pragma GCC diagnostic pop
  }
  return interfaces;
}

void TypedSubscriptionInterface::reset() {
  values_ = std::vector<double>(interface_offsets_.size(),
                                std::numeric_limits<double>::quiet_NaN());
  values_rt_.writeFromNonRT(values_);
}

void TypedSubscriptionInterface::cleanup() {
  if (msg_) {
    members_->fini_function(msg_);
    allocator_.deallocate(msg_, allocator_.state);
    msg_ = nullptr;
  }
}

controller_interface::return_type
TypedSubscriptionInterface::update_from_subscriber() {
  auto values = *values_rt_.readFromRT();

  bool all_finite = true;
  for (auto &value : values) {
    if (!std::isfinite(value)) {
      all_finite = false;
      break;
    }
  }

  if (!all_finite) {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(
        node_->get_logger(), *node_->get_clock(),
        static_cast<rcutils_duration_value_t>(timeout_.seconds() * 1000),
        "Command message contains NaNs. Not updating reference interfaces.");
  } else {
    for (size_t i = 0; i < values.size(); ++i) {
      values_[start_index_ + i] = values[i];
    }
  }

  // previous_update_timestamp_ = time;

  return controller_interface::return_type::OK;
}
} // namespace onnxruntime_controller
