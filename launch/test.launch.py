from launch import LaunchDescription
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("onnxruntime_controller"),
                    "urdf",
                    "test.ros2_control.xacro",
                ]
            ),
        ]
    )
    model_path = {
        "model_path": ParameterValue(PathJoinSubstitution(
            [
                FindPackageShare("onnxruntime_controller"),
                "models",
                "policy.onnx",
            ]
        ), value_type=str)
    }
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("onnxruntime_controller"),
            "config",
            "test.yaml",
        ]
    )
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers, model_path],
        # prefix=['gdbserver localhost:3000'],
        output="both",
    )
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    onnx_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "onnxruntime_controller",
            "--controller-manager",
            "/controller_manager",
        ],
    )
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    nodes = [
        control_node,
        robot_state_pub_node,
        onnx_spawner,
        joint_state_broadcaster,
    ]

    return LaunchDescription(nodes)
