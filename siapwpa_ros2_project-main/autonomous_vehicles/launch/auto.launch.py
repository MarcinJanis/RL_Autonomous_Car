from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    world_path = "/home/developer/ros2_ws/src/models/mecanum.sdf"
    gz_bridge_path = "/home/developer/ros2_ws/src/autonomous_vehicles/config/bridge_config.yaml"

    gz_sim_world = ExecuteProcess(
        cmd=["gz", "sim", world_path, "-r"],
        output="screen"
    )

    gz_bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "--bridge-config", gz_bridge_path
        ],
        output="screen"
    )

    control_node = Node(
        package='autonomous_vehicles',  # poprawna nazwa pakietu
        executable='control_node',
        name='control_node',
        output="screen"
    )


    return LaunchDescription([
        gz_sim_world,
        gz_bridge,
        control_node
    ])
