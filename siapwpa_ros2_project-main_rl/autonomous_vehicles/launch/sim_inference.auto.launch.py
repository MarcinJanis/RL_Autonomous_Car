from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os 


def generate_launch_description():
    world_path = "/home/developer/ros2_ws/src/models/walls/mecanum.sdf"
    inference_script_path = "/home/developer/ros2_ws/src/autonomous_vehicles/autonomous_vehicles/training_SB3.py"
    working_dir = "/home/developer/ros2_ws/src/autonomous_vehicles"

    gz_sim_world = ExecuteProcess(
        # cmd=["gz", "sim", world_path, "-r"],
        cmd=["gz", "sim", world_path, "-s", "-r"],
        output="screen"
    )    

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/world/mecanum_drive/model/vehicle_blue/link/chassis/sensor/chassis_contact_sensor/contact@ros_gz_interfaces/msg/Contacts@gz.msgs.Contacts',
            '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/model/vehicle_blue/pose@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V'
            ],
        output='screen'
    )

    sim_inference_node = ExecuteProcess(
        cmd=['python3', inference_script_path],
        output='screen',
        cwd=working_dir 
    )

    return LaunchDescription([
        gz_sim_world,
        gz_bridge,
        sim_inference_node
    ])



