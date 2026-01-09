
from controller_def import MasterController, LidarDisplayNode, LidarPreprocessNode
import rclpy
from rclpy.executors import MultiThreadedExecutor
import numpy as np

car_config = {
    'r': 0.05,  # przykładowe realne wartości
    'lx': 0.2, 
    'ly': 0.2, 
    'v_lin_max': 2.0, 
    'v_ang_max': 2.0
}

sensor_config = {'lidar_beams': 280, 
                'lidar_max_range':12, 
                'lidar_min_range':0.01, 
                'lidar_angle_min': -1.099, 
                'lidar_angle_step': 1 * np.pi / 180, 
                'lidar_display_range':1.0, 
                'img_shape':(255, 255, 3)
                }

motors_config = {'serial_port':'/dev/ttyACM0', 
                 'baud_rate':9600, 
                 'CALIB_MOTOR_LB':1.0, 
                 'CALIB_MOTOR_RB':1.0, 
                 'CALIB_MOTOR_LF':1.0, 
                 'CALIB_MOTOR_RF':1.0, 
                 'max_wheel_speed':2}

def main():
    rclpy.init()
    
    model_path = "path/to/your/model.zip"

    master_node = MasterController(car_config, sensor_config, motors_config, model_pth=model_path)
    preprocess_node = LidarPreprocessNode(sensor_config)
    display_node = LidarDisplayNode(sensor_config)

    executor = MultiThreadedExecutor()
    executor.add_node(master_node)
    executor.add_node(preprocess_node)
    executor.add_node(display_node)

    try:
        print("Running nodes...")
        executor.spin()
        print("Nodes are running.")
    except KeyboardInterrupt:
        print("\nKilling nodes...")
    finally:
        master_node.arduino.close()
        display_node.visu_quit()
        master_node.destroy_node()
        preprocess_node.destroy_node()
        display_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()