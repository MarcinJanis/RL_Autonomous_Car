from controller_def import MasterController
import rclpy
import numpy as np

car_config = {'r':1, 'lx':2, 'ly':2, 'v_lin_max':2.0, 'v_ang_max':2.0}

sensor_config = {'lidar_beams': 280, 
                'lidar_max_range':12, 
                'lidar_min_range':0.01, 
                'lidar_angle_min': -1.099, 
                'lidar_angle_step': 1 * np.pi / 180, 
                'lidar_display_range':1.0, 
                'img_shape':(255, 255, 3)}

def main():

  rclpy.init()
  CarController = MasterController(car_config, sensor_config, model_pth=None, display = True)

  try:
      # --- Main loop ---
      rclpy.spin(CarController)
  except KeyboardInterrupt:
      print("Keybord interput. Closing...")
  finally:
      CarController.visu_quit()
      CarController.destroy_node()
      rclpy.shutdown()


if __name__ == '__main__':
    main()
