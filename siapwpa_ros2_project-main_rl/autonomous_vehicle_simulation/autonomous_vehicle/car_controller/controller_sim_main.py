from controller_sim_def import CarController
from stable_baselines3 import PPO
import rclpy

import time 
import sys


sys.path.append('/home/developer/ros2_ws/src/autonomous_vehicles/autonomous_vehicles')
# --Parameters ---

V_lin_max = 6.5
V_ang_max = 2.0
step_time = 0.1 # [s] Update frequency -> the same as in training
# model_pth = '/home/developer/ros2_ws/src/autonomous_vehicles/models/test_run10/model_e22_rp168_99.zip' # trained for v=3m/s,
model_pth = '/home/developer/ros2_ws/src/autonomous_vehicles/models/test_run11/model_e13_rp163_15.zip' # trained for v=6m/s, (good result for v_lin 6.5, v_ang 2.0, drift for v_lin 8.0, v_ang 2.0)


  
# ----------------


def main(args=None):
    rclpy.init(args=args)

    # create controller
    Controller = CarController(step_time = step_time, Vmax_lin = V_lin_max, Vmax_ang = V_ang_max, lidar_max_range = 12, lidar_n_beans = 280)

    try:
        model = PPO.load(model_pth)
        Controller.setup(model)
        print("> Model imported.")
    except Exception as e:
        print(f"[Error] Error during model importing: {e}")
        return


    Controller.start_time = time.time() # set start time 

    try:
        # --- Main loop ---
        rclpy.spin(Controller)
    except KeyboardInterrupt:
        print("Keybord interput. Closing...")
    finally:
        Controller.visu_quit()
        Controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
