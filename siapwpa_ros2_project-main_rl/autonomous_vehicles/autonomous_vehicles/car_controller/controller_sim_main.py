from controller_sim_def import CarController
from stable_baselines3 import PPO
import rclpy

import time 

# --Parameters ---

V_lin_max = 3.0
V_ang_max = 2.0
update_time = 0.1 # [s] Update frequency -> the same as in training
model_pth = ''
  
# ----------------


def main(args=None):
    rclpy.init(args=args)

    # create controller
    Controller = CarController(Vmax_lin = V_lin_max, Vmax_ang = V_ang_max, lidar_max_range = 12, lidar_n_beans = 280, mem_sample_max = 200)

    # create timer for controller update frequency 
    rate = Controller.create_rate(1/update_time) # takes time of additional computing into account when waiting

    try:
        model = PPO.load(model_pth)
        Controller.setup(model)
        print("> Model imported.")
    except Exception as e:
        print(f"[Error] Error during model importing: {e}")
        return

    try:
        # --- Main loop ---
        while rclpy.ok():
            rclpy.spin_once(Controller, timeout_sec=0.001)
            Controller.act() # Get obs, inference, send control cmd
            Controller.log() # log additional data 
            Controller.visu(speed_grad=True, draw_collision=True) # visualisation
            rate.sleep()

    except KeyboardInterrupt:
        print("Keybord interput. Closing...")
    finally:
        Controller.visu_quit()
        Controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
