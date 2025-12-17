import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts

from stable_baselines3 import PPO

import time 

from utils_real import wheelSpeedDistributor 

class MasterController(Node):

    #TODO: create node for reading camera data, assign to self.camera_img
    #TODO: create node for reading lidar data, assign to self.lidar_scan
    
    def __init__(self, car_config: dict, model_pth):
        super().__init__('MasterControllerNode')
        self.WSD = wheelSpeedDistributor(car_config)
        # car_config = {'r':1, 'lx':2, 'ly':2, 'v_lin_max':2.0, 'v_ang_max':2.0}

        self.v_max_lin = car_config['v_lin_max']
        self.v_max_ang = car_config['v_ang_max']

        self.model = PPO.load(model_pth)

        # --- inner state --- #
        self.camera_img = None
        self.lidar_scan = None


    def act(self):
        if not self.camera_img is None or not self.lidar_scan is None:
            obs = {"image": self.camera_img, "lidar": self.lidar_scan}
            action, _ = self.model.predict(obs, deterministic=True)
            v_norm, w_norm = action
            v = v_norm * self.v_max_lin
            w = w_norm * self.v_max_ang
            v = np.clip(v, -self.Vmax_lin, self.Vmax_lin)
            w = np.clip(w, -self.Vmax_ang, self.Vmax_ang)
            ws = self.WSD.allocate_wheelspeed(v, w) 
        else:
            ws = np.zeros((4, 1), dtype = np.float32)

        # wheel speeds: 
        #  ws[0] - left front 
        #  ws[1] - right front
        #  ws[2] - right back 
        #  ws[3] - left back
     
            return ws
        


class CameraController(Node):
    '''
    > Get data from camera
    > Preprocessing 
    > Send via ROS
    '''
    def __init__(self):
        super().__init__('CameraControllerNode')

    pass

class LidarController(Node):
    '''
    > Get data from lidar
    > Preprocessing (cut off to leave 280 bins (120 angle))
    > Send via ROS
    '''
    def __init__(self):
        super().__init__('LidarControllerNode')

    pass