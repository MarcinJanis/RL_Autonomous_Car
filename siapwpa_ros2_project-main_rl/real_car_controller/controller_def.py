import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

from stable_baselines3 import PPO

import time 

from utils_real import wheelSpeedDistributor 

class MasterController(Node):

    #TODO: create node for reading camera data, assign to self.camera_img
    #TODO: create node for reading lidar data, assign to self.lidar_scan
    
    def __init__(self, car_config: dict, sensor_config: dict,  model_pth):
        super().__init__('MasterControllerNode')
        self.WSD = wheelSpeedDistributor(car_config)
        # car_config = {'r':1, 'lx':2, 'ly':2, 'v_lin_max':2.0, 'v_ang_max':2.0}
        # sensor_config = {'lidar_beams':280, 'lidar_range':12, 'img_shape':(255, 255, 3)}

        # car config:
        self.v_max_lin = car_config['v_lin_max']
        self.v_max_ang = car_config['v_ang_max']

        # lidar config:
        self.lidar_beams = sensor_config['lidar_beams']
        self.lidar_range = sensor_config['lidar_range']

        # camera config:
        self.img_shape = sensor_config['img_shape']

        self.model = PPO.load(model_pth)

        # --- inner state --- #
        self.camera_img = None
        self.lidar_scan = None

        # --- ROS subsribers --- 
        self.bridge = CvBridge()

        self.ros_camera_sub = self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            self._camera_cb,
            10 
        )

        self.ros_lidar_sub = self.create_subscription(
            LaserScan,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan",
            self._lidar_cb,
            10 
        )

        # --- ROS Callbacks --- 
    def _camera_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            #TODO: image preprocessing
            self.camera_img = img
        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from camera:\n{e}")


    def _lidar_cb(self, msg: LaserScan):
        try:
            scan = np.array(msg.ranges, dtype=np.float32)
            
            #TODO: Correct lidar preprocessing 
            #TODO: Instead of fixed paraemters, propably it is possible to get them from ros2 msg 
           
            scan = np.clip(scan, 0.0, self.lidar_range)
            self.lidar_scan = scan / self.lidar_range
        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from lidaer:\n{e}")


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


# Not necessery if provided node will work 
# class LidarController(Node):
#     '''
#     > Get data from lidar
#     > Preprocessing (cut off to leave 280 bins (120 angle))
#     > Send via ROS
#     '''
#     def __init__(self):
#         super().__init__('LidarControllerNode')

#     pass