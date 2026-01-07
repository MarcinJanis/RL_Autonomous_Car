import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

from stable_baselines3 import PPO

import time 

from utils_real import wheelSpeedDistributor 

class MasterController(Node):
    
    def __init__(self, car_config: dict, sensor_config: dict,  model_pth, display = True, dt = 0.1):
        super().__init__('MasterControllerNode')

        self.dt = dt
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.display = display 
        self.WSD = wheelSpeedDistributor(car_config)
 
        # car config:
        self.v_max_lin = car_config['v_lin_max']
        self.v_max_ang = car_config['v_ang_max']

        # lidar config:
        self.lidar_beams = sensor_config['lidar_beams']
        self.lidar_max_range = sensor_config['lidar_max_range']
        self.lidar_min_range = sensor_config['lidar_min_range']
        self.lidar_angle_min = sensor_config['lidar_angle_min']
        self.lidar_angle_step = sensor_config['lidar_angle_step']
        self.lidar_display_range = sensor_config['lidar_display_range']

        # camera config:
        self.img_shape = sensor_config['img_shape']

        try:
            self.model = PPO.load(model_pth)
        except Exception as e:
            print(f'[Warning] Cannot load model from path: \n {e}')


        # --- inner state --- #
        self.camera_img = None
        self.lidar_scan = None

        # --- visu state --- #
        self.lidar_scan_visu = None

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
            "/scan",
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

            # preprocess and normalize
            scan_left_side = scan[-int(self.lidar_beams/2):]
            scan_right_side = scan[:int(self.lidar_beams/2)]
            scan = np.concatenate((scan_left_side, scan_right_side), axis=0)
            scan = np.clip(scan, 0.0, self.lidar_max_range)
            self.lidar_scan = scan / self.lidar_max_range

            if self.display:
                self.lidar_scan_visu = self.lidar_disp_transform(self.lidar_scan, 
                                                                 self.lidar_min_range,
                                                                 self.lidar_max_range, 
                                                                 self.lidar_angle_min, 
                                                                 self.lidar_angle_step,
                                                                 bev_size=500, 
                                                                 display_range=self.lidar_display_range, 
                                                                 scale=None)
                cv2.imshow("Lidar BEV", self.lidar_scan_visu)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from lidaer:\n{e}")


    def act(self):
        if not self.camera_img is None and not self.lidar_scan is None:
            obs = {"image": self.camera_img, "lidar": self.lidar_scan}
            action, _ = self.model.predict(obs, deterministic=True)
            v_norm, w_norm = action
            v = v_norm * self.v_max_lin
            w = w_norm * self.v_max_ang
            v = np.clip(v, -self.v_max_lin, self.v_max_lin)
            w = np.clip(w, -self.v_max_ang, self.v_max_ang)
            ws = self.WSD.allocate_wheelspeed(v, w) 
        else:
            print('Waiting for sensors data...')
            ws = np.zeros((4, 1), dtype = np.float32)

        # wheel speeds: 
        #  ws[0] - left front 
        #  ws[1] - right front
        #  ws[2] - right back 
        #  ws[3] - left back
     
            return ws
        
    def control_loop(self):
        
        w = self.act()
        #TODO: wystawić sterowanie!?


    def lidar_disp_transform(self, scan, range_min, range_max, angle_min, angle_increment, bev_size=500, display_range=1.0, scale=None):

        # rescale to real values
        ranges = scan * range_max

        # scale: 1m = scale pixels
        if scale is None:
            scale = bev_size / (2 * display_range)

        # center
        cx, cy = bev_size // 2, bev_size // 2

        # raw map
        bev_map = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

        # filter in range (min_range, display_range)
        valid_indices = np.where((ranges >= range_min) & (ranges <= display_range))[0]

        if len(valid_indices) > 0:
            angles = angle_min + valid_indices * angle_increment
            dists = ranges[valid_indices]

            # ROS coords
            x_ros = dists * np.cos(angles)
            y_ros = dists * np.sin(angles)

            # OpenCV pixels 
            px = (cx - y_ros * scale).astype(int)
            py = (cy - x_ros * scale).astype(int)

            # Discard points out of map
            mask = (px >= 0) & (px < bev_size) & (py >= 0) & (py < bev_size)

            # Draw points
            bev_map[py[mask], px[mask]] = (0, 255, 0)

        cv2.circle(bev_map, (cx, cy), 3, (255, 0, 0), -1)
        return bev_map

    def visu_quit(self):
        cv2.destroyAllWindows()

# class CameraController(Node):
#     '''
#     > Get data from camera
#     > Preprocessing 
#     > Send via ROS
#     '''
#     def __init__(self):
#         super().__init__('CameraControllerNode')

#     pass


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