import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

# from stable_baselines3 import PPO

import time 

from utils_real import wheelSpeedDistributor 

class MasterController(Node):

    #TODO: create node for reading camera data, assign to self.camera_img
    #TODO: create node for reading lidar data, assign to self.lidar_scan
    
    def __init__(self, car_config: dict, sensor_config: dict,  model_pth, display = True):
        super().__init__('MasterControllerNode')

        self.display = display 
        self.WSD = wheelSpeedDistributor(car_config)
        # car_config = {'r':1, 'lx':2, 'ly':2, 'v_lin_max':2.0, 'v_ang_max':2.0}
        # sensor_config = {'lidar_beams':280, 'lidar_range':12, 'angle_min': -1.099, 'angle_increment': 1 * np.pi / 180, 'img_shape':(255, 255, 3)}


        # car config:
        self.v_max_lin = car_config['v_lin_max']
        self.v_max_ang = car_config['v_ang_max']

        # lidar config:
        self.lidar_beams = sensor_config['lidar_beams']
        self.lidar_range = sensor_config['lidar_range']

        # camera config:
        self.img_shape = sensor_config['img_shape']

        # if not model_pth is None:
            # self.model = PPO.load(model_pth)

        # --- inner state --- #
        self.camera_img = None
        self.lidar_scan = None
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
            
            angle_increment = 1 * np.pi / 180# [deg]
            samples = 280 
            angle_min = -1.099 # [rad]
            range_min = 0.01 # [m]
            range_max = 12 # [m]
            range_resolution = 0.01 # [m]
      
            #TODO: Correct lidar preprocessing 
            #TODO: Instead of fixed paraemters, propably it is possible to get them from ros2 msg 
           
            scan = np.clip(scan, 0.0, self.lidar_range)
            self.lidar_scan = scan / self.lidar_range

            if self.display:
                self.lidar_scan_visu = self.lidar_disp_transform(self.lidar_scan, range_min, range_max, angle_min, angle_increment)
                cv2.imshow("Lidar BEV", self.lidar_scan_visu)
                cv2.waitKey(1)

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
        
    def lidar_disp_transform(self, scan, range_min, range_max, angle_min, angle_increment, bev_size=500, display_range=1.0, scale=None):
        """
        Tworzy BEV LiDAR tylko dla punktów do display_range.
        """
        # Zamiana na rzeczywiste odległości
        ranges = scan * range_max

        # Skala tak, aby display_range wypełniało cały obraz
        if scale is None:
            scale = bev_size / (2 * display_range)

        # Środek mapy
        cx, cy = bev_size // 2, bev_size // 2

        # Pusta mapa
        bev_map = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

        # Punkty w zakresie range_min..display_range
        valid_indices = np.where((ranges >= range_min) & (ranges <= display_range))[0]

        if len(valid_indices) > 0:
            angles = angle_min + valid_indices * angle_increment
            dists = ranges[valid_indices]

            # ROS coords
            x_ros = dists * np.cos(angles)
            y_ros = dists * np.sin(angles)

            # Pixele OpenCV – **tu ważna zmiana**
            px = (cx - y_ros * scale).astype(int)
            py = (cy - x_ros * scale).astype(int)

            # Maskowanie punktów poza obrazem
            mask = (px >= 0) & (px < bev_size) & (py >= 0) & (py < bev_size)

            # Rysowanie punktów
            bev_map[py[mask], px[mask]] = (0, 255, 0)
      

        # Robot

        cv2.circle(bev_map, (cx, cy), 3, (255, 0, 0), -1)

        # # Kierunek przodu (opcjonalnie)
        # front_length_px = int(0.5 * scale)
        # cv2.line(bev_map, (cx, cy), (cx, cy - front_length_px), (0, 0, 255), 2)

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