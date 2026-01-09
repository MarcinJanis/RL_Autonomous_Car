import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

from stable_baselines3 import PPO

import time 

from utils_real import wheelSpeedDistributor 

import serial
class MasterController(Node):
    def __init__(self, car_config: dict, sensor_config: dict, motors_config: dict, model_pth,dt = 0.1):
        super().__init__('MasterControllerNode')

        self.dt = dt
        self.timer = self.create_timer(self.dt, self.control_loop)

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

        # communication config
        self.serial_port = motors_config['serial_port']
        self.baud_rate = motors_config['baud_rate']
        self.CALIB_MOTOR_LB = motors_config['CALIB_MOTOR_LB']  # left back
        self.CALIB_MOTOR_RB = motors_config['CALIB_MOTOR_RB']  # right back
        self.CALIB_MOTOR_LF = motors_config['CALIB_MOTOR_LF']  # left front
        self.CALIB_MOTOR_RF = motors_config['CALIB_MOTOR_RF']  # right front

        self.max_wheel_speed = motors_config['max_wheel_speed'] 

        # camera config:
        self.img_shape = sensor_config['img_shape']

        try:
            self.model = PPO.load(model_pth)
        except Exception as e:
            print(f'[Warning] Cannot load model from path: \n {e}')

        # --- inner state --- #
        self.camera_img = None
        self.lidar_scan = None

        # --- ROS subsribers --- 
        self.bridge = CvBridge()

        self.ros_camera_sub = self.create_subscription(
            Image,
            # "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            "/camera",
            self._camera_cb,
            10 
        )

        self.ros_lidar_sub = self.create_subscription(
            LaserScan,
            "/lidar_scan",
            self._lidar_cb,
            10 
        )

        # serial port init
        try:
            self.arduino =  serial.Serial(port=self.serial_port, baudrate=self.baud_rate, timeout=1)
            time.sleep(2)
            self.arduino.flush()
            print(f"Connected to {self.serial_port}")
        except Exception as e:
            print(f"Cannot conect to {self.serial_port}: \n{e}")


        # --- ROS Callbacks --- 
    def _camera_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img = cv2.resize(img, (256, 256), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            self.camera_img = img
        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from camera:\n{e}")

    def _lidar_cb(self, msg: LaserScan):
        try:
            self.lidar_scan = np.array(msg.ranges, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from lidar:\n{e}")

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
            self.camera_img = None
            self.lidar_scan = None
        else:
            print('Waiting for sensors data...')
            ws = np.zeros((4, 1), dtype = np.float32)
            return ws
        
    def send_cmd(self, w, distance=0, complex_mode=250):

        pwm = np.clip((np.abs(w) / self.max_wheel_speed * 255), 0, 255).astype(np.uint8)
        dir = np.clip(np.sign(w), 0, 1).astype(np.uint8)
        data_packet = [complex_mode, distance, pwm[0], pwm[1], pwm[3], pwm[2], dir[0], dir[1], dir[3], dir[2]]
        self.arduino.write(bytes(data_packet))

        # data packet:
        # 1 left front -> w[0] - left front 
        # 2 right front -> 	w[1] - right front
        # 3 left back -> w[3] - left back
        # 4 right back -> w[2] - right back 

    def control_loop(self):
        w = self.act()
        self.send_cmd(w)

    def destroy_node(self):
        self.send_cmd(np.zeros((4,1))) 
        self.arduino.close()
        super().destroy_node()

class LidarPreprocessNode(Node):
    def __init__(self, sensor_config: dict, dt = 10):

        super().__init__('LidarPreprocessNode')

        self.dt = dt

        self.lidar_beams = sensor_config['lidar_beams']
        self.lidar_max_range = sensor_config['lidar_max_range']
        self.lidar_min_range = sensor_config['lidar_min_range']
        self.lidar_angle_min = sensor_config['lidar_angle_min']
        self.lidar_angle_step = sensor_config['lidar_angle_step']
        self.lidar_display_range = sensor_config['lidar_display_range']

        self.ros_lidar_raw_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self._lidar_cb,
            10 
        )

        self.ros_lidar_scan_pub = self.create_publisher(
            LaserScan, 
            '/lidar_scan', 
            10
        )

    def _lidar_cb(self, msg: LaserScan):
        try:
            scan = np.array(msg.ranges, dtype=np.float32)

            # preprocess and normalize
            scan_left_side = scan[-int(self.lidar_beams/2):]
            scan_right_side = scan[:int(self.lidar_beams/2)]
            scan = np.concatenate((scan_left_side, scan_right_side), axis=0)
            scan = np.clip(scan, 0.0, self.lidar_max_range)
            self.lidar_scan = scan / self.lidar_max_range

        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from lidar:\n{e}")
        
        try:
            # Scan to send
            pub_msg_lidar_scan = LaserScan()
            
            pub_msg_lidar_scan.header = msg.header 
            pub_msg_lidar_scan.header.stamp = self.get_clock().now().to_msg() 

            pub_msg_lidar_scan.angle_min = self.lidar_angle_min
            pub_msg_lidar_scan.angle_increment = self.lidar_angle_step
            pub_msg_lidar_scan.range_min = self.lidar_min_range
            pub_msg_lidar_scan.range_max = self.lidar_max_range

            pub_msg_lidar_scan.ranges = self.lidar_scan.astype(np.float32).tolist()

            self.ros_lidar_scan_pub.publish(pub_msg_lidar_scan)

        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot send lidar data:\n{e}")
        

class LidarDisplayNode(Node):
    def __init__(self, sensor_config: dict, dt = 10):

        super().__init__('LidarDisplayNode')

        self.bridge = CvBridge()

        self.lidar_beams = sensor_config['lidar_beams']
        self.lidar_max_range = sensor_config['lidar_max_range']
        self.lidar_min_range = sensor_config['lidar_min_range']
        self.lidar_angle_min = sensor_config['lidar_angle_min']
        self.lidar_angle_step = sensor_config['lidar_angle_step']
        self.lidar_display_range = sensor_config['lidar_display_range']

        # --- visu --- #
        self.lidar_scan_visu = None

        self.ros_lidar_raw_sub = self.create_subscription(
            LaserScan,
            "/lidar_scan",
            self._lidar_cb,
            10 
        )

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

    def _lidar_cb(self, msg: LaserScan):

        lidar_bev_map = np.array(msg.ranges, dtype=np.float32)

        try:
            lidar_scan_visu = self.lidar_disp_transform(lidar_bev_map, 
                                                        self.lidar_min_range,
                                                        self.lidar_max_range, 
                                                        self.lidar_angle_min, 
                                                        self.lidar_angle_step,
                                                        bev_size=500, 
                                                        display_range=self.lidar_display_range, 
                                                        scale=None)

            cv2.imshow("Lidar BEV", lidar_scan_visu)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot display lidar data:\n{e}")