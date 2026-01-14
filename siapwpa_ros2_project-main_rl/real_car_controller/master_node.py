import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

from stable_baselines3 import PPO

import time 

from wheel_speed_distribution import wheelSpeedDistributor 

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
            # obraz z preprocess node jest JUŻ rgb8 i 256x256
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
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
            # self.camera_img = None
            # self.lidar_scan = None
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

