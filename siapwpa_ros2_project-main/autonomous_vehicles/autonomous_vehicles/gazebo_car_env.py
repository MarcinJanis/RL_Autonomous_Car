import gym
from gym import spaces
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2


class GazeboCarEnv(gym.Env):
    """
    Prosty wrapper Gym dla Twojego autka:
    - akcja: [v_lin_norm, v_ang_norm] w [-1,1]
    - obserwacja: obraz z kamery 84x84x3 (uint8)
    """

    def __init__(self):
        super().__init__()

        # --- ROS2 init ---
        rclpy.init(args=None)
        self.node = Node("gym_mecanum_env")

        # --- bridge kamery i lidaru ---
        self.bridge = CvBridge()
        self.camera_img = None
        self.laser = None

        # SUB: kamera z Gazebo (po bridge’u)
        self.camera_sub = self.node.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            self._camera_cb,
            10
        )

        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan",
            self._lidar_cb,
            10
        )


        # PUB: sterowanie MecanumDrive
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

        # --- Gym spaces ---
        # akcja znormalizowana [-1,1]x[-1,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # obserwacja = obraz 84x84x3 (uint8)
        self.obs_h = 84
        self.obs_w = 84
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_h, self.obs_w, 3),
            dtype=np.uint8
        )

        # parametry fizyczne sterowania
        self.max_lin = 3.0   # maks. prędkość liniowa [m/s]
        self.max_ang = 2.0   # maks. prędkość kątowa [rad/s]

        self.step_count = 0
        self.max_steps = 500  # długość jednego epizodu w krokach

    # ------------- ROS CALLBACKS ------------- #
    def _camera_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # zmniejszamy i zamieniamy na RGB (opcjonalnie)
            img = cv2.resize(img, (self.obs_w, self.obs_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.camera_img = img
        except Exception as e:
            self.node.get_logger().warn(f"Camera conversion error: {e}")

    def _lidar_cb(self, msg: LaserScan):
        self.laser = np.array(msg.ranges, dtype=np.float32)

        # MEGA DEBUG:
        self.node.get_logger().info(
            f"[LIDAR CB] Otrzymano skan: {len(self.laser)} próbek, "
            f"min={np.nanmin(self.laser):.2f}, max={np.nanmax(self.laser):.2f}"
        )

    # ------------- GYM API ------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # zatrzymaj robota
        self._send_cmd(0.0, 0.0)

        self.step_count = 0
        # TODO: tutaj możesz kiedyś dodać prawdziwy reset świata (gz service)

        obs = self._get_obs_blocking()
        return obs  # Gym<=0.25: samo obs; Gym>=0.26: (obs, info)

    def step(self, action):
        self.step_count += 1

        # przeskaluj akcję z [-1,1] na zakres prędkości
        v_norm = float(np.clip(action[0], -1.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))

        v = v_norm * self.max_lin
        w = w_norm * self.max_ang

        self._send_cmd(v, w)

        # obsługa callbacków ROS
        rclpy.spin_once(self.node, timeout_sec=0.05)

        if self.laser is None:
          self.node.get_logger().warn("[STEP] LIDAR: self.laser is None")
        else:
          self.node.get_logger().info(
              f"[STEP] LIDAR: {len(self.laser)} próbek, min={np.nanmin(self.laser):.2f}"
          )

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self.step_count >= self.max_steps  # na razie kończymy po N krokach
        info = {}

        return obs, reward, done, info

    # ------------- POMOCNICZE ------------- #
    def _send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _get_obs(self):
        if self.camera_img is None:
            # jak jeszcze nie przyszło nic z kamery, zwróć zero
            return np.zeros((self.obs_h, self.obs_w, 3), dtype=np.uint8)
        return self.camera_img.copy()

    def _get_obs_blocking(self, timeout=2.0):
        """
        Poczekaj aż przyjdzie pierwsza klatka z kamery (do resetu).
        """
        waited = 0.0
        dt = 0.05
        while self.camera_img is None and waited < timeout:
            rclpy.spin_once(self.node, timeout_sec=dt)
            waited += dt
        return self._get_obs()

    def _compute_reward(self, obs):
        # NA RAZIE: zero – do zdefiniowania wg Twojego zadania
        # np. można użyć lidar/camera do karania za zbliżanie się do przeszkód
        return 0.0

    def close(self):
        self._send_cmd(0.0, 0.0)
        self.node.destroy_node()
        rclpy.shutdown()
