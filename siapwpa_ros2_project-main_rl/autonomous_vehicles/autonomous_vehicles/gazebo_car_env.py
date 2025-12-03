import gymnasium
from gymnasium import spaces
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from gazebo_msgs.msg import ModelStates
from ros_gz_interfaces.msg import Contacts
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import trajectory_gt as gt

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

        # state: 
        self.camera_img = None
        self.laser = None
        self.global_pose = None 
        self.collision_flag = False


        # rewards
        self.rewards = { 'velocity': 1, 'trajectory': 5, 'collision': -15, 'timeout': -5}

        trajectory_points_pth = './siapwpa_ros2_project-main_rl/models/walls/waypoints_il.csv'
        self.trajectory = gt.traj_gt(trajectory_points_pth)
        self.trajectory.setup()

        # --- info ---
        self.rest_info = {
            # Add here all logs when reset 
            "is_success": False
        }

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

        self.pose_sub = self.node.create_subscription(
            ModelStates,
            "/model/vehicle_blue/odometry",
            self._model_states_cb,
            10
        )

        self.collision_event_sub = self.node.create_subscription(
            Contacts,
            "/world/mecanum_drive/model/track_model/link/track_link/sensor/walls_contact_sensor/contact",
            self._collision_cb,
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

        # obserwacja = obraz 256x256x3 (uint8)
        self.img_h = 256
        self.img_w = 256
        self.camera_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.img_h, self.img_w, 3),
            dtype=np.uint8
        )
        self.lidar_l = 280
        self.lidar_space = spaces.Box(
            low=0.0,
            high=12.0,
            shape=(self.lidar_l),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
                    "image": self.camera_space,
                    "lidar": self.lidar_space,
                    # "pose": self.state_space
                })

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
            img = cv2.resize(img, (self.img_w, self.img_h))
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

    def _model_states_cb(self, msg: ModelStates):

        if "vehicle_blue" in msg.name:
            idx = msg.name.index("vehicle_blue")
            state = msg.pose[idx]
            self.global_pose = np.array((state.position.x, state.position.y, state.velocity.x, state.velocity.y))

            # if oritntation will be needed:
            # q = state.orientation
            # roll, pitch, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])


    def _collision_cb(self, msg: Contacts):
        if len(msg.contacts) > 0:
            self.collision_flag = True
        else: 
            self.collision_flag = False
    # ------------- GYM API ------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # zatrzymaj robota
        self._send_cmd(0.0, 0.0)

        self.step_count = 0
        self.collision_flag = False
        # TODO: tutaj możesz kiedyś dodać prawdziwy reset świata (gz service)
        # ------------------------


        # ------------------------
        obs = self._get_obs_blocking()
        return obs, self.rest_info  # co to info 

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

        terminated = False
        truncated = False
        # if max steps reached -> terminated
        if self.step_count >= self.max_steps: truncated = True
        # if collision detected -> terminated 
        if self.collision_flag: terminated = True

        if terminated: truncated = False # Terminated more important than truncated

        info = {}

        # return obs, reward, done, info
        return obs, reward, terminated, truncated, info

    # ------------- POMOCNICZE ------------- #
    def _send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _get_obs(self):

        if self.camera_img is None:
            self.camera_img  = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        
        if self.laser is None:
            self.laser = np.ones((self.lidar_l), dtype=np.float32) * 12.0

        return {"image": self.camera_img, "lidar": self.laser}

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
        reward = 0 
        # self.rewards = { 'velocity': 1, 'trajectory': 5, 'collision': -15, 'timeout': -5}
        x, y, vx, vy = self.global_pose

        # 1 - reward for velocity
        v_xy = np.sqrt(vx**2, vy**2)
        reward += v_xy * self.rewards['velocity']

        # 2 - reward for distance from desire trajectory
        x_cp, y_cp, dist = self.trajectory.get_dist(x, y, n=100) # x_cp, y_cp - closet points on trajectory
        reward += v_xy * self.rewards['trajectory'] * -1
        
        # 3 - reward for collision

        # 4 - reward for timeout

        
        # NA RAZIE: zero – do zdefiniowania wg Twojego zadania
        # np. można użyć lidar/camera do karania za zbliżanie się do przeszkód
        return 0.0

    def close(self):
        self._send_cmd(0.0, 0.0)
        self.node.destroy_node()
        rclpy.shutdown()


