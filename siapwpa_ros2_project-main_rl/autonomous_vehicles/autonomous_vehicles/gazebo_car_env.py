import gymnasium
from gymnasium import spaces
import numpy as np
import cv2
import subprocess
import time
import os


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
# from gazebo_msgs.msg import Pose_V, Pose
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from geometry_msgs.msg import Quaternion

import trajectory_gt as gt

from rclpy.qos import qos_profile_sensor_data

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class GazeboCarEnv(gymnasium.Env):

    def __init__(self, time_step : float , rewards: dict, trajectory_points_pth: str, max_steps_per_episode: int, max_lin_vel: float, max_ang_vel: float, render_mode = True):
        super().__init__()


        # --- General inits ---
        # Create subfolder for saving logs from training
        self.LOG_DIR = os.path.join(os.getcwd(), f'./training_logs')
        os.makedirs(name=self.LOG_DIR, exist_ok=True)

        self.time_step = time_step
        self.render_mode = True

        self.episode_count = 0
        # --- ROS2 node init ---
        if not rclpy.ok(): rclpy.init(args=None)

        self.node = Node("gym_mecanum_env")
        # self.set_state_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.set_state_client = self.node.create_client(SetEntityPose, '/world/mecanum_drive/set_pose')
        

        # --- bridge for camera and lidar ---
        self.bridge = CvBridge()

        # --- state variables ---
        # keeps values of obserwation
        self.camera_img = None
        self.laser = None
        self.global_pose = np.zeros((2), dtype = np.float32)
        self.global_vel = np.zeros((3), dtype = np.float32)

        self.odom_received = False

        self.collision_flag = False
        self.timeout_flag = False
        self.destination_reached_flag = False

        # --- rewards value ---
        self.rewards = rewards 
        self.rewards_components = np.zeros((len(self.rewards)), dtype = np.float32)
        self.rewards_components_sum = np.zeros((len(self.rewards)), dtype = np.float32)
        # shape like: { 'velocity': 1, 'trajectory': 5, 'collision': -15, 'timeout': -5, 'destin': 20}

        # init object for distance from trajectory calc
        self.trajectory = gt.traj_gt()
        self.trajectory.setup(trajectory_points_pth, n=100)

        # --- info ---
        self.reset_info = {}

        # --- Ros2 Subscribers ---
        self.camera_sub = self.node.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            self._camera_cb,
            10  # qos_profile_sensor_data
        )

        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan",
            self._lidar_cb,
            10 # qos_profile_sensor_data
        )

        self.pose_sub = self.node.create_subscription(
            Odometry, 
            "/model/vehicle_blue/odometry",
            self._global_pose_cb,
            10 # qos_profile_sensor_data
        )

        # self.vel_sub = self.node.create_subscription(
        #     Twist,
        #     "/cmd_vel",
        #     self._global_vel_cb,
        #     10
        # )

        self.collision_event_sub = self.node.create_subscription(
            Contacts,
            # "/world/mecanum_drive/model/track_model/link/track_link/sensor/walls_contact_sensor/contact",
            "/world/mecanum_drive/model/vehicle_blue/link/chassis/sensor/chassis_contact_sensor/contact",
            self._collision_cb,
            qos_profile_sensor_data
        )


        # --- Ros2 Publisher --- 
        cmd_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Zgodny z domyślnym
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", cmd_qos_profile)

        # --- Gym spaces ---

        # action (norm) shape: [-1,1]x[-1,1]

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: RGB image 256x256x3 (uint8)
        self.img_h = 256
        self.img_w = 256
        self.camera_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.img_h, self.img_w, 3),
            dtype=np.uint8
        )
        # Observation: Lidar 280 (float32)
        self.lidar_l = 280
        self.laser_range = 12.0
        self.lidar_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.lidar_l,),
            dtype=np.float32
        )

        # Observation: coposition
        self.observation_space = spaces.Dict({
                    "image": self.camera_space,
                    "lidar": self.lidar_space,
                })

        # --- Boundaries for actions and similation --
        self.max_lin = max_lin_vel   # maks. linaer velocity [m/s]
        self.max_ang = max_ang_vel  # maks. angular velocity [rad/s]

        self.step_count = 0
        self.max_steps = max_steps_per_episode  # steps per each episode

        # # --- perform some action befor first step:
        # self.reset_before_first_step()



    # --- Ros2 Callbacks --- 
    def _camera_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # img = cv2.resize(img, (self.img_w, self.img_h)) # Images shall be already resized 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.camera_img = img
        except Exception as e:
            self.node.get_logger().warn(f"[Err] Cannot get data from camera:\n{e}")

    def _lidar_cb(self, msg: LaserScan):
        try:
            self.laser = np.array(msg.ranges, dtype=np.float32)
            self.laser = np.clip(self.laser, 0.0, self.laser_range) # clip 
            self.laser = self.laser / self.laser_range
        except Exception as e:
            self.node.get_logger().warn(f"[Err] Cannot get data from lidaer:\n{e}")
       
    def _global_pose_cb(self, msg: Odometry):
        try:
                self.global_pose = np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y
                ])

                self.global_vel = np.array([
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.angular.z
                ])

                self.odom_received = True
        except Exception as e:
            self.node.get_logger().warn(f"[Err] Cannot get data from odometry:\n{e}")

    # def _global_vel_cb(self, msg: Twist):
    #         try: 
    #             self.global_vel = np.array((msg.linear.x, msg.linear.y))
    #         except Exception as e: 
    #             self.node.get_logger().warn(f"[Err] Cannot get data from twist:\n{e}")

    def _collision_cb(self, msg: Contacts):
        if len(msg.contacts) > 0:
            self.collision_flag = True
        else: 
            self.collision_flag = False

    # ------------- GYM API ------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # stop robot
        self._start_gz()
        self._send_cmd(0.0, 0.0)
        if self.episode_count > 0:
            self.node.get_logger().warn(f"> Episode {self.episode_count} finished with {self.step_count} steps.") 
            self.node.get_logger().warn(f"> Rewards: \n\
                                        > velocity: {self.rewards_components_sum[0]} \n\
                                        > trajectory: {self.rewards_components_sum[1]} \n\
                                        > ang_vel: {self.rewards_components_sum[2]} \n\
                                        > collision: {self.rewards_components_sum[3]} \n\
                                        > timeout: {self.rewards_components_sum[4]} \n\
                                        > destin: {self.rewards_components_sum[5]} ")
            self.rewards_components_sum = np.zeros((len(self.rewards)), dtype = np.float32)

        self.episode_count += 1
        self.step_count = 0
        self.collision_flag = False
        self.timeout_flag = False

        self.node.get_logger().warn(f"[Episode|{self.episode_count}] Episode start") 

        self.trajectory.visu_reset()
        # put on random posiition:
        x_st, y_st, yaw_st = self.trajectory.new_rand_pt()
        self.node.get_logger().warn(f"> Starting from new pos: x =  {x_st}, y = {y_st}, yaw = {yaw_st}") 
        self._teleport_car(x_st, y_st, yaw_st)

        obs = self._get_obs_blocking()
        self._stop_gz()

        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        return obs, self.reset_info   

    def step(self, action):
        if self.step_count == 0:
            self.node.get_logger().warn(f"> Episode in progres...") 
        self.step_count += 1
        # scale norm action (-1, 1) to action boundaries
        self._start_gz()
        v_norm = float(np.clip(action[0], 0.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))

        v = v_norm * self.max_lin
        w = w_norm * self.max_ang

        # perform action
        self._send_cmd(v, w)
        
        # wait for get response
        start_time = time.time()
        while time.time() - start_time < self.time_step:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        # get obs  
        obs = self._get_obs()
        self._stop_gz()

        x, y = self.global_pose
        vx, vy, ang_vz = self.global_vel
        self.trajectory.add2trajectory((x, y, vx, vy))
        self.destination_reached_flag  = self.trajectory.check_if_dest_reached(x, y, 
                                                                               fin_line_o = (-9.12, 14.61), 
                                                                               fin_line_i = (-4.4, 14.61), 
                                                                               y_offset = 0.5)

        reward = self._compute_reward(obs)

        # consider termination conditions
        terminated = False
        truncated = False

        # if goal reached -> terminated
        if self.destination_reached_flag:
            terminated = True

        # if max steps reached -> terminated
        if self.step_count >= self.max_steps: 
            self.timeout_flag = True
            truncated = True

        # if collision detected -> terminated 
        if self.collision_flag: terminated = True

        # Terminated more important than truncated
        if terminated: truncated = False 

        # log info 
        info = {}

        return obs, reward, terminated, truncated, info


    def render(self):
        self.trajectory.visu_save(self.LOG_DIR, self.episode_count)
        self.trajectory.traj_save(self.LOG_DIR, self.episode_count)
        self.node.get_logger().warn(f"[Visualisation render finished.]")

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
            self.laser = np.ones((self.lidar_l), dtype=np.float32)

        return {"image": self.camera_img, "lidar": self.laser}

 
    def _get_obs_blocking(self, timeout=2.0):
        waited = 0.0
        dt = 0.05
        while (self.camera_img is None or self.laser is None or not self.odom_received) and waited < timeout:
            rclpy.spin_once(self.node, timeout_sec=dt)
            waited += dt
        return self._get_obs()


    def _compute_reward(self, obs):
        # reward = 0 
        
        # self.rewards = { 'velocity': 1, 'trajectory': -5, 'collision': -15, 'timeout': -5, 'destin': 20}
    
        # get data
        x, y = self.global_pose
        vx, vy, ang_vz = self.global_vel

        # 1 - reward for velocity
        v_xy = np.sqrt(vx*vx + vy*vy)
        # reward += v_xy * self.rewards['velocity']
        self.rewards_components[0] = v_xy * self.rewards['velocity']
        # 2 - reward for distance from desire trajectory
        _, _, dist = self.trajectory.get_dist(x, y) # x_cp, y_cp - closet points on trajectory
        # reward += dist * self.rewards['trajectory'] 
        self.rewards_components[1] = dist * self.rewards['trajectory'] 
        # 3 - reward for collision

        self.rewards_components[2] = np.abs(ang_vz) * self.rewards['ang_vel'] 

        if self.collision_flag:
            self.rewards_components[3] = self.rewards['collision']
            # reward += self.rewards['collision']
            self.node.get_logger().warn(f"[Event] Collision detected.")
        else:
            self.rewards_components[3] = 0.0

        # 4 - reward for timeout
        if self.timeout_flag:
            self.rewards_components[4] = self.rewards['timeout']
            # reward += self.rewards['timeout']
        else:
            self.rewards_components[4] = 0.0

        # 5 - check if destination reached:
        if self.destination_reached_flag:
            self.rewards_components[5] = self.rewards['destin']
            # reward += self.rewards['destin']
            self.node.get_logger().warn(f"[Event] Destination reached.")
        else:
            self.rewards_components[5] = 0.0

        reward = np.sum(self.rewards_components)

        self.rewards_components_sum += self.rewards_components

        return float(reward)

    def _teleport_car(self, x, y, yaw):
        q = self._get_quaternion_from_yaw(yaw)
        req_content = (
            f'name: "vehicle_blue", '
            f'position: {{x: {x}, y: {y}, z: 0.05}}, '
            f'orientation: {{x: {q.x}, y: {q.y}, z: {q.z}, w: {q.w}}}'
        )

        command = [
            'gz', 'service',
            '-s', '/world/mecanum_drive/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req_content
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if "data: true" in result.stdout:
                self.node.get_logger().info(f"[Event] Teleport succes: x={x:.2f}, y={y:.2f}")
                # self._send_cmd(0.0, 0.0)
            else:
                self.node.get_logger().warn(f"[Error] Teleport executed but return false: {result.stdout}")

        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"[Error] Teleport failed: {e.stderr}")


    def _stop_gz(self):
        req_content = 'pause: true'
        command = [
            'gz', 'service',
            '-s', '/world/mecanum_drive/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req_content
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"[Error] Gz pause failed: {e.stderr}")

    def _start_gz(self):
        req_content = 'pause: false'
        command = [
            'gz', 'service',
            '-s', '/world/mecanum_drive/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req_content
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"[Error] Gz pause failed: {e.stderr}")
            


        # req = SetEntityPose.Request() # request object
        # req.state.name = 'vehicle_blue' # object identification
        # req.state.pose.position.x = float(x)
        # req.state.pose.position.y = float(y)
        # req.state.pose.position.z = 0.05 # a little bit over ground to avoid blocking
        # req.state.twist.linear.x = 0.0
        # req.state.twist.linear.y = 0.0
        # req.state.twist.angular.z = 0.0
        # req.state.pose.orientation = self._get_quaternion_from_yaw(yaw)
        # # check if service is available
        # if not self.set_state_client.wait_for_service(timeout_sec=2.0):
        #      self.node.get_logger().error('[Error] Request service is not available.')
        # # send request and block superior fcn until request done
        # future = self.set_state_client.call_async(req)
        # rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        # if future.result() is not None and future.result().success:
        #     self.node.get_logger().info(f"[Event] Teleport successful to x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        # elif future.result() is not None:
        #     self.node.get_logger().error(f"[Error] Teleport failed: {future.result().status_message}")
        # else:
        #     self.node.get_logger().error("[Error] Teleport request timed out or failed to get result.")


    def _get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q
        
    def close(self):
        self._send_cmd(0.0, 0.0)
        self.node.destroy_node()
        rclpy.shutdown()


