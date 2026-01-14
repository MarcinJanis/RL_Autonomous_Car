import gymnasium
from gymnasium import spaces
import numpy as np
import cv2
import subprocess
import time
import os

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped, PoseArray

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from geometry_msgs.msg import Quaternion

import trajectory_gt as gt

from rclpy.qos import qos_profile_sensor_data

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class GazeboCarEnv(gymnasium.Env):

    # def __init__(self, time_step : float , rewards: dict, trajectory_points_pth: str, max_steps_per_episode: int, max_lin_vel: float, max_ang_vel: float, render_mode = True, eval = False):
    def __init__(self, time_step : float , rewards: dict, trajectory_points_pth: str, max_steps_per_episode: int, max_lin_vel: float, max_ang_vel: float, render_mode = True):
        super().__init__()

        # self.is_eval_env = eval
   
        # calculation time estimation 
        self.temp_time_step_mean = 0 
        self.temp_time_calc_mean = 0 
        self.t2 = 0 

        # Create subfolder for saving logs from training
        self.LOG_DIR = os.path.join(os.getcwd(), f'./training_logs')
        os.makedirs(name=self.LOG_DIR, exist_ok=True)

        self.log_file_path = os.path.join(self.LOG_DIR, 'episode_summaries.log')
        self.log_file = open(self.log_file_path, 'a')
        self.log_file.write("--- Training Log Start ---\n")

        self.render_mode = True
        self.time_step = time_step
        self.episode_count = 0

        # --- ROS2 node init ---
        if not rclpy.ok(): rclpy.init(args=None)

        self.node = Node(
                         "gym_mecanum_env",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)]
                         )
        # self.set_state_client = self.node.create_client(SetEntityPose, '/world/mecanum_drive/set_pose')
        
        # --- bridge for sensors ---
        self.bridge = CvBridge()

        # --- state variables ---
        # observations state:
        self.camera_img = None
        self.laser = None
        self.global_pose = np.zeros((2), dtype = np.float32)
        self.global_vel = np.zeros((3), dtype = np.float32)

        self.odom_received = False
        self.vel_received = False

        # events flags
        self.collision_flag = False
        self.timeout_flag = False
        self.destination_reached_flag = False

        # rewards state
        self.rewards = rewards 
        self.rewards_components = np.zeros((len(self.rewards)), dtype = np.float32)
        self.rewards_components_sum = np.zeros((len(self.rewards)), dtype = np.float32)

        # init object for distance from trajectory calc
        self.trajectory = gt.traj_gt()
        self.trajectory.setup(trajectory_points_pth, n=300)

        self.last_episode_traj = None
        self.last_episode_vel = None

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

        # self.pose_sub = self.node.create_subscription(
        #     Odometry, 
        #     "/model/vehicle_blue/odometry",
        #     self._global_pose_cb,
        #     10 # qos_profile_sensor_data
        # )


        # self.pose_sub = self.node.create_subscription(
        #     PoseArray, 
        #     "/model/vehicle_blue/pose", 
        #     self._global_pose_cb,
        #     10 # qos_profile_sensor_data
        # )

        self.odom_sub = self.node.create_subscription(
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
            reliability=ReliabilityPolicy.RELIABLE, 
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
            self.node.get_logger().info(f"[Err] Cannot get data from camera:\n{e}")

    def _lidar_cb(self, msg: LaserScan):
        try:
            # self.laser = np.array(msg.ranges, dtype=np.float32)
            # self.laser = np.clip(self.laser, 0.0, self.laser_range) # clip 
            # self.laser = self.laser / self.laser_range


            ranges = np.array(msg.ranges, dtype=np.float32)
            p_spike = 0.007
            mask = np.random.rand(ranges.size) < p_spike
            ranges[mask] = np.random.uniform(msg.range_min, msg.range_max, size=mask.sum()).astype(np.float32)
            
            ranges = np.clip(ranges, 0.0, self.laser_range)
            self.laser = ranges / self.laser_range


        except Exception as e:
            self.node.get_logger().info(f"[Err] Cannot get data from lidaer:\n{e}")
       
    def _global_pose_cb(self, msg: Odometry):
        try:
                if msg.header.frame_id == "world":
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
            self.node.get_logger().info(f"[Err] Cannot get data from odometry:\n{e}")


    # def _global_pose_cb(self, msg: PoseArray):
    #     try:
    #             self.global_pose = np.array([
    #                 msg.pose.position.x,
    #                 msg.poses.position.y
    #             ])
    #             self.odom_received = True 
    #     except Exception as e:
    #         self.node.get_logger().info(f"[Err] Cannot get data from Pose (position):\n{e}")

    # Callback dla Odometry (PRĘDKOŚCI)

    # def _odometry_vel_cb(self, msg: Odometry):
    #     try:
    #             self.global_vel = np.array([
    #                 msg.twist.twist.linear.x,
    #                 msg.twist.twist.linear.y,
    #                 msg.twist.twist.angular.z
    #             ])
    #             self.vel_received = True 
    #     except Exception as e:
    #         self.node.get_logger().info(f"[Err] Cannot get data from Odometry (velocity):\n{e}")


    def _collision_cb(self, msg: Contacts):
        # for c in msg.contacts:
        #     self.node.get_logger().info(f"Contact: {c.collision1} and {c.collision2}")
        if len(msg.contacts) > 0 and self.step_count > 3:
            self.collision_flag = True
        # else: 
        #     self.collision_flag = False

    # ------------- GYM API ------------- #
    # def reset(self, *, seed=None, options=None, eval = None):
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # current_eval_mode = eval if eval is not None else self.is_eval_env

        # stop robot
        # self._start_gz()
        self._send_cmd(0.0, 0.0)
        reset_info = {}
        if self.episode_count > 0:

            log_message = (
                f"\n--- Episode {self.episode_count} Finished ---\n"
                f"> Episode finished with {self.step_count} steps.\n" 
                f"> Mean step time:{self.temp_time_step_mean/(self.step_count+1)}\n" 
                f"> Mean calc per step time: {self.temp_time_calc_mean/(self.step_count+1)}\n" 
                f"> Rewards components:\n"
                f"> velocity: {self.rewards_components_sum[0]} \n"
                f"> trajectory: {self.rewards_components_sum[1]} \n"
                f"> progress: {self.rewards_components_sum[2]} \n"
                f"> collision: {self.rewards_components_sum[3]} \n"
                f"> timeout: {self.rewards_components_sum[4]} \n"
                f"> destin: {self.rewards_components_sum[5]} \n"
                f"-----------------------------------\n"
            )
            if hasattr(self, 'log_file'):
                self.log_file.write(log_message)
                self.log_file.flush()



            self.node.get_logger().info(f"> Episode {self.episode_count} finished with {self.step_count} steps.") 
            self.node.get_logger().info(f"> Mean step time: {self.temp_time_step_mean/(self.step_count+1)}") 
            self.node.get_logger().info(f"> Mean calc per step time: {self.temp_time_calc_mean/(self.step_count+1)}") 
            self.temp_time_step_mean = 0 
            self.temp_time_calc_mean = 0 
            self.node.get_logger().info(f"> Rewards: \n\
                                        > velocity: {self.rewards_components_sum[0]} \n\
                                        > trajectory: {self.rewards_components_sum[1]} \n\
                                        > progress: {self.rewards_components_sum[2]} \n\
                                        > collision: {self.rewards_components_sum[3]} \n\
                                        > timeout: {self.rewards_components_sum[4]} \n\
                                        > destin: {self.rewards_components_sum[5]} ")
            reset_info = {
                'total': np.sum(self.rewards_components_sum),
                'velocity': self.rewards_components_sum[0], 
                'trajectory': self.rewards_components_sum[1], 
                'progression': self.rewards_components_sum[2], 
                'collision': self.rewards_components_sum[3], 
                'timeout': self.rewards_components_sum[4], 
                'destin': self.rewards_components_sum[5]
                }
            

            self.rewards_components_sum = np.zeros((len(self.rewards)), dtype = np.float32)

        self.destination_reached_flag = False
        
        self.odom_received = False
        self.vel_received = False
        self.camera_img = None
        self.laser = None

        self.episode_count += 1
        self.step_count = 0
        self.collision_flag = False
        self.timeout_flag = False

        # change dir for next episode
        if self.episode_count % 2 == 0: 
            self.trajectory.change_dir(clkwise = False)
        else:
            self.trajectory.change_dir(clkwise = True)

        self.node.get_logger().info(f"[Episode|{self.episode_count}] Episode start") 


        self.last_episode_traj = self.trajectory.get_trajectory()
        self.last_episode_vel = self.trajectory.get_velocity()

        self.trajectory.visu_reset()

        # put on random posiition:
        # x_st, y_st, yaw_st = self.trajectory.new_rand_pt(eval=current_eval_mode)
        x_st, y_st, yaw_st = self.trajectory.new_rand_pt()
        self.node.get_logger().info(f"> Starting from new pos: x =  {x_st}, y = {y_st}, yaw = {yaw_st}") 

        # old_pose = self.global_pose.copy() # Zapisujemy poprzednią pozycję odczytaną

        self._teleport_car(x_st, y_st, yaw_st)

        # pętla synchronizacji
        # timeout_start = time.time()
        # max_wait_time = 5.0
        # pos_tolerance = 0.1
        
        # while time.time() - timeout_start < max_wait_time:
        #     rclpy.spin_once(self.node, timeout_sec=0.1) 
            
        #     current_x, current_y = self.global_pose 
        #     distance = np.sqrt((current_x - x_st)**2 + (current_y - y_st)**2)
            
        #     if distance < pos_tolerance:
        #         self.node.get_logger().info(f"[Event] Odometry sync succes after teleport (Distance: {distance:.3f}m).")
        #         break

        # else:
        #     self.node.get_logger().warn(f"[Warning] Odometry failed to sync within {max_wait_time}s. Proceeding with last known position: ({self.global_pose[0]:.2f}, {self.global_pose[1]:.2f}).")
        
        obs = self._get_obs_blocking()
        
        x_start_sync, y_start_sync = self.global_pose
        vx_start_sync, vy_start_sync, ang_vz_start_sync = self.global_vel
        
        self.trajectory.add2trajectory((x_start_sync, y_start_sync, vx_start_sync, vy_start_sync))
        self.node.get_logger().info(f"[Debug] Trajectory start point added: ({x_start_sync:.2f}, {y_start_sync:.2f}).")
        
        
        obs = self._get_obs_blocking()
        self.node.get_logger().info(f"> [Debug] Real new pos: x =  {self.global_pose[0]}, y = {self.global_pose[1]}") 
        # self._stop_gz()

        self.t2 = time.time()
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        return obs, reset_info   

    def step(self, action):

        self.temp_time_calc_mean +=  time.time() - self.t2 
        t1 = time.time() 
        if self.step_count == 0:
            self.node.get_logger().info(f"> Episode in progres...") 
        self.step_count += 1
        # scale norm action (-1, 1) to action boundaries
        # self._start_gz()
        v_norm = float(np.clip(action[0], 0.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))

        v = v_norm * self.max_lin
        w = w_norm * self.max_ang

        # perform action
        self._send_cmd(v, w)
        
        # wait for get response
        start_time =  time.time()
        while time.time() - start_time < self.time_step:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        # get obs  
        obs = self._get_obs()
        # self._stop_gz()

        x, y = self.global_pose
        vx, vy, ang_vz = self.global_vel
        self.trajectory.add2trajectory((x, y, vx, vy))
        self.destination_reached_flag  = self.trajectory.check_if_dest_reached(x, y, 
                                                                               fin_line_o = (-9.12, 14.61), 
                                                                               fin_line_i = (-4.4, 14.61), 
                                                                               y_offset = 0.5)

        

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


        reward = self._compute_reward(obs)


        # log info 
        info = {}
        self.t2 = time.time() # usunąć
        self.temp_time_step_mean += self.t2 - t1 # usunąć
        
        return obs, reward, terminated, truncated, info


    def render(self):
        self.trajectory.visu_save(
            self.LOG_DIR, 
            self.episode_count, 
            traj_override=self.last_episode_traj
        )
        self.trajectory.traj_save(
            self.LOG_DIR, 
            self.episode_count, 
            traj_override=self.last_episode_traj,
            vel_override=self.last_episode_vel
        )
        self.trajectory.traj_save_csv(
            self.LOG_DIR,
            self.episode_count,
            traj_override=self.last_episode_traj
        )
        self.node.get_logger().info(f"[Visualisation render finished.]")

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

 
    def _get_obs_blocking(self, timeout=8.0):
        waited = 0.0
        dt = 0.05
        while (self.camera_img is None or self.laser is None or not self.odom_received or not self.vel_received) and waited < timeout:
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
        self.rewards_components[0] = v_xy/self.max_lin * self.rewards['velocity']
        # 2 - reward for distance from desire trajectory
        _, _, dist, prog = self.trajectory.get_stats(x, y) 
        # reward += dist * self.rewards['trajectory'] 
        self.rewards_components[1] = dist * self.rewards['trajectory'] 
        # 3 - reward for collision

        # self.rewards_components[2] = np.abs(ang_vz) * self.rewards['ang_vel'] 

        # test kary za kręcenie w kółku
        # sit_ratio = np.abs(ang_vz) / (v_xy+ 1e-3)
        # normalized_penalty = np.tanh(sit_ratio / 5.0)
        # self.rewards_components[2] = normalized_penalty * self.rewards['ang_vel']

        self.rewards_components[2] = prog * self.rewards['prog']

        if self.collision_flag:
            self.rewards_components[3] = self.rewards['collision']
            # reward += self.rewards['collision']
            self.node.get_logger().info(f"[Event] Collision detected.")
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
            self.node.get_logger().info(f"[Event] Destination reached.")
        else:
            self.rewards_components[5] = 0.0

        reward = np.sum(self.rewards_components)

        self.rewards_components_sum += self.rewards_components

        return float(reward)

    def _teleport_car(self, x, y, yaw):
        q = self._get_quaternion_from_yaw(yaw)
        req_content = (
            f'name: "vehicle_blue", '
            f'position: {{x: {x}, y: {y}, z: 0.35}}, '
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
            else:
                self.node.get_logger().info(f"[Error] Teleport executed but return false: {result.stdout}")

        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"[Error] Teleport failed: {e.stderr}")

    def _get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q
        
    def _get_time_seconds(self):
        return self.node.get_clock().now().nanoseconds / 1e9

    def close(self):
        self._send_cmd(0.0, 0.0)

        if hasattr(self, 'log_file'):
            self.log_file.write("\n--- Training Log End ---\n")
            self.log_file.close()
            
        self.node.destroy_node()
        rclpy.shutdown()


