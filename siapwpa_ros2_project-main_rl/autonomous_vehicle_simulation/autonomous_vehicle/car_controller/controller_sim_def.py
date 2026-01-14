import cv2
import numpy as np 

from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts

import time 

class CarController(Node):
    def __init__(self, step_time, Vmax_lin, Vmax_ang, lidar_max_range = 12, lidar_n_beans = 280):
        super().__init__('rl_car_controller')
        # --- Node timer --- #
        self.timer = self.create_timer(step_time, self.step)

        # --- model --- #
        self.model = None

        # --- Car parameters --- #
        self.Vmax_lin = Vmax_lin
        self.Vmax_ang = Vmax_ang

        self.lidar_max_range = lidar_max_range
        self.lidar_n_beans = lidar_n_beans

        # --- Inner state --- # 
        self.camera_img = None
        self.lidar_scan = None

        self.pose_act = (0.0, 0.0)
        self.vel_act = (0.0, 0.0) 

        self.trajectory = []
        self.velocity = []
        self.collisions = []
        self.mem_sample_max = 200

        self.map_pts = np.loadtxt('/home/developer/ros2_ws/src/autonomous_vehicles/autonomous_vehicles/car_controller/data/map.csv', delimiter=',', dtype=float, skiprows = 1)
    
        self.collision_event = False
        self.full_lap_event = False
        self.full_lap_prev_event = False

        self.start_time = 0.0
        self.act_lap_time = 0.0
        self.best_lap_time = float(np.inf)

        # --- ROS subscribers --- #
        self.bridge = CvBridge() # create birdge object 

        # subscribers for vehicle observations
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

        # subscribers for additional information
        self.ros_odom_sub = self.create_subscription(
            Odometry, 
            "/model/vehicle_blue/odometry",
            self._global_pose_cb,
            10 
        )

        self.ros_collision_sub = self.create_subscription(
            Contacts,
            "/world/mecanum_drive/model/vehicle_blue/link/chassis/sensor/chassis_contact_sensor/contact",
            self._collision_cb,
            10
        )

        # --- ROS publisher --- #

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # --- ROS callbacks --- #

    def _camera_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # img = cv2.resize(img, (self.img_w, self.img_h)) # Images shall be already resized 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.camera_img = img
        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from camera:\n{e}")


    def _lidar_cb(self, msg: LaserScan):
        try:
            # self.lidar_scan = np.array(msg.ranges, dtype=np.float32)
            # self.lidar_scan = np.clip(self.lidar_scan, 0.0, self.lidar_max_range) # clip 
            # self.lidar_scan = self.lidar_scan / self.lidar_max_range



            scan = np.array(msg.ranges, dtype=np.float32)

            p = 0.007
            mask = (np.random.rand(scan.size) < p)
            scan[mask] = np.random.uniform(0.0, self.lidar_max_range, size=mask.sum()).astype(np.float32)

            scan = np.clip(scan, 0.0, self.lidar_max_range)
            self.lidar_scan = scan / self.lidar_max_range

        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from lidaer:\n{e}")

    def _global_pose_cb(self, msg: Odometry):
        try:
            if msg.header.frame_id == "world":
                self.pose_act = ((msg.pose.pose.position.x, msg.pose.pose.position.y))
                self.vel_act = ((msg.twist.twist.linear.x, msg.twist.twist.angular.z))  
        except Exception as e:
            self.get_logger().warn(f"[Err] Cannot get data from odometry:\n{e}")

    def _collision_cb(self, msg: Contacts):
        if len(msg.contacts) > 0:
            self.collision_event = True
            self.get_logger().warn(f"[Event] Collision!")
    
    def _send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    # --- Functions --- 
    def setup(self, model,  mem_sample_max = 200):
        self.model = model 
        self.mem_sample_max = mem_sample_max

    def act(self):
        if self.camera_img is None or self.lidar_scan is None:
            self.get_logger().warn("[Warning] Waiting for sensors data...")
            return
        
        obs = {"image": self.camera_img, "lidar": self.lidar_scan}
        action, _ = self.model.predict(obs, deterministic=True)
        v_norm, w_norm = action
        v = v_norm * self.Vmax_lin
        w = w_norm * self.Vmax_ang
        v = np.clip(v, -self.Vmax_lin, self.Vmax_lin)
        w = np.clip(w, -self.Vmax_ang, self.Vmax_ang)
        self._send_cmd(v, w)
        
    
    def log(self):
        # Add pt to trajectory
        self.trajectory.append(self.pose_act)
        self.velocity.append(self.vel_act)
        if self.collision_event:
            self.collisions.append(self.pose_act)
            self.collision_event = False
        if not self.mem_sample_max is None:
            if len(self.trajectory) > self.mem_sample_max: self.trajectory.pop(0)
            if len(self.velocity) > self.mem_sample_max: self.velocity.pop(0)

    def check_if_dest_reached(self, x, y, fin_line_o = (-9.12, 15), fin_line_i = (-4.4, 15), y_offset = 0.5):
        # check x coords
        goal_reached = False
        if x > fin_line_o[0] and x < fin_line_i[0]:
            # check y coords with offset
            if y > fin_line_i[1] - y_offset and y < fin_line_i[1] + y_offset:
                goal_reached = True
        return goal_reached

    def step(self):
        self.act() # Get obs, inference, send control cmd
        if not self.pose_act == (0.0, 0.0):
            self.log() # log additional data 
            self.visu(speed_grad=True, draw_collision=True) # visualisation

        # calc lap's time
        self.full_lap_event = self.check_if_dest_reached(self.pose_act[0], self.pose_act[1])

        if self.full_lap_event and not self.full_lap_prev_event:
            self.act_lap_time = time.time() - self.start_time
            if self.act_lap_time < self.best_lap_time:
                self.best_lap_time = self.act_lap_time  
            # self.full_lap_prev_event = False
            # self.full_lap_event = False
            self.start_time = time.time()
        self.full_lap_prev_event = self.full_lap_event
        # self.start_time = 0.0
        # self.act_lap_time = 0.0
        # self.best_lap_time = 0.0


    def visu(self, speed_grad=True, draw_collision = True):
        if len(self.trajectory) < 2:
            return

        # --- Configuration of display ---
        img_h, img_w = 800, 1000 
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Set map limits
        all_x = np.concatenate([self.map_pts[:, 0], self.map_pts[:, 2]])
        all_y = np.concatenate([self.map_pts[:, 1], self.map_pts[:, 3]])
        
        min_x, max_x = np.nanmin(all_x) - 2, all_x.max() + 2
        min_y, max_y = np.nanmin(all_y) - 2, all_y.max() + 2
        
        range_x = max_x - min_x
        range_y = max_y - min_y

        # Scaling, keep correct proportion
        scale_x = img_w / range_x
        scale_y = img_h / range_y
        scale = min(scale_x, scale_y) 

        # offsets to center map
        offset_x = (img_w - range_x * scale) / 2
        offset_y = (img_h - range_y * scale) / 2

        # --- Additional fcn: [m] -> [px] ---
        def _to_pix(x, y):
            # X: shift with min_x, scale, add margin
            px = int((x - min_x) * scale + offset_x)
            # Y: revers Y axis, scale, add margin
            py = int(img_h - ((y - min_y) * scale + offset_y)) 
            return (px, py)

        # --- Draw map ---
        pts_in = [_to_pix(x, y) for x, y in zip(self.map_pts[:, 0], self.map_pts[:, 1])]
        pts_out = [_to_pix(x, y) for x, y in zip(self.map_pts[:, 2], self.map_pts[:, 3])]
        pts_center = [_to_pix(x, y) for x, y in zip(self.map_pts[:, 4], self.map_pts[:, 5])]
        # road, lines
        cv2.polylines(canvas, [np.array(pts_in)], False, (255, 255, 255), 1)
        cv2.polylines(canvas, [np.array(pts_out)], False, (255, 255, 255), 1)
        cv2.polylines(canvas, [np.array(pts_center)], False, (0, 255, 255), 1) 
        # finish line
        p1 = _to_pix(-4, 15)
        p2 = _to_pix(-9.8, 15)
        cv2.line(canvas, p1, p2, (255, 0, 0), 2) # BGR: Blue

        # --- Draw trajectory --- 
        traj_pixels = [_to_pix(pt[0], pt[1]) for pt in self.trajectory]
        
        if speed_grad and len(self.velocity) > 0:
            vels = np.array(self.velocity)[:, 0] 
            # v_min, v_max = vels.min(), vels.max()
            
            if self.Vmax_lin == 0: div = 1  # to avoid div with 0
            else: div = self.Vmax_lin

            # Draw lines with segments
            for i in range(len(traj_pixels) - 1):
                pt1 = traj_pixels[i]
                pt2 = traj_pixels[i+1]
                
                curr_v = vels[i]
                norm_v = curr_v / div
                
                hue = int((1 - norm_v) * 120) 

                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
                
                cv2.line(canvas, pt1, pt2, color, 3)
                
        else:
            cv2.polylines(canvas, [np.array(traj_pixels)], False, (0, 0, 255), 2)


        # --- draw collision ---
        if draw_collision:
            for pt in self.collisions:
                cv2.circle(canvas, _to_pix(pt[0], pt[1]), 10, (255, 0, 255), 1)

        # --- draw car ---
        if len(traj_pixels) > 3:
            cv2.arrowedLine(
                            canvas,
                            traj_pixels[-3],
                            traj_pixels[-1],
                            (255, 255, 255),
                            4,
                            tipLength=0.2
            )

        # --- Show ---
        # add velocity and time info text
        if len(self.velocity) > 0:
            v_now = self.velocity[-1][0]
            cv2.putText(canvas, f"V: {v_now:.2f} [m/s]", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            
            cv2.putText(canvas, f"last lap: {self.act_lap_time:.2f} [s], best lap: {self.best_lap_time:.2f} [s]", (280, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("Trajectory", canvas)
        cv2.waitKey(1) 
      
    def visu_quit(self):
        cv2.destroyAllWindows()




    # def visu(self, speed_grad = True):
    #     if len(self.trajectory) < 2:
    #         return
    #     plt.style.use('dark_background')
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     # --- map contures ---
    #     ax.plot(self.map_pts[:,0], self.map_pts[:,1], 'w-', linewidth=1)
    #     ax.plot(self.map_pts[:,2], self.map_pts[:,3], 'w-', linewidth=1)
    #     ax.plot(self.map_pts[:,4], self.map_pts[:,5], 'y--', alpha=0.5)
        
    #     # --- finish line --- 
    #     ax.plot((-4, -9.8), (15, 15), 'b--', linewidth=2)
        
    #     # --- trajectory --- 
    #     traj = np.array(self.trajectory) # shape (N, 2)
        
    #     if speed_grad:
    #         # ----- Option 2: LineCollection ----
    #         vel_vct = np.array(self.velocity) # shape (N, 2)
    #         speed = vel_vct[:, 0] 
    #         points = traj.reshape(-1, 1, 2)
    #         segments = np.concatenate([points[:-1], points[1:]], axis=1) # segments: [(x0,y0)->(x1,y1), (x1,y1)->(x2,y2)...]
            
    #         norm = plt.Normalize(speed.min(), speed.max())
    #         lc = LineCollection(segments, cmap='turbo', norm=norm)
            
    #         lc.set_array(speed[:-1]) # set collors according to speed
    #         lc.set_linewidth(3)

    #         ax.add_collection(lc)

    #         # ---- add color legend --- 
    #         cbar = fig.colorbar(lc, ax=ax, label="Velocity  [m/s]")
    #     else:
    #         # ---- Option 1: Standard ----
    #         ax.plot(traj[:,0], traj[:,1], 'r-')
    #     # --- formatting ---
    #     ax.minorticks_on()
    #     ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    #     ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    #     ax.grid(which='major', color='#444444', linestyle='-', linewidth=0.8)
    #     ax.grid(which='minor', color='#222222', linestyle=':', linewidth=0.5)
        
    #     all_x = np.concatenate([self.map_pts[:,0], self.map_pts[:,2]])
    #     all_y = np.concatenate([self.map_pts[:,1], self.map_pts[:,3]])
    #     ax.set_xlim(all_x.min() - 2, all_x.max() + 2)
    #     ax.set_ylim(all_y.min() - 2, all_y.max() + 2)
        
    #     ax.set_aspect('equal')

    #     plt.show()