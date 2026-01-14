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
   
    def destroy_node(self):
        super().destroy_node()

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

    def destroy_node(self):
        super().destroy_node()