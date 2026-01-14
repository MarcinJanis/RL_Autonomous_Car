#xhost +local:root (zezwala lokalnym rootom na wyświetlanie).
# -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix.

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class SensorVisualizer(Node):
    def __init__(self):
        super().__init__('sensor_visualizer_node')
        
        # --- Konfiguracja ---
        # Tematy zgodne z Twoim plikiem gazebo_car_env.py
        self.topic_camera = "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image"
        self.topic_lidar = "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan"
        
        # --- Stan ---
        self.bridge = CvBridge()
        self.current_camera_img = None
        self.current_lidar_img = None
        
        # Parametry mapy BEV (Bird's Eye View)
        self.bev_size = 500       # Rozmiar okna BEV (piksele)
        self.bev_scale = 30.0     # Skala: piksele na metr (zoom)
        self.robot_radius = 5     # Wielkość kropki robota
        
        # --- Subskrybenci ---
        # Używamy qos_profile_sensor_data (Best Effort), bo Gazebo tak wysyła
        self.sub_cam = self.create_subscription(
            Image, 
            self.topic_camera, 
            self.camera_cb, 
            qos_profile_sensor_data
        )

        self.sub_lidar = self.create_subscription(
            LaserScan, 
            self.topic_lidar, 
            self.lidar_cb, 
            qos_profile_sensor_data
        )

        # Timer do odświeżania GUI (30 Hz) - oddzielamy logikę GUI od callbacków ROS
        self.create_timer(0.033, self.display_loop)
        
        self.get_logger().info("Visualizer started. Waiting for data...")

    def camera_cb(self, msg):
        try:
            # Konwersja ROS Image -> OpenCV
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.current_camera_img = img
        except Exception as e:
            self.get_logger().error(f"Cam Error: {e}")

    def lidar_cb(self, msg):
        # Tworzymy pustą czarną mapę
        bev_map = np.zeros((self.bev_size, self.bev_size, 3), dtype=np.uint8)
        
        # Środek mapy (tam gdzie stoi robot)
        cx, cy = self.bev_size // 2, self.bev_size // 2
        
        # Rysujemy robota (niebieska kropka na środku)
        cv2.circle(bev_map, (cx, cy), self.robot_radius, (255, 0, 0), -1) 
        
        # Przetwarzanie punktów LiDARa
        ranges = np.array(msg.ranges)

        # lidar noise
        p = 0.007
        mask = np.random.rand(ranges.size) < p
        ranges[mask] = np.random.uniform(msg.range_min, msg.range_max, mask.sum())
        
        # Filtrujemy 'inf' (nieskończoność) i błędne pomiary
        valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))[0]
        
        if len(valid_indices) > 0:
            # Obliczamy kąty dla każdego poprawnego pomiaru
            # angle = min_angle + index * increment
            angles = msg.angle_min + valid_indices * msg.angle_increment
            dists = ranges[valid_indices]
            
            # Zamiana współrzędnych biegunowych (kąt, odległość) na kartezjańskie (x, y)
            # W ROS X to przód, Y to lewo.
            # W OpenCV X to prawo, Y to dół.
            # Musimy to zmapować: 
            # ROS x -> OpenCV -y (góra)
            # ROS y -> OpenCV -x (lewo)
            # Ale prościej matematycznie:
            
            x_ros = dists * np.cos(angles)
            y_ros = dists * np.sin(angles)
            
            # Konwersja na piksele obrazu (obrót o -90 stopni, żeby przód był na górze ekranu)
            # Screen X = Center X - Y_ROS * scale
            # Screen Y = Center Y - X_ROS * scale
            px = (cx - y_ros * self.bev_scale).astype(int)
            py = (cy - x_ros * self.bev_scale).astype(int)
            
            # Filtrowanie punktów, które wypadają poza obraz
            mask = (px >= 0) & (px < self.bev_size) & (py >= 0) & (py < self.bev_size)
            
            # Rysowanie punktów (zielone kropki)
            # Używamy pętli lub fancy indexing (fancy jest szybsze w numpy)
            bev_map[py[mask], px[mask]] = (0, 255, 0)

        self.current_lidar_img = bev_map

    def display_loop(self):
        # Jeśli nie mamy jeszcze danych, nic nie rób
        if self.current_camera_img is None and self.current_lidar_img is None:
            return

        # Przygotuj obrazy (jeśli któregoś brakuje, wstaw czarny prostokąt)
        if self.current_camera_img is not None:
            cam_view = self.current_camera_img
        else:
            cam_view = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.current_lidar_img is not None:
            lidar_view = self.current_lidar_img
        else:
            lidar_view = np.zeros((self.bev_size, self.bev_size, 3), dtype=np.uint8)

        # Skalowanie kamery, żeby pasowała wysokością do Lidara (opcjonalne, dla estetyki)
        # Tutaj po prostu skleimy je horyzontalnie, jeśli mają różną wysokość, resize kamery
        target_h = self.bev_size
        cam_resized = cv2.resize(cam_view, (target_h, target_h)) # Skalujemy kamerę do 500x500
        
        # Łączenie obrazów (Kamera po lewej, Lidar BEV po prawej)
        combined_view = np.hstack((cam_resized, lidar_view))
        
        # Wyświetlanie
        cv2.imshow("Robot Sensory Debug", combined_view)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    vis_node = SensorVisualizer()
    
    try:
        rclpy.spin(vis_node)
    except KeyboardInterrupt:
        pass
    finally:
        vis_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()