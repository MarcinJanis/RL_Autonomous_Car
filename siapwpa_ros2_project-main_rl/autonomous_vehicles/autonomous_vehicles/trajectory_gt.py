import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
# import cv2
import os

import random
import math

class traj_gt: 
    def __init__(self):
        self.tck = None
        self.spline_pts = None

        # -- for visualisation -- 
        self.trajectory = []
        self.velocity = []

    def setup(self, file_pth, n = 200):
        pts = np.loadtxt(file_pth, delimiter=',', skiprows=1)
        tck, u = splprep([pts[:,0], pts[:,1]], s=0.0, per=1)
        self.tck = tck 
        ts = np.linspace(0, 1, n)
        xs, ys = splev(ts, self.tck)
        self.spline_pts = np.column_stack((xs, ys))

    def get_dist(self, x0, y0):
        dx = self.spline_pts[:,0] - x0
        dy = self.spline_pts[:,1] - y0
        i = np.argmin(dx*dx + dy*dy)
        xmin = self.spline_pts[i,0]
        ymin = self.spline_pts[i,1]
        dist = np.sqrt(dx[i]*dx[i] + dy[i]*dy[i])
        return xmin, ymin, dist
    
    def add2trajectory(self, global_pose):
        x, y, vx, vy = global_pose
        self.trajectory.append((x, y))
        self.velocity.append((vx, vy))
    
    def visu_reset(self):
        self.trajectory = []
        self.velocity = []

    def get_trajectory(self):
        if not self.trajectory:
            return None
        trajectory = np.array(self.trajectory)
        return trajectory
    
    def get_velocity(self):
        if not self.velocity:
            return None
        velocity = np.array(self.velocity)
        return velocity
    
    def get_gt(self):
        gt = np.array(self.spline_pts)
        return gt
    
    def visu_show(self):
        traj_pts = self.get_trajectory()
        gt_pts = self.get_gt()
        plt.figure(figsize=(8, 6))
        if not traj_pts is None:
            plt.plot(traj_pts[:,0], traj_pts[:,1], 'r', label='trajectory')
        plt.plot(gt_pts[:,0], gt_pts[:,1], 'g', label='gt')
        
        # plt.plot(x_new, y_new, 'b-', label='Interpolated trajectory')
        # plt.plot((xtest), (ytest), 'go-', label='test point')
        # plt.plot((xmin), (ymin), 'ko-', label='closest point')   
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title("Trajectory")
        plt.show()

    def visu_save(self, dir, step):
        traj_pts = self.get_trajectory()
        gt_pts = self.get_gt()
        plt.figure(figsize=(8, 6))
        if not traj_pts is None:
            plt.plot(traj_pts[:,0], traj_pts[:,1], 'r', label='trajectory')
        plt.plot(gt_pts[:,0], gt_pts[:,1], 'g', label='gt')
       
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title(f'Trajectory (step:{step})')
        plot_dir = os.path.join(dir, f'step_{step}.png')
        plt.savefig(plot_dir, dpi=300, bbox_inches="tight")
        plt.close()
        
    def traj_save(self, dir, step):
      
        traj = self.get_trajectory()
        gt = self.get_gt()
        velo = self.get_velocity()


        max_len = max(len(traj), len(gt))

        padding = np.full((max_len - len(gt), gt.shape[1]), np.nan)
        gt = np.concatenate((gt, padding))

        padding = np.full((max_len - len(traj), traj.shape[1]), np.nan)
        traj = np.concatenate((traj, padding))
        velo = np.concatenate((velo, padding))

        data = np.concatenate((traj, velo, gt), axis = 1)

        file_dir = os.path.join(dir, f'step_{step}.csv')
        np.savetxt(
            file_dir,
            data,
            delimiter=',',
            fmt='%.4f', 
            header='X,Y,Vx,VY,Xref,Yref', 
            comments='' 
        )

    def check_if_dest_reached(self, x, y, fin_line_o = (-9.12, 14.61), fin_line_i = (-4.4, 14.61), y_offset = 0.5):
        # check x coords
        goal_reached = False
        if x > fin_line_o[0] and x < fin_line_i[0]:
            # check y coords with offset
            if y > fin_line_i[1] - y_offset and y < fin_line_i[1] + y_offset:
                goal_reached = True
        return goal_reached


    def new_rand_pt(self):
        idx = random.randint(0, int(0.8 * self.spline_pts.shape[0]))
        # ensure that car will have to through 20% of map
        x, y = self.spline_pts[idx, :] # spawn point
        x_n, y_n = self.spline_pts[idx + 1, :] # next point
        yaw = np.arctan2(y_n - y, x_n - x) # yaw [rad]
        return x, y, yaw
       
# gt = traj_gt()

# print(gt.check_if_dest_reached(-8, 17, fin_line_o = (-9.12, 14.61), fin_line_i = (-4.4, 14.61), y_offset = 0.5))
        

# gt.add2trajectory((-1, 5, 0, 0))
# gt.add2trajectory((2, 3,0,0))
# # gt.visu_show()


# # how to use:

# # Save data to csv

# gt = traj_gt()
# trajectory_points_pth = './siapwpa_ros2_project-main_rl/models/walls/waypoints_il_srodek.csv'
# gt.setup(trajectory_points_pth, n=100)
# gt.add2trajectory((-1, 5, 0, 0))
# gt.add2trajectory((2, 3,0,0))
# # gt.visu_show()
# # gt.visu_save('./siapwpa_ros2_project-main_rl/models/walls',500)
# gt.traj_save('./siapwpa_ros2_project-main_rl/models/walls',500)


# # Get Distance
# gt = traj_gt()
# trajectory_points_pth = './siapwpa_ros2_project-main_rl/models/walls/waypoints_il_srodek.csv'
# gt.setup(trajectory_points_pth, n=100)
# pt = (3, 2)
# gt.add2trajectory((pt[0], pt[1], 0, 0))
# xmin, ymin, dist = dist_val = gt.get_dist(3, 2)
# gt.add2trajectory((xmin, ymin, 0, 0))
# gt.visu_show()
# print(f'dist between:X({pt[0]}, {xmin}),Y({pt[1]},{ymin}): {dist}')
