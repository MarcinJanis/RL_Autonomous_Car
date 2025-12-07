import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import cv2

class traj_gt: 
    def __init__(self):
        self.tck = None
        self.spline_pts = None

        # -- for visualisation -- 
        self.trajectory = []


    def setup(self, file_pth, n = 200):
        pts = np.loadtxt(file_pth, delimiter=',', skiprows=1)
        tck, u = splprep([pts[:,0], pts[:,1]], s=0.0, per=1)
        self.tck = tck 
        ts = np.linspace(0, 1, n)
        xs, ys = splev(ts, self.tck)
        self.spline_pts = np.column_stack((xs, ys))


        # u_new = np.linspace(u.min(), u.max(), 1000)
        # x_new, y_new = splev(u_new, tck, der=0)
        # self.traj = np.vstack((x_new, y_new)).T

        # xtest = 6
        # ytest = 2
        # xmin, ymin, i = self.get_dist(xtest, ytest)

        # plt.figure(figsize=(8, 6))
        # plt.plot(pts[:,0], pts[:,1], 'ro', label='Orginal pts')
        # plt.plot(x_new, y_new, 'b-', label='Interpolated trajectory')
        # plt.plot((xtest), (ytest), 'go-', label='test point')
        # plt.plot((xmin), (ymin), 'ko-', label='closest point')   
        # plt.legend()
        # plt.grid(True)
        # plt.axis('equal')
        # plt.title("Interpolacja zamkniętej trajektorii (B-Spline)")
        # plt.show()


    def get_dist(self, x0, y0):
        dx = self.spline_pts[:,0] - x0
        dy = self.spline_pts[:,1] - y0
        i = np.argmin(dx*dx + dy*dy)
        xmin = self.spline_pts[i,0]
        ymin = self.spline_pts[i,1]
        dist = np.sqrt(dx[i]*dx[i] + dy[i]*dy[i])
        return xmin, ymin, dist
    
    def add2trajectory(self, x, y):
        self.trajectory.append((x, y))
    
    
    def visu_reset(self):
        self.trajectory = []

    def get_trajectory(self):
        trajectory = np.array(self.trajectory)
        return trajectory
    
    def get_gt(self):
        gt = np.array(self.spline_pts)
        return gt
    
    def visu_show(self, n=0):
        traj_pts = self.get_trajectory()
        gt_pts = self.get_gt()
        plt.figure(figsize=(8, 6))
        plt.plot(gt_pts[:,0], gt_pts[:,1], 'g', label='gt')
        plt.plot(traj_pts[:,0], traj_pts[:,1], 'r', label='trajectory')
        # plt.plot(x_new, y_new, 'b-', label='Interpolated trajectory')
        # plt.plot((xtest), (ytest), 'go-', label='test point')
        # plt.plot((xmin), (ymin), 'ko-', label='closest point')   
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title("Trajectory")
        plt.show()



gt = traj_gt()
trajectory_points_pth = './siapwpa_ros2_project-main_rl/models/walls/waypoints_il.csv'
gt.setup(trajectory_points_pth, n=100)
