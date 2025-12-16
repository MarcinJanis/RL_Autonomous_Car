import numpy as np
import matplotlib
matplotlib.use('Agg')
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

        self.pt_dist = 0

        self.prog_prev = 0

        self.x_start = 0
        self.y_start = 0

        self.i_max_achieved = 0

    def setup(self, file_pth, n = 300):
        pts = np.loadtxt(file_pth, delimiter=',', skiprows=1)
        tck, u = splprep([pts[:,0], pts[:,1]], s=0.0, per=1)
        self.tck = tck
        ts = np.linspace(0, 1, n)
        xs, ys = splev(ts, self.tck)
        self.spline_pts = np.column_stack((xs, ys))

        self.pt_dist = np.sqrt(((self.spline_pts[0,0])-(self.spline_pts[1,0]))**2 + ((self.spline_pts[0,1])-(self.spline_pts[1,1]))**2)

        dx = self.spline_pts[:,0] - self.x_start
        dy = self.spline_pts[:,1] - self.y_start
        i_start = np.argmin(dx*dx + dy*dy)
        self.prog_prev = i_start * self.pt_dist


    # def get_stats(self, x0, y0):

    #     # find idx of actual point in spline
    #     dx = self.spline_pts[:,0] - x0
    #     dy = self.spline_pts[:,1] - y0
    #     i_act = np.argmin(dx*dx + dy*dy)

    #     # calc distance form traj
    #     xmin = self.spline_pts[i_act,0]
    #     ymin = self.spline_pts[i_act,1]
    #     dist = np.sqrt(dx[i_act]*dx[i_act] + dy[i_act]*dy[i_act])

    #     # calc progression
    #     prog = i_act * self.pt_dist
    #     prog_relative = prog - self.prog_prev


    #     # print(f'prev prog: {self.prog_prev} -> aact prog: {prog}')
    #     self.prog_prev = prog
    #     return xmin, ymin, dist, prog_relative
    
    # próba nowej definicji progresu
    def get_stats(self, x0, y0):

        # find idx of actual point in spline
        dx = self.spline_pts[:,0] - x0
        dy = self.spline_pts[:,1] - y0
        i_act = np.argmin(dx*dx + dy*dy)

        L = 1 #ilość punktów do przewidywania do przodu
        i_target = min(self.spline_pts.shape[0] - 1, i_act + L)

        # calc distance form traj
        xmin = self.spline_pts[i_target,0]
        ymin = self.spline_pts[i_target,1]
        # dist = np.sqrt(dx[i_act]*dx[i_act] + dy[i_act]*dy[i_act])

        dx_target = xmin - x0
        dy_target = ymin - y0
        dist = np.sqrt(dx_target*dx_target + dy_target*dy_target)

        if i_act > self.i_max_achieved:
            self.i_max_achieved = i_act

        # calc progression
        prog = self.i_max_achieved * self.pt_dist
        prog_relative = prog - self.prog_prev


        # print(f'prev prog: {self.prog_prev} -> aact prog: {prog}')
        self.prog_prev = prog
        return xmin, ymin, dist, prog_relative

    #     dx = self.spline_pts[:,0] - x0
    #     dy = self.spline_pts[:,1] - y0
    #     i = np.argmin(dx*dx + dy*dy)
    #     xmin = self.spline_pts[i,0]
    #     ymin = self.spline_pts[i,1]
    #     dist = np.sqrt(dx[i]*dx[i] + dy[i]*dy[i])

    # def get_progress(self, x, y):
    #     self.prev_cast_x = 0
    #     self.prev_cast_y = 0


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
        plt.plot(gt_pts[:,0], gt_pts[:,1], 'g*-', label='gt')

        # plt.plot(x_new, y_new, 'b-', label='Interpolated trajectory')
        # plt.plot((xtest), (ytest), 'go-', label='test point')
        # plt.plot((xmin), (ymin), 'ko-', label='closest point')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title("Trajectory")
        plt.show()

    def visu_save(self, dir, step, traj_override = None):
        if traj_override is not None:
            traj_pts = traj_override
        else:
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

    def traj_save(self, dir, step, traj_override=None, vel_override=None):
        if traj_override is not None:
            traj = traj_override
        else:
            traj = self.get_trajectory()

        if vel_override is not None:
            velo = vel_override
        else:
            velo = self.get_velocity()

        gt = self.get_gt()
        # velo = self.get_velocity()


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


    def new_rand_pt(self, set_start_pt = True, eval= False):
        if eval:
            idx = 473
        else:
            idx = random.randint(0, int(0.8 * self.spline_pts.shape[0]))

        # ensure that car will have to through 20% of map
        x, y = self.spline_pts[idx, :] # spawn point
        x_n, y_n = self.spline_pts[idx + 1, :] # next point
        yaw = np.arctan2(y_n - y, x_n - x) # yaw [rad]
        if set_start_pt:
                    self.x_start = x
                    self.y_start = y
                    self.prog_prev = idx * self.pt_dist
                    self.i_max_achieved = idx
                    print(f'set start pt: {self.x_start}, {self.y_start}')
        return x, y, yaw
    

    def traj_save_csv(self, log_dir, episode_count, traj_override=None):

        if traj_override is not None:
            trajectory_data = traj_override
        else:
            trajectory_data = self.trajectory

        if not trajectory_data:
            print(f"[Warning] Episode {episode_count}: Trajectory data is empty, skipping CSV save.")
            return

        header = ['x', 'y', 'vx', 'vy']
        
        data_array = np.array(trajectory_data)

        filename = os.path.join(log_dir, f'trajectory_data_{episode_count}.csv')

        try:
            np.savetxt(
                filename,
                data_array,
                fmt='%.6f',
                delimiter=',',
                header=','.join(header),
                comments=''
            )
            # print(f"[INFO] Trajectory data saved to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save trajectory CSV for episode {episode_count}: {e}")






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


# Get Distance
# gt = traj_gt()
# trajectory_points_pth = './siapwpa_ros2_project-main_rl/models/walls/waypoints_il_srodek.csv'
# gt.setup(trajectory_points_pth, n=300)

# x, y, yaw = gt.new_rand_pt()

# # x = x + 1
# # y = y + 1.5

# print(f'pt1: {x}, {y}')
# gt.add2trajectory((x, y, 0, 0))
# # xmin, ymin, dist, _ = dist_val = gt.get_dist(x, y)
# # gt.add2trajectory((xmin, ymin, 0, 0))

# x2 = x + 3
# y2 = y + 2
# gt.add2trajectory((x2, y2, 0, 0))
# print(f'pt2: {x2}, {y2}')

# print('first progression')
# xmin, ymin, dist, prog = gt.get_stats(x2, y2)
# # print(f'pt2 cast for: {xmin}, {ymin}')
# gt.add2trajectory((xmin, ymin, 0, 0))

# print(f'progression: {prog}')
# gt.visu_show()


# print(f'dist between:X({pt[0]}, {xmin}),Y({pt[1]},{ymin}): {dist}')