import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


class traj_gt: 
    def __init__(self, file_pth):
        self.file_pth = file_pth

    def setup(self):
        pts = np.loadtxt(self.file_pth, delimiter=',', skiprows=1)
        tck, u = splprep([pts[:,0], pts[:,1]], s=0.0, per=1)
        self.tck = tck 

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


    def get_dist(self, x0, y0, n):
        ts = np.linspace(0, 1, n)
        xs, ys = splev(ts, self.tck)
        spline_pts = np.column_stack((xs, ys))
        dx = spline_pts[:,0] - x0
        dy = spline_pts[:,1] - y0
        i = np.argmin(dx*dx + dy*dy)
        xmin = spline_pts[i,0]
        ymin = spline_pts[i,1]
        return xmin, ymin, i
    
# pth = './siapwpa_ros2_project-main_rl/models/walls/waypoints_il.csv'
# gt = traj_gt(pth)

# gt.setup()



# Wynik to macierz (array):