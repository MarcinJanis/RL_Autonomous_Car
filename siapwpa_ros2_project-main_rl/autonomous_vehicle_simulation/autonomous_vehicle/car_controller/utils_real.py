import numpy as np


class wheelSpeedDistributor():
  # Example of car config dict: car_config = {'r':1, 'lx':2, 'ly':2, 'v_lin_max':2.0, 'v_ang_max':2.0}
  def __init__(self, car_config):
    self.T = (1/car_config['r']) * np.array([[1, -1, -(car_config['lx'] + car_config['ly'])],
                                             [1, 1, (car_config['lx'] + car_config['ly'])],
                                             [1, 1, -(car_config['lx'] + car_config['ly'])],
                                             [1, -1, (car_config['lx'] + car_config['ly'])]
                                            ])
    # print(self.T.shape)
    self.v_lin_max = car_config['v_lin_max']
    self.v_ang_max = car_config['v_ang_max']

  def allocate_wheelspeed(self, v_lin, v_ang):

      # | w1 |         |  1  -1  -(lx+ly) |   | vx |
      # | w2 | = (1/r) |  1   1   (lx+ly) | * | vy |
      # | w3 |         |  1   1  -(lx+ly) |   | wz |
      # | w4 |         |  1  -1   (lx+ly) |

      #  w[0] - left front 
      #  w[1] - right front
      #  w[2] - right back 
      #  w[3] - left back
  
    v_lin = np.clip(v_lin, -self.v_lin_max, self.v_lin_max)
    v_ang = np.clip(v_ang, -self.v_ang_max, self.v_ang_max)
    robot_vel = np.array([[v_lin], [0], [v_ang]])
    w = self.T @ robot_vel
    return w.flatten()
  

# how to use:
# a = wheelSpeedDistributor(car_config = {'r':1, 'lx':2, 'ly':2, 'v_lin_max':2.0, 'v_ang_max':2.0})
# w = a.allocate_wheelspeed(3, 2)
# print(w)