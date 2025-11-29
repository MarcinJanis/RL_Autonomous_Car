import os
import sys
import time
import numpy as np

# 🔧 DODANE: wrzuć lokalny katalog z kodem na początek PYTHONPATH
THIS_DIR = os.path.dirname(__file__)
LOCAL_PKG_DIR = os.path.join(THIS_DIR, "autonomous_vehicles", "autonomous_vehicles")
sys.path.insert(0, LOCAL_PKG_DIR)

import gazebo_car_env as gmod
print("gazebo_car_env module file (LOCAL):", gmod.__file__)

from gazebo_car_env import GazeboCarEnv


def main():
    env = GazeboCarEnv()
    obs = env.reset()
    print("Initial obs shape:", obs.shape)

    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"step {i}: reward={reward}, done={done}")
        time.sleep(0.05)
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
