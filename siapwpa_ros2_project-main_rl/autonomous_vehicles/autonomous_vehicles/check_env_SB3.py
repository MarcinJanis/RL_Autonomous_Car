import os 

import torch.nn as nn
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from net_agent.net_v1 import AgentNet as agent
from gazebo_car_env import GazeboCarEnv as gazebo_env

import wandb
from wandb.integration.sb3 import WandbCallback

# Parameters
ENV_PARALLEL = 4 # How many envornment shall work parallel during training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 1000000 # Total steps
EVAL_STEPS = 10000 # Evaluation after this amount of steps
MAX_STEPS_PER_EPISODE = 5000 # Steps per episoed (max)

rewards =  { 'velocity': 1, 'trajectory': -5, 'collision': -15, 'timeout': -5, 'destin': 20 }
# velocity - reward for velocity to motive car to explore
# trajectory - punishment for distance from desired trajectory 
# collision - punishment for collision
# timeout - punishment for exceed max steps before reached goal
# destin - reward for reach goal  

trajectory_goal = '/home/developer/ros2_ws/src/models/walls/waypoints_prawy_srodek.csv'

# boundaries for car
max_linear_velocity = 3.0
max_angular_velocity = 2.0

# --- WandB intergation ---
#TODO: dodać hiperparametry o których chcemy zachować informację!! Przykładowe: 
config = {
   "policy_type": "MultiInputPolicy",
    "total_timesteps": TOTAL_STEPS,
    "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
}

wandb.login()

run = wandb.init(
    project="RL_Autonomous_Car",
    entity="deep-neural-network-course",
    name='RL_TestRun_1', # Name
    settings=wandb.Settings(save_code=False),
    config=config,
    sync_tensorboard=True,
    monitor_gym=False,
    save_code=False,
    mode='offline'
)

wandb_callback = WandbCallback(
    model_save_path=None,   
    # save_model=False,      
    verbose=2,
    # gradient_save_freq=0
)

# --- Init Environment ---
env = gazebo_env(rewards = rewards, 
                trajectory_points_pth = trajectory_goal, 
                max_steps_per_episode = MAX_STEPS_PER_EPISODE, 
                max_lin_vel = max_linear_velocity,
                max_ang_vel = max_angular_velocity)

# check compiliance

print("Checking environment...")
check_env(env, warn=True)
del env 
print("Environment checked.")
