import os
import sys

import torch.nn as nn
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import net_agent.net_v1 as agent

from gazebo_car_env import GazeboCarEnv as gazebo_env

import wandb
from wandb.integration.sb3 import WandbCallback
from training_callbacks import wandb_callback_extra, EnvEvalCallback

# Parameters
ENV_PARALLEL = 1 # How many envornment shall work parallel during training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 1000000 # Total steps
EVAL_STEPS = 10000 # Evaluation after this amount of steps
MAX_STEPS_PER_EPISODE = 1500 # Steps per episoed (max)
TIME_STEP = 0.1 # [s]

# rewards =  { 'velocity': 0.1, 'trajectory': -0.001, 'prog': 1, 'collision': -100, 'timeout': -100, 'destin': 100 } # to są parametry co były do uczenia 9 i 10 
rewards =  { 'velocity': 0.1, 'trajectory': -0.1, 'prog': 1, 'collision': -100, 'timeout': -100, 'destin': 100 }
# velocity - reward for velocity to motive car to explore
# trajectory - punishment for distance from desired trajectory 
# collision - punishment for collision
# timeout - punishment for exceed max steps before reached goal
# destin - reward for reach goal  

trajectory_goal = '/home/developer/ros2_ws/src/models/walls/waypoints_prawy_srodek.csv'

pretrained_model_pth = '/home/developer/ros2_ws/src/autonomous_vehicles/models/test_run11/model_e13_rp163_15.zip' # set to None if init model from zero 

# boundaries for car
max_linear_velocity = 6.0
max_angular_velocity = 2.0

# --- WandB intergation ---
#TODO: dodać hiperparametry o których chcemy zachować informację!! Przykładowe: 
config = {
   "policy_type": "MultiInputPolicy",
    "total_timesteps": TOTAL_STEPS,
    "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
    "rewards" : rewards
}

wandb.login()

run = wandb.init(
    project="RL_Autonomous_Car",
    entity="deep-neural-network-course",
    name='RL_TestRun_11', # Name
    settings=wandb.Settings(save_code=False),
    config=config,
    sync_tensorboard=True,
    monitor_gym=False,
    save_code=False,
    # mode='online'
    mode='offline'
)

# --- Init Environment ---
env = gazebo_env(time_step = TIME_STEP,
                 rewards = rewards, 
                 trajectory_points_pth = trajectory_goal, 
                 max_steps_per_episode = MAX_STEPS_PER_EPISODE, 
                 max_lin_vel = max_linear_velocity,
                 max_ang_vel = max_angular_velocity,
                 render_mode=True)

# check compiliance
print("Checking environment...")
check_env(env, warn=True)
del env 
print("Environment checked.")


# Environment vectorization - for parallel training
env_id = lambda: gazebo_env(time_step = TIME_STEP,
                            rewards = rewards, 
                            trajectory_points_pth = trajectory_goal, 
                            max_steps_per_episode = MAX_STEPS_PER_EPISODE, 
                            max_lin_vel = max_linear_velocity,
                            max_ang_vel = max_angular_velocity,
                            render_mode=True,
                            )

# Evaluation callback

# vec_env = make_vec_env(lambda: env_id(eval= False), n_envs=ENV_PARALLEL)
vec_env = make_vec_env(env_id, n_envs=ENV_PARALLEL)

# -> Faster trainging (GPU waits until simulation ennds on CPU, its better to use all of available CPU threads 
# -> More stable training (when model learns based on one simulation, step t and t+1 are really similar. When more simulation are used during one learning step, data are more diverse, this leads to better problem generalisation).

# Additional env for evauation and callbacks
# eval_env = make_vec_env(env_id, n_envs=1)
models_dir = './models'
os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

eval_callback = EnvEvalCallback(
    eval_freq=EVAL_STEPS,       
    log_dir=models_dir,   
    verbose=1
)

wandb_callback1 = WandbCallback(
    model_save_path=None,      
    verbose=2,
)

wandb_callback2 = wandb_callback_extra()

# Agent initialisation
if pretrained_model_pth is None:
    # --- Init model and policy config---
    encoder_check_point_pth = '/home/developer/ros2_ws/src/autonomous_vehicles/autonomous_vehicles/net_agent/best_weights.ckpt'

    head_arch = dict(
        pi=[512, 64], # Policy head - action
        vf=[128, 64] # Value head - reward
    )

    policy_kwargs = dict(
        features_extractor_class=agent.AgentNet, #(encoder_check_point_pth, action = 2, device = DEVICE), # featuers extractor model 
        features_extractor_kwargs=dict(     
            encoder_check_point_pth=encoder_check_point_pth,  
            device=DEVICE,
            features_dim=1024
        ),
        net_arch=head_arch # Action head
    )
    model = PPO(
        "MultiInputPolicy",  # Define input type, "MultiInputPolicy" for many inputs with different shape
        vec_env, 
        n_steps=1024,          # n_steps training samples kept in buffer
        batch_size=128,        # 
        n_epochs=6,           #
        learning_rate=0.0001,  # 
        gamma=0.995,           # discount factor - how algorythm takes into account future rewards (close to 1 -> future rewards important, close to 0 -> only current reward has importance, greedy)
        policy_kwargs=policy_kwargs, 
        verbose=2,
        device=DEVICE,
        tensorboard_log=f"runs/{run.id}",
    )

else:
    print(f"Loading model from: {pretrained_model_pth}")
    model = PPO.load(
        pretrained_model_pth, 
        env=vec_env, 
        device=DEVICE,
        tensorboard_log=f"runs/{run.id}" 
    )

# Final Architecture visualization
print("--- Architecture: ---")
print(model.policy)
print('-'*10)

# Trening
print('-'*10)
print('Starting training')
model.learn(total_timesteps=TOTAL_STEPS, callback=[wandb_callback1, wandb_callback2, eval_callback])
model.save("RL_Autonomous_Car_finalmodel_1.zip")
run.finish()