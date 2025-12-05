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

import wandb
from wandb.integration.sb3 import WandbCallback

# Parameters
ENV_PARALLEL = 4 # How many envornment shall work parallel during training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_TRAINING_STEPS = 100000 # Total steps
EVAL_STEPS = 5000 # Evaluation after this amount of steps

# --- WandB intergation ---
config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": TOTAL_TRAINING_STEPS,
    "env_id": "GazeboCarEnv", # Sprawdzić czy poprawne uzupełnione
    "net_arch": [128, 64] # Podmieć!!!
}

run = wandb.init(
    project=".........",
    config=config,
    sync_tensorboard=True, # Bardzo przydatne
    monitor_gym=True
)

wandb_callback = WandbCallback(
    model_save_path=f"models/{run.id}", #Save models to wandb
    verbose=2,
)

# --- check compliance with StableBaseline3 ---
env = GazeboCarEnv()
check_env(env, warn=True)
#TODO: delte fragment or destructor to not defince environment two times

# --- Init model ---
#TODO
# zmienić model żeby był tylko ekstraktorem cech (pozbyć się głowy) 
# Zamrozić pierwsze warstwy Uneta już na etapie inicjalizacji ( w konstruktorze klasy)

head_arch = dict(
    pi=[128, 64], # Policy head - action
    vf=[256] # Value head - reward
)

policy_kwargs = dict(
    features_extractor_class=MultimodalModel, # featuers extractor model 
    features_extractor_kwargs=dict(features_dim=1024), # Output features vector dimension
    net_arch=head_arch # Action head
)

# Environment vectorization - for parallel training
env_id = lambda: GazeboCarEnv()
vec_env = make_vec_env(env_id, n_envs=ENV_PARALLEL)
# -> Faster trainging (GPU waits until simulation ennds on CPU, its better to use all of available CPU threads 
# -> More stable training (when model learns based on one simulation, step t and t+1 are really similar. When more simulation are used during one learning step, data are more diverse, this leads to better problem generalisation).

# Additional env for evauation and callbacks
eval_env = make_vec_env(lambda: TwojeSrodowisko(), n_envs=1)
os.makedirs(os.path.dirname('./models'), exist_ok=True)
os.makedirs(os.path.dirname('./logs'), exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models', # Katalog, gdzie trafi najlepszy model
    log_path='./logs',
    eval_freq=EVAL_STEPS, # Evaluation with each
    deterministic=True, # True to block exploration, only actions with highest prop    
    render=True, # To visualize best trajectory (methods implemented in enronment: render() and close().
    callback_on_new_best=None # Call back when new best model, if None, default option: save best model            
)

# Agent initialisation
model = PPO(
    "MultiInputPolicy",  # Define input type, "MultiInputPolicy" for many inputs with different shape
    vec_env, 
    policy_kwargs=policy_kwargs, 
    verbose=2,
    device=DEVICE 
)

# Final Architecture visualization
print("--- Print architecture ---")
print(model.policy)
print("----------------------------------")

# Trening
model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=[wandb_callback])
model.save("RL_Autonomous_Car_finalmodel_1.zip")
