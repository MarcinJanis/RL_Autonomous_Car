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

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import net_agent.net_v1 as agent
from gazebo_car_env import GazeboCarEnv as gazebo_env

import wandb
from wandb.integration.sb3 import WandbCallback

# Parameters
ENV_PARALLEL = 1 # How many envornment shall work parallel during training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 1000000 # Total steps
EVAL_STEPS = 10000 # Evaluation after this amount of steps
MAX_STEPS_PER_EPISODE = 1800 # Steps per episoed (max)
TIME_STEP = 0.1 # [s]

rewards =  { 'velocity': 0.1, 'trajectory': -0.001, 'prog': 1, 'collision': -100, 'timeout': -100, 'destin': 100 }
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
    "rewards" : rewards
}

wandb.login()

run = wandb.init(
    project="RL_Autonomous_Car",
    entity="deep-neural-network-course",
    name='RL_TestRun_9', # Name
    settings=wandb.Settings(save_code=False),
    config=config,
    sync_tensorboard=True,
    monitor_gym=False,
    save_code=False,
    mode='online'
)

wandb_callback = WandbCallback(
    model_save_path=None,   
    # save_model=False,      
    verbose=2,
    # gradient_save_freq=0
)

class wandb_callback_extra(BaseCallback):
    def _on_step(self):
        info = self.locals["infos"][0]
        if "reset_info" in info:
            ri = info["reset_info"]
            for key, value in ri.items():
                self.logger.record(f"rewards_components/{key}", value)
        return True

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

# Environment vectorization - for parallel training
env_id = lambda: gazebo_env(time_step = TIME_STEP,
                            rewards = rewards, 
                            trajectory_points_pth = trajectory_goal, 
                            max_steps_per_episode = MAX_STEPS_PER_EPISODE, 
                            max_lin_vel = max_linear_velocity,
                            max_ang_vel = max_angular_velocity,
                            render_mode=True)

# Evaluation callback

class EnvEvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, log_dir: str, verbose = 1):
        super(EnvEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.best_mean_reward = -float('inf')
        self.eval_cntr = 0 

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if  self.verbose > 0:
                print('[Eval] Evaluation starting ...')
            self.eval_cntr += 1
            total_rewards = []
            n_episodes = 3

            for i in range(n_episodes):
                done = False
                episode_reward = 0.0
                obs = self.training_env.reset() # reset env

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = self.training_env.step(action)

                    reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards # pierwsze czy ostatnie
                    done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

                    episode_reward += reward

                total_rewards.append(episode_reward)
                try:
                    single_env = self.training_env.envs[0].unwrapped
                    single_env.render()
                    print('[Eval] Render suceed.')
                except Exception as e:
                    print(f'[Error] Cant render: {e} ')

            mean_reward = np.mean(total_rewards)
            if self.verbose > 0:
                print(f'[Eval] Statstic: ')
                print(f'Mean reward: {mean_reward} +/- {np.std(total_rewards)}')
                for k in range(n_episodes):
                    print(f'Episode {k}: rewards: {total_rewards[k]}')

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                name = f"best_model_e{self.eval_cntr}"
                self.model.save(os.path.join(self.log_dir, name))

                print(f'[Eval] New best model save: {os.path.join(self.log_dir, name)}')
 
            self.training_env.reset()
        return True
    


    # mean_reward, std_reward = evaluate_policy( self.model,
    #                                            self.training_env,
    #                                            n_eval_episodes=1,    # Liczba przejazdów testowych (możesz zmienić)
    #                                            deterministic=True,   # Testujemy bez losowego szumu (czysta wiedza sieci)
    #                                            render=False          # Wyłączamy renderowanie, żeby nie blokować wątków
    #                                             )



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





# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path='./models', # Dir with best models 
#     log_path='./logs',
#     eval_freq=EVAL_STEPS, # Evaluation with each
#     deterministic=True, # True to block exploration, only actions with highest prop    
#     render=True, # To visualize best trajectory (methods implemented in enronment: render() and close().
#     callback_on_new_best=None, # Call back when new best model, if None, default option: save best model      
#     verbose=2      
# )





# Agent initialisation
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




# Final Architecture visualization
print("--- Architecture: ---")
print(model.policy)
print('-'*10)

# Trening
print('-'*10)
print('Starting training')
model.learn(total_timesteps=TOTAL_STEPS, callback=[wandb_callback, eval_callback])
model.save("RL_Autonomous_Car_finalmodel_1.zip")
run.finish()