from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import os

class wandb_callback_extra(BaseCallback):
    def _on_step(self):
        info = self.locals["infos"][0]
        if "reset_info" in info:
            ri = info["reset_info"]
            for key, value in ri.items():
                self.logger.record(f"rewards_components/{key}", value)
        return True
    

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

            # if mean_reward > self.best_mean_reward:
            #     self.best_mean_reward = mean_reward
            #     name = f"best_model_e{self.eval_cntr}"
            #     self.model.save(os.path.join(self.log_dir, name))

            #     print(f'[Eval] New best model save: {os.path.join(self.log_dir, name)}')
 
            # self.training_env.reset()


            reward_str = f"{mean_reward:+.2f}".replace('.', '_').replace('+', 'p').replace('-', 'm')
            
            name = f"model_e{self.eval_cntr}_r{reward_str}" 
            save_path = os.path.join(self.log_dir, name)
            
            self.model.save(save_path)
            print(f'[Eval] Model saved: {save_path}')
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f'[Eval] **New BEST model found!** Current Best Mean Reward: {self.best_mean_reward:.2f}')

            self.training_env.reset()
        return True