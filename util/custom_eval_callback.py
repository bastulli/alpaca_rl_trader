from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=10000, deterministic=True, render=False):
        super(EvalCallback, self).__init__()
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=1, deterministic=self.deterministic, render=self.render)
            # round mean reward for logging purposes
            mean_reward = round(mean_reward, 2)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                formatted_mean_reward = str(mean_reward).replace('.', '_')
                self.model.save(os.path.join(
                    self.best_model_save_path, f"best_model_{formatted_mean_reward}.zip"))

            with open(self.log_path, 'a') as file:
                file.write(
                    f"Step: {self.n_calls}, Mean reward: {mean_reward}\n")
        return True
