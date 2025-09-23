from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import gymnasium as gym
import os


def train(n_eps=5_000_000):
    envname = "Humanoid-v5"
    # making and assigning folders
    dirs = ["model", "log", "env"]
    for name in dirs:
        os.makedirs(name, exist_ok=True)
    model_path, log_path, _ = dirs

    train_env = VecNormalize(make_vec_env(envname, 4))
    eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(gym.make(envname))]))

    eval_callback = EvalCallback(
        eval_env, log_path=log_path, best_model_save_path=model_path
    )

    # the rest will be defaults for now
    # TODO: start tuning some hyperparams
    model = PPO(
        "MlpPolicy",
        env=train_env,
        gamma=0.999,
        ent_coef=5e-3,
        batch_size=512,
        n_steps=4096,
    )
    model.learn(n_eps, eval_callback, progress_bar=True)

    # save train env to use its VecNormalize data later
    train_env.save("env/train_env.pkl")


if __name__ == "__main__":
    train()
