from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import gymnasium as gym
import os


def decay(initial, final):
    def step(progress_remaining):
        return max(final, initial * progress_remaining)

    return step


def sync_normalization(train_env: VecNormalize, eval_env: VecNormalize) -> None:
    # making sure eval matches training
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.clip_obs = train_env.clip_obs
    eval_env.clip_reward = train_env.clip_reward


def train(n_eps=100_000):
    envname = "Humanoid-v5"
    # making and assigning folders
    dirs = ["model", "log", "env"]
    for name in dirs:
        os.makedirs(name, exist_ok=True)
    model_path, log_path, _ = dirs

    train_env = VecNormalize(make_vec_env(envname, 4))
    # VecNormalize defaults to training=True and norm_reward=True, fixing that
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(gym.make(envname))]),
        training=False,
        norm_reward=False,
    )
    sync_normalization(train_env, eval_env)

    eval_callback = EvalCallback(
        eval_env, log_path=log_path, best_model_save_path=model_path
    )

    # the rest will be defaults for now
    # previous training run got us very limited gait
    # adding more ent_coef, implementing learning rate decay, training for longer
    model = PPO(
        "MlpPolicy",
        env=train_env,
        gamma=0.999,
        ent_coef=1e-2,
        batch_size=512,
        n_steps=4096,
        learning_rate=decay(3e-4, 3e-5),
    )
    model.learn(n_eps, eval_callback, progress_bar=True)

    # save train env to use its VecNormalize data later
    train_env.save("env/train_env.pkl")


if __name__ == "__main__":
    train()
