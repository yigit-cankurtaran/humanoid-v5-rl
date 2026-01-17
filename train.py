from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import gymnasium as gym
import argparse
import os
from typing import Optional


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


class VecNormalizeEvalCallback(EvalCallback):
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, **kwargs):
        super().__init__(eval_env=eval_env, **kwargs)
        self._train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_normalization(self._train_env, self.eval_env)
        return super()._on_step()


def train(
    env_id: str = "Humanoid-v4",
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 10,
    seed: Optional[int] = 1,
):
    # making and assigning folders
    dirs = ["model", "log", "env"]
    for name in dirs:
        os.makedirs(name, exist_ok=True)
    model_path, log_path, _ = dirs

    train_env = VecNormalize(
        make_vec_env(env_id, n_envs, seed=seed),
        gamma=0.99,
        clip_obs=10.0,
    )
    # VecNormalize defaults to training=True and norm_reward=True, fixing that
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(gym.make(env_id))]),
        training=False,
        norm_reward=False,
    )
    sync_normalization(train_env, eval_env)

    eval_callback = VecNormalizeEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        log_path=log_path,
        best_model_save_path=os.path.join(model_path, "best"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=os.path.join(model_path, "checkpoints"),
        name_prefix="ppo_humanoid",
    )

    # the rest will be defaults for now
    # previous training run got us very limited gait
    # adding more ent_coef, implementing learning rate decay, training for longer
    model = PPO(
        "MlpPolicy",
        env=train_env,
        gamma=0.99,
        ent_coef=1e-3,
        batch_size=1024,
        n_steps=2048,
        n_epochs=10,
        learning_rate=decay(3e-4, 1e-4),
        clip_range=0.2,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
        use_sde=True,
        sde_sample_freq=4,
        device="auto",
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # save train env to use its VecNormalize data later
    train_env.save("env/train_env.pkl")
    model.save(os.path.join(model_path, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Humanoid-v4")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    train(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
    )
