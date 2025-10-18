"""Minimal Optuna example for PPO hyperparameter tuning.

This script keeps everything in one file and focuses on the core ideas:
  * define a search space inside an objective
  * train PPO briefly per trial
  * report the mean evaluation reward back to Optuna

Usage:
    python tune_basic.py --trials 5 --timesteps 100000
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import optuna
from optuna import Trial
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(env_id: str) -> VecNormalize:
    """Create a single VecNormalize-wrapped environment for training/eval."""
    env = DummyVecEnv([lambda: Monitor(gym.make(env_id))])
    return VecNormalize(env)


def objective(trial: Trial, env_id: str, timesteps: int) -> float:
    """Train PPO with sampled hyperparameters and return evaluation reward."""
    # Sample a handful of influential PPO knobs. Keeping the list small makes the
    # example easy to follow and keeps tuning time short.
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)

    train_env = make_env(env_id)
    eval_env = make_env(env_id)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO(
        "MlpPolicy",
        train_env,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps)

    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, return_episode_rewards=False)

    train_env.close()
    eval_env.close()

    return mean_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic Optuna example for PPO")
    parser.add_argument("--trials", type=int, default=5, help="How many Optuna trials to run")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps per trial")
    parser.add_argument("--env-id", type=str, default="Humanoid-v5", help="Gymnasium environment id")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.env_id, args.timesteps), n_trials=args.trials)

    print("Best trial:", study.best_trial.number)
    print("Best reward:", study.best_trial.value)
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
