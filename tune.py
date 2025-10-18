"""Optuna-powered PPO hyperparameter search entry point.

Typical usage:
    python tune.py --trials 20 --timesteps 200000

Add `--retrain` if you want to immediately train a fresh agent with the best
sampled parameters once the study finishes. The retrain step saves the latest
VecNormalize stats so `watch.py` can replay the policy.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import gymnasium as gym
import optuna
from optuna import Trial
from optuna.exceptions import TrialPruned
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from train import decay, sync_normalization

# Default configuration targets the Humanoid benchmark you already train in
# train.py. Override via CLI flags if you want to explore a different task.
DEFAULT_ENV_ID = "Humanoid-v5"
DEFAULT_NUM_ENVS = 4
DEFAULT_EVAL_EPISODES = 5
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_STUDY_NAME = "humanoid-ppo"
DEFAULT_MODEL_ROOT = os.path.join("model", "tune")
DEFAULT_LOG_ROOT = os.path.join("log", "tune")
DEFAULT_ENV_SAVE = os.path.join("env", "tune_vecnormalize.pkl")


@dataclass
class ObjectiveConfig:
    env_id: str
    num_envs: int
    total_timesteps: int
    eval_episodes: int
    model_root: str
    log_root: str
    seed: Optional[int]


def build_train_env(env_id: str, num_envs: int, seed: Optional[int]) -> VecNormalize:
    """Create the vectorized training environment with normalization enabled."""
    vec_env = make_vec_env(env_id, n_envs=num_envs, seed=seed)
    return VecNormalize(vec_env)


def build_eval_env(env_id: str, seed: Optional[int]) -> VecNormalize:
    """Create an evaluation environment that mirrors the training statistics."""

    def _make_env() -> Monitor:
        env = gym.make(env_id)
        env.reset(seed=seed)
        return Monitor(env)

    eval_vec = DummyVecEnv([_make_env])
    # We disable reward normalization for evaluation so scores stay interpretable.
    return VecNormalize(eval_vec, training=False, norm_reward=False)


def sample_hyperparameters(trial: Trial, num_envs: int) -> Dict[str, float]:
    """Describe the PPO search space.

    We deliberately keep the ranges tight around values that behaved well in
    earlier experiments to speed up convergence. The batch size is constrained
    to divide the rollout size (n_steps * num_envs) so PPO's batching logic
    never raises shape errors.
    """
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 3072, 4096, 6144])
    rollout_size = n_steps * num_envs

    valid_batches = [b for b in [128, 256, 512, 1024, 2048] if rollout_size % b == 0]
    if not valid_batches:
        raise TrialPruned("No valid batch size for sampled n_steps.")

    batch_size = trial.suggest_categorical("batch_size", valid_batches)
    gamma = trial.suggest_float("gamma", 0.97, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.8, 1.2)
    lr_initial = trial.suggest_float("lr_initial", 5e-5, 5e-4, log=True)
    lr_decay_ratio = trial.suggest_float("lr_decay_ratio", 0.05, 0.5)

    params = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "lr_initial": lr_initial,
        "lr_decay_ratio": lr_decay_ratio,
        "lr_final": lr_initial * lr_decay_ratio,
    }

    trial.set_user_attr("rollout_size", rollout_size)
    return params


def build_model(
    train_env: VecNormalize, params: Dict[str, float], seed: Optional[int]
) -> PPO:
    """Instantiate PPO with the sampled hyperparameters."""
    lr_initial = params["lr_initial"]
    if "lr_final" in params:
        lr_final = params["lr_final"]
    else:
        # When using study.best_params() only sampled values are present, so
        # reconstruct the final LR from the decay ratio.
        lr_decay_ratio = params.get("lr_decay_ratio", 0.1)
        lr_final = lr_initial * lr_decay_ratio
    lr_schedule = decay(lr_initial, lr_final)
    return PPO(
        "MlpPolicy",
        env=train_env,
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        learning_rate=lr_schedule,
        seed=seed,
        verbose=0,
    )


def objective(trial: Trial, config: ObjectiveConfig) -> float:
    """Optuna objective that trains PPO and returns the evaluation reward."""
    params = sample_hyperparameters(trial, config.num_envs)

    train_env = build_train_env(config.env_id, config.num_envs, config.seed)
    eval_env = build_eval_env(config.env_id, config.seed)
    sync_normalization(train_env, eval_env)

    trial_model_dir = os.path.join(config.model_root, f"trial_{trial.number}")
    trial_log_dir = os.path.join(config.log_root, f"trial_{trial.number}")
    os.makedirs(trial_model_dir, exist_ok=True)
    os.makedirs(trial_log_dir, exist_ok=True)

    model = build_model(train_env, params, config.seed)

    eval_freq = max(params["n_steps"], config.total_timesteps // 10)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=trial_model_dir,
        log_path=trial_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        n_eval_episodes=config.eval_episodes,
        verbose=0,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )

    # After learning the training statistics changed, so keep the eval env synced
    # before computing the final score we will hand back to Optuna.
    sync_normalization(train_env, eval_env)

    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        return_episode_rewards=False,
    )

    trial.report(mean_reward, step=config.total_timesteps)
    if trial.should_prune():
        raise TrialPruned()

    # Stash VecNormalize stats next to the best model in case this trial wins.
    train_env.save(os.path.join(trial_model_dir, "vecnormalize.pkl"))

    train_env.close()
    eval_env.close()

    return mean_reward


def run_study(args: argparse.Namespace) -> optuna.Study:
    if args.storage:
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", study_name=args.study_name)

    config = ObjectiveConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        model_root=args.model_root,
        log_root=args.log_root,
        seed=args.seed,
    )

    os.makedirs(args.model_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)

    study.optimize(
        partial(objective, config=config), n_trials=args.trials, gc_after_trial=True
    )
    return study


def save_best_params(study: optuna.Study, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(study.best_params, handle, indent=2)
    print(f"Stored best parameters in {path}")


def retrain_with_best(
    best_params: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    """Train a fresh agent with the best sampled parameters."""
    print("Retraining PPO with the best Optuna suggestions...")

    train_env = build_train_env(args.env_id, args.num_envs, args.seed)
    eval_env = build_eval_env(args.env_id, args.seed)
    sync_normalization(train_env, eval_env)

    retrain_params = dict(best_params)
    retrain_params.setdefault(
        "lr_final", retrain_params["lr_initial"] * retrain_params.get("lr_decay_ratio", 0.1)
    )
    model = build_model(train_env, retrain_params, args.seed)

    os.makedirs(args.final_model_dir, exist_ok=True)
    os.makedirs(args.final_log_dir, exist_ok=True)

    eval_freq = max(best_params["n_steps"], args.retrain_timesteps // 10)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.final_model_dir,
        log_path=args.final_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        n_eval_episodes=args.eval_episodes,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.retrain_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    sync_normalization(train_env, eval_env)
    train_env.save(args.final_env_path)

    train_env.close()
    eval_env.close()

    print("Retrain finished. Best checkpoint lives at:")
    print(
        f"  model: {os.path.join(args.final_model_dir, 'best_model.zip')}\n  vec normalize: {args.final_env_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for humanoid PPO with Optuna."
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Training timesteps per trial",
    )
    parser.add_argument(
        "--env-id", type=str, default=DEFAULT_ENV_ID, help="Gymnasium environment id"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help="Vectorized environment count",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Episodes per evaluation pass",
    )
    parser.add_argument(
        "--study-name", type=str, default=DEFAULT_STUDY_NAME, help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Directory to store per-trial models",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=DEFAULT_LOG_ROOT,
        help="Directory to store evaluation logs",
    )
    parser.add_argument(
        "--save-params",
        type=str,
        default=None,
        help="Optional path to dump best parameters as JSON",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain a policy with the best parameters once tuning completes",
    )
    parser.add_argument(
        "--retrain-timesteps",
        type=int,
        default=500_000,
        help="Timesteps for the optional retrain run",
    )
    parser.add_argument(
        "--final-model-dir",
        type=str,
        default=os.path.join("model", "optuna_best"),
        help="Destination directory for the retrained best model",
    )
    parser.add_argument(
        "--final-log-dir",
        type=str,
        default=os.path.join("log", "optuna_best"),
        help="Destination eval log dir for the retrained best model",
    )
    parser.add_argument(
        "--final-env-path",
        type=str,
        default=DEFAULT_ENV_SAVE,
        help="Where to store VecNormalize stats for the retrained best model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = run_study(args)

    print("Best trial reward:", study.best_value)
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    if args.save_params:
        save_best_params(study, args.save_params)

    if args.retrain:
        retrain_with_best(study.best_params, args)
        print(
            "Retrain artifacts saved. Update watch.py paths if you want to visualise the tuned policy."
        )


if __name__ == "__main__":
    main()
