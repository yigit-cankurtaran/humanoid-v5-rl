from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def watch(
    model_path="./model/optuna_best/best_model.zip",
    env_path="./env/tune_vecnormalize.pkl",
    n_eps=5,
):
    watch_env = DummyVecEnv(
        [lambda: Monitor(gym.make("Humanoid-v5", render_mode="human"))]
    )
    watch_env = VecNormalize.load(env_path, watch_env)
    watch_env.norm_reward = False
    watch_env.training = False

    model = PPO.load(model_path)

    rewards, lengths = evaluate_policy(
        model, watch_env, n_eps, render=True, return_episode_rewards=True
    )

    for i in range(len(rewards)):
        print(f"epoch:{i + 1}, reward:{rewards[i]}, length:{lengths[i]}")

    watch_env.close()


if __name__ == "__main__":
    watch()
