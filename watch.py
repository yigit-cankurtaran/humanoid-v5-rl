from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse


def watch(
    model_path="./model/best/best_model.zip",
    env_path="./env/train_env.pkl",
    env_id="Humanoid-v4",
    n_eps=5,
):
    watch_env = DummyVecEnv(
        [lambda: Monitor(gym.make(env_id, render_mode="human"))]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./model/best/best_model.zip")
    parser.add_argument("--env-path", default="./env/train_env.pkl")
    parser.add_argument("--env-id", default="Humanoid-v4")
    parser.add_argument("--n-eps", type=int, default=5)
    args = parser.parse_args()
    watch(
        model_path=args.model_path,
        env_path=args.env_path,
        env_id=args.env_id,
        n_eps=args.n_eps,
    )
