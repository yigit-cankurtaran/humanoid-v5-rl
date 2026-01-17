import argparse
import os
import shutil
import subprocess
import sys
import zipfile


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


def install_deps(
    sb3_version: str,
    gymnasium_version: str,
    mujoco_version: str,
    upgrade_latest: bool,
) -> None:
    if upgrade_latest:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-U",
                "stable-baselines3",
                "gymnasium[mujoco]",
                "mujoco",
            ],
            check=True,
        )
        return

    pkgs = [
        f"stable-baselines3=={sb3_version}",
        f"gymnasium[mujoco]=={gymnasium_version}",
    ]
    if mujoco_version:
        pkgs.append(f"mujoco=={mujoco_version}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def assert_env_available(env_id: str) -> None:
    import gymnasium as gym

    if env_id not in gym.envs.registry:
        available = sorted([k for k in gym.envs.registry.keys() if k.startswith("Humanoid-")])
        raise SystemExit(
            f"{env_id} is not available. Installed Humanoid envs: {available}"
        )


def zip_artifacts(repo_root: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("model", "env"):
            base_dir = os.path.join(repo_root, folder)
            if not os.path.isdir(base_dir):
                continue
            for root, _, files in os.walk(base_dir):
                for name in files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, repo_root)
                    zf.write(full_path, rel_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--sb3-version", default="2.3.2")
    parser.add_argument("--gymnasium-version", default="0.29.1")
    parser.add_argument("--mujoco-version", default="3.1.1")
    parser.add_argument("--upgrade-latest", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--drive-dir", default="")
    args = parser.parse_args()

    if not in_colab():
        print("This script is intended to run inside Google Colab.")
        print("Upload this repo to Colab, then run: python colab_train.py")
        return 0

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    if not args.skip_install:
        install_deps(
            args.sb3_version,
            args.gymnasium_version,
            args.mujoco_version,
            args.upgrade_latest,
        )
    assert_env_available(args.env_id)

    cmd = [
        sys.executable,
        "train.py",
        "--total-timesteps",
        str(args.total_timesteps),
        "--n-envs",
        str(args.n_envs),
        "--eval-freq",
        str(args.eval_freq),
        "--n-eval-episodes",
        str(args.n_eval_episodes),
        "--seed",
        str(args.seed),
        "--env-id",
        str(args.env_id),
    ]
    subprocess.run(cmd, check=True)

    zip_path = os.path.join(repo_root, "humanoid_artifacts.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    zip_artifacts(repo_root, zip_path)

    if args.drive_dir:
        os.makedirs(args.drive_dir, exist_ok=True)
        shutil.copy2(zip_path, os.path.join(args.drive_dir, "humanoid_artifacts.zip"))

    from google.colab import files

    files.download(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
