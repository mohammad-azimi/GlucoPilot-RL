"""Train the first PPO glucose-control policy on development-only episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.protocol import TRAINING_PATIENTS, TRAINING_SCENARIOS  # noqa: E402
from glucopilot_rl.rl_env import AdaptiveGlucoseControlEnv, MAX_SIMULATOR_ACTION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO model for simulated glucose control.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000,
        help="Training timesteps. Use 5000 first as a plumbing/smoke test.",
    )
    parser.add_argument(
        "--model-name",
        default="ppo_smoke_model",
        help="Output filename without .zip in the models directory.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="PPO and training-pool seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs" / "training"
    model_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    check_environment = AdaptiveGlucoseControlEnv(selection_seed=args.seed)
    check_env(check_environment, warn=True, skip_render_check=True)
    check_environment.close()

    monitor_path = output_dir / f"{args.model_name}_monitor"
    env = Monitor(
        AdaptiveGlucoseControlEnv(selection_seed=args.seed),
        filename=str(monitor_path),
        info_keywords=("patient_name", "scenario_name", "simulator_seed"),
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        verbose=1,
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps)
    model_path = model_dir / args.model_name
    model.save(str(model_path))
    env.close()

    monitor_csv = Path(f"{monitor_path}.monitor.csv")
    completed_episodes = 0
    if monitor_csv.exists():
        monitor_data = pd.read_csv(monitor_csv, comment="#")
        completed_episodes = len(monitor_data)

    print("PPO training run completed.")
    print(f"Purpose: {'smoke test only' if args.timesteps <= 10000 else 'experimental training run'}")
    print(f"Timesteps requested: {args.timesteps}")
    print(f"Training patients: {', '.join(TRAINING_PATIENTS)}")
    print(f"Training meal schedules: {', '.join(TRAINING_SCENARIOS)}")
    print(f"Maximum simulator action allowed during training: {MAX_SIMULATOR_ACTION:.4f}")
    print(f"Completed episodes recorded: {completed_episodes}")
    print(f"Model saved to: {model_path}.zip")
    print(f"Training monitor saved to: {monitor_csv}")
    if args.timesteps <= 10000:
        print("This short run verifies the PPO pipeline; it is not a final performance result.")
        print("Next run the held-out evaluator to confirm inference works end to end.")


if __name__ == "__main__":
    main()
