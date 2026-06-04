"""Train a normalized residual PPO policy on development-only episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.protocol import TRAINING_PATIENTS, TRAINING_SCENARIOS, TUNED_FIXED_ACTION  # noqa: E402
from glucopilot_rl.rl_env import (  # noqa: E402
    MAX_SIMULATOR_ACTION,
    MIN_SIMULATOR_ACTION,
    ResidualGlucoseControlEnv,
    residual_to_simulator_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual PPO for simulated glucose control.")
    parser.add_argument(
        "--timesteps", type=int, default=5_000,
        help="Training timesteps. Start with 5000 as a corrected smoke test.",
    )
    parser.add_argument(
        "--model-name", default="ppo_residual_smoke_model",
        help="Output filename without .zip in the models directory.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="PPO and training-pool seed.")
    return parser.parse_args()


def initialize_neutral_residual_policy(model: PPO) -> None:
    """Make initial deterministic control exactly reproduce the fixed baseline."""
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        # Smaller exploration than the native-action prototype: perturbations
        # begin around the reference controller rather than jumping to bounds.
        model.policy.log_std.fill_(-2.0)


def plot_monitor_curve(monitor_csv: Path, output_path: Path) -> int:
    if not monitor_csv.exists():
        return 0
    monitor = pd.read_csv(monitor_csv, comment="#")
    if monitor.empty:
        return 0
    monitor["episode"] = range(1, len(monitor) + 1)
    monitor["rolling_reward"] = monitor["r"].rolling(window=min(10, len(monitor)), min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(monitor["episode"], monitor["r"], marker="o", linewidth=1, label="Episode reward")
    axis.plot(monitor["episode"], monitor["rolling_reward"], linewidth=2, label="Rolling mean")
    axis.set_title("Residual PPO Training Monitor")
    axis.set_xlabel("Completed episode")
    axis.set_ylabel("Episode reward")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)
    return len(monitor)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs" / "training"
    model_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    check_environment = ResidualGlucoseControlEnv(selection_seed=args.seed)
    check_env(check_environment, warn=True, skip_render_check=True)
    check_environment.close()

    monitor_path = output_dir / f"{args.model_name}_monitor"
    env = Monitor(
        ResidualGlucoseControlEnv(selection_seed=args.seed),
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
        ent_coef=0.0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=args.seed,
        verbose=1,
        device="auto",
    )
    initialize_neutral_residual_policy(model)
    model.learn(total_timesteps=args.timesteps)
    model_path = model_dir / args.model_name
    model.save(str(model_path))
    env.close()

    monitor_csv = Path(f"{monitor_path}.monitor.csv")
    training_chart = output_dir / f"{args.model_name}_training_rewards.png"
    completed_episodes = plot_monitor_curve(monitor_csv, training_chart)

    print("Residual PPO training run completed.")
    print(f"Purpose: {'corrected smoke test only' if args.timesteps <= 10000 else 'experimental training run'}")
    print(f"Timesteps requested: {args.timesteps}")
    print(f"Training patients: {', '.join(TRAINING_PATIENTS)}")
    print(f"Training meal schedules: {', '.join(TRAINING_SCENARIOS)}")
    print("Policy action space: normalized residual adjustment [-1.0, 1.0]")
    print(
        "Residual mapping: "
        f"-1.0 -> {residual_to_simulator_action(-1.0):.4f}, "
        f"0.0 -> {residual_to_simulator_action(0.0):.4f}, "
        f"+1.0 -> {residual_to_simulator_action(1.0):.4f}"
    )
    print(f"Frozen reference simulator action reproduced at residual 0.0: {TUNED_FIXED_ACTION:.4f}")
    print(f"Simulator action bounds: {MIN_SIMULATOR_ACTION:.4f} to {MAX_SIMULATOR_ACTION:.4f}")
    print(f"Completed episodes recorded: {completed_episodes}")
    print(f"Model saved to: {model_path}.zip")
    print(f"Training monitor saved to: {monitor_csv}")
    print(f"Training reward chart saved to: {training_chart}")
    if args.timesteps <= 10000:
        print("This short run validates the corrected residual-policy pipeline.")
        print("Evaluate it on the validation suite, not the final held-out suite.")


if __name__ == "__main__":
    main()
