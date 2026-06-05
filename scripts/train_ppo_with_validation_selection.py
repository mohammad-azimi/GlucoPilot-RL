"""Train residual PPO with validation-only checkpoint selection.

This phase trains from scratch on the development pool, evaluates intermediate
checkpoints only on the validation pool, and keeps the frozen held-out suite
closed. Step zero is the neutral residual policy and exactly reproduces the
fixed-action reference controller under deterministic inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import summarize_episode  # noqa: E402
from glucopilot_rl.protocol import (  # noqa: E402
    TRAINING_PATIENTS,
    TRAINING_SCENARIOS,
    TUNED_FIXED_ACTION,
    validation_cases,
)
from glucopilot_rl.rl_env import ResidualGlucoseControlEnv  # noqa: E402
from glucopilot_rl.rl_experiment import run_policy_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train residual PPO and select a checkpoint using validation cases only."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=51_200,
        help="Total intended training steps. Use multiples of 1024 for exact checkpoint timing.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_240,
        help="Save and validate after this many intended steps. Use a multiple of 1024.",
    )
    parser.add_argument(
        "--run-name",
        default="ppo_residual_selection_51k",
        help="Name used for model and output files.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Training and policy seed.")
    return parser.parse_args()


def initialize_neutral_residual_policy(model: PPO) -> None:
    """Make deterministic step-zero inference reproduce the fixed baseline."""
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.log_std.fill_(-2.0)


def build_model(env: Monitor, seed: int) -> PPO:
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
        seed=seed,
        verbose=1,
        device="auto",
    )
    initialize_neutral_residual_policy(model)
    return model


def save_model(model: PPO, model_dir: Path, run_name: str, steps: int) -> Path:
    no_suffix = model_dir / f"{run_name}_step-{steps:06d}"
    model.save(str(no_suffix))
    return Path(f"{no_suffix}.zip")


def evaluate_checkpoint(
    model: PPO,
    *,
    steps: int,
    output_dir: Path,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for patient_name, scenario_name, seed in validation_cases():
        fixed_trace = run_constant_action_episode(
            TUNED_FIXED_ACTION,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )
        ppo_trace = run_policy_episode(
            model,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )
        fixed_metrics = summarize_episode(fixed_trace, "fixed_action")
        ppo_metrics = summarize_episode(ppo_trace, "residual_ppo")
        row: dict[str, float | int | str] = {
            "checkpoint_steps": steps,
            "patient_name": patient_name,
            "scenario_name": scenario_name,
            "seed": seed,
        }
        row.update({f"{key}_fixed": value for key, value in fixed_metrics.items()})
        row.update({f"{key}_ppo": value for key, value in ppo_metrics.items()})
        row["tir_delta_ppo_minus_fixed"] = (
            float(ppo_metrics["time_in_range_pct"]) - float(fixed_metrics["time_in_range_pct"])
        )
        row["risk_delta_ppo_minus_fixed"] = (
            float(ppo_metrics["mean_risk"]) - float(fixed_metrics["mean_risk"])
        )
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / f"validation_step-{steps:06d}.csv", index=False)

    aggregate: dict[str, float | int] = {
        "checkpoint_steps": steps,
        "validation_episodes": len(detail),
        "ppo_mean_tir_pct": float(detail["time_in_range_pct_ppo"].mean()),
        "fixed_mean_tir_pct": float(detail["time_in_range_pct_fixed"].mean()),
        "mean_tir_delta_pct_points": float(detail["tir_delta_ppo_minus_fixed"].mean()),
        "ppo_min_tir_pct": float(detail["time_in_range_pct_ppo"].min()),
        "fixed_min_tir_pct": float(detail["time_in_range_pct_fixed"].min()),
        "ppo_mean_below_pct": float(detail["time_below_range_pct_ppo"].mean()),
        "ppo_max_below_pct": float(detail["time_below_range_pct_ppo"].max()),
        "ppo_mean_very_low_pct": float(detail["time_very_low_pct_ppo"].mean()),
        "ppo_max_very_low_pct": float(detail["time_very_low_pct_ppo"].max()),
        "ppo_mean_risk": float(detail["mean_risk_ppo"].mean()),
        "fixed_mean_risk": float(detail["mean_risk_fixed"].mean()),
        "mean_risk_delta": float(detail["risk_delta_ppo_minus_fixed"].mean()),
    }
    return aggregate, detail


def plot_progress(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(summary["checkpoint_steps"], summary["ppo_mean_tir_pct"], marker="o", label="Residual PPO")
    axis.plot(summary["checkpoint_steps"], summary["fixed_mean_tir_pct"], linestyle="--", label="Fixed reference")
    axis.set_title("Validation Mean Time in Range Across PPO Checkpoints")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean time in range (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_tir_by_checkpoint.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(summary["checkpoint_steps"], summary["ppo_mean_risk"], marker="o", label="Residual PPO")
    axis.plot(summary["checkpoint_steps"], summary["fixed_mean_risk"], linestyle="--", label="Fixed reference")
    axis.set_title("Validation Mean Simulated Risk Across PPO Checkpoints")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean simulated risk")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_risk_by_checkpoint.png", dpi=190)
    plt.close(figure)


def select_checkpoint(summary: pd.DataFrame) -> pd.Series:
    """Apply a conservative validation gate before selecting a learned model.

    Step zero is the neutral residual policy and reproduces the fixed reference.
    A learned checkpoint is allowed to replace it only if it preserves severe-low
    safety, does not increase worst below-range exposure, and either improves
    mean simulated risk or improves mean time in range. This prevents selecting
    a learned policy that merely avoids lows by accepting worse overall control.
    """
    reference = summary.loc[summary["checkpoint_steps"] == 0].iloc[0]
    learned = summary.loc[summary["checkpoint_steps"] > 0].copy()
    accepted = learned.loc[
        (learned["ppo_max_very_low_pct"] <= reference["ppo_max_very_low_pct"])
        & (learned["ppo_max_below_pct"] <= reference["ppo_max_below_pct"])
        & (
            (learned["ppo_mean_risk"] <= reference["fixed_mean_risk"])
            | (learned["mean_tir_delta_pct_points"] >= 0.0)
        )
    ]
    if accepted.empty:
        return reference
    ranked = accepted.sort_values(
        by=[
            "ppo_max_very_low_pct",
            "ppo_max_below_pct",
            "ppo_mean_risk",
            "ppo_mean_tir_pct",
        ],
        ascending=[True, True, True, False],
    )
    return ranked.iloc[0]


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0 or args.checkpoint_every <= 0:
        raise ValueError("total-timesteps and checkpoint-every must be positive.")
    if args.total_timesteps % args.checkpoint_every != 0:
        raise ValueError("total-timesteps must be divisible by checkpoint-every.")
    if args.checkpoint_every % 1024 != 0:
        raise ValueError("checkpoint-every must be a multiple of PPO n_steps=1024.")

    output_dir = ROOT / "outputs" / "model_selection" / args.run_name
    detail_dir = output_dir / "checkpoint_validation"
    model_dir = ROOT / "models" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    check_environment = ResidualGlucoseControlEnv(selection_seed=args.seed)
    check_env(check_environment, warn=True, skip_render_check=True)
    check_environment.close()

    monitor_path = output_dir / "training_monitor"
    env = Monitor(
        ResidualGlucoseControlEnv(selection_seed=args.seed),
        filename=str(monitor_path),
        info_keywords=("patient_name", "scenario_name", "simulator_seed"),
    )
    model = build_model(env, args.seed)

    aggregate_rows: list[dict[str, float | int]] = []

    neutral_model_path = save_model(model, model_dir, args.run_name, 0)
    neutral_result, _ = evaluate_checkpoint(model, steps=0, output_dir=detail_dir)
    aggregate_rows.append(neutral_result)
    print("Validation checkpoint completed: step 0 (neutral policy / fixed reference).")
    print(
        f"  TIR={neutral_result['ppo_mean_tir_pct']:.2f}% | "
        f"mean risk={neutral_result['ppo_mean_risk']:.4f} | model={neutral_model_path.name}"
    )

    completed = 0
    while completed < args.total_timesteps:
        model.learn(total_timesteps=args.checkpoint_every, reset_num_timesteps=False)
        completed = int(model.num_timesteps)
        model_path = save_model(model, model_dir, args.run_name, completed)
        result, _ = evaluate_checkpoint(model, steps=completed, output_dir=detail_dir)
        aggregate_rows.append(result)
        print(f"Validation checkpoint completed: step {completed}.")
        print(
            f"  TIR={result['ppo_mean_tir_pct']:.2f}% | "
            f"delta={result['mean_tir_delta_pct_points']:+.2f} pp | "
            f"mean risk={result['ppo_mean_risk']:.4f} | model={model_path.name}"
        )

    env.close()

    summary = pd.DataFrame(aggregate_rows).sort_values("checkpoint_steps").reset_index(drop=True)
    summary_path = output_dir / "validation_checkpoint_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_progress(summary, output_dir)

    selected = select_checkpoint(summary)
    selected_steps = int(selected["checkpoint_steps"])
    selected_source = model_dir / f"{args.run_name}_step-{selected_steps:06d}.zip"
    selected_target = ROOT / "models" / f"{args.run_name}_selected_best.zip"
    shutil.copy2(selected_source, selected_target)

    selection_report = output_dir / "selected_checkpoint.txt"
    selection_report.write_text(
        "\n".join(
            [
                "Validation-only residual PPO checkpoint selection",
                f"run_name: {args.run_name}",
                f"selected_steps: {selected_steps}",
                f"selected_model: {selected_target}",
                f"mean_tir_pct: {selected['ppo_mean_tir_pct']:.4f}",
                f"mean_tir_delta_pct_points: {selected['mean_tir_delta_pct_points']:.4f}",
                f"max_below_pct: {selected['ppo_max_below_pct']:.4f}",
                f"max_very_low_pct: {selected['ppo_max_very_low_pct']:.4f}",
                f"mean_risk: {selected['ppo_mean_risk']:.4f}",
                "ranking: validation gate first; then minimum max very-low, minimum max below-range, minimum mean risk, maximum mean TIR",
                "final_held_out_suite_used: no",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nResidual PPO checkpoint-selection run completed.")
    print(f"Training patients: {', '.join(TRAINING_PATIENTS)}")
    print(f"Training schedules: {', '.join(TRAINING_SCENARIOS)}")
    print(f"Requested training timesteps: {args.total_timesteps}")
    print(f"Validation checkpoints evaluated: {len(summary)} (including step zero)")
    print(f"Validation summary saved to: {summary_path}")
    print(f"Selected checkpoint: {selected_steps} timesteps")
    print(f"Selected model copied to: {selected_target}")
    print(f"Selected validation mean TIR: {selected['ppo_mean_tir_pct']:.2f}%")
    print(f"Selected validation mean TIR delta: {selected['mean_tir_delta_pct_points']:+.2f} percentage points")
    print(f"Selected validation mean risk: {selected['ppo_mean_risk']:.4f}")
    if selected_steps == 0:
        print("No learned checkpoint passed the validation gate against the neutral reference.")
        print("Do not open the final held-out suite yet; redesign the policy or observations next.")
    else:
        print("A learned checkpoint was selected using validation only.")
        print("Do not run final held-out evaluation until this model is explicitly locked.")
    print(f"Selection report saved to: {selection_report}")


if __name__ == "__main__":
    main()
