"""Train shielded discrete DQN with validation-only checkpoint selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.discrete_env import DiscreteShieldedGlucoseControlEnv  # noqa: E402
from glucopilot_rl.discrete_experiment import ConstantDiscretePolicy, run_discrete_policy_episode  # noqa: E402
from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import summarize_episode  # noqa: E402
from glucopilot_rl.protocol import TRAINING_PATIENTS, TRAINING_SCENARIOS, TUNED_FIXED_ACTION, validation_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shielded discrete DQN and select with validation only.")
    parser.add_argument("--total-timesteps", type=int, default=51200)
    parser.add_argument("--checkpoint-every", type=int, default=10240)
    parser.add_argument("--run-name", default="dqn_discrete_shield_selection_51k")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def build_model(env: Monitor, seed: int) -> DQN:
    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=2048,
        batch_size=128,
        tau=1.0,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1024,
        exploration_fraction=0.35,
        exploration_initial_eps=0.20,
        exploration_final_eps=0.02,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        verbose=1,
        device="auto",
    )


def evaluate_candidate(model_or_policy, *, steps: int, output_dir: Path) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for patient_name, scenario_name, seed in validation_cases():
        fixed_trace = run_constant_action_episode(
            TUNED_FIXED_ACTION,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )
        policy_trace = run_discrete_policy_episode(
            model_or_policy,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )
        fixed_metrics = summarize_episode(fixed_trace, "fixed_action")
        policy_metrics = summarize_episode(policy_trace, "shielded_discrete_policy")
        row: dict[str, float | int | str] = {
            "checkpoint_steps": steps,
            "patient_name": patient_name,
            "scenario_name": scenario_name,
            "seed": seed,
        }
        row.update({f"{key}_fixed": value for key, value in fixed_metrics.items()})
        row.update({f"{key}_policy": value for key, value in policy_metrics.items()})
        row["tir_delta_policy_minus_fixed"] = (
            float(policy_metrics["time_in_range_pct"]) - float(fixed_metrics["time_in_range_pct"])
        )
        row["risk_delta_policy_minus_fixed"] = (
            float(policy_metrics["mean_risk"]) - float(fixed_metrics["mean_risk"])
        )
        row["shield_intervention_pct"] = float(
            100.0 * (policy_trace["shield_reason"] != "unchanged").mean()
        )
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / f"validation_step-{steps:06d}.csv", index=False)
    aggregate: dict[str, float | int] = {
        "checkpoint_steps": steps,
        "validation_episodes": len(detail),
        "policy_mean_tir_pct": float(detail["time_in_range_pct_policy"].mean()),
        "fixed_mean_tir_pct": float(detail["time_in_range_pct_fixed"].mean()),
        "mean_tir_delta_pct_points": float(detail["tir_delta_policy_minus_fixed"].mean()),
        "policy_min_tir_pct": float(detail["time_in_range_pct_policy"].min()),
        "policy_max_below_pct": float(detail["time_below_range_pct_policy"].max()),
        "policy_max_very_low_pct": float(detail["time_very_low_pct_policy"].max()),
        "policy_mean_risk": float(detail["mean_risk_policy"].mean()),
        "fixed_mean_risk": float(detail["mean_risk_fixed"].mean()),
        "mean_risk_delta": float(detail["risk_delta_policy_minus_fixed"].mean()),
        "mean_shield_intervention_pct": float(detail["shield_intervention_pct"].mean()),
    }
    return aggregate, detail


def plot_progress(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(summary["checkpoint_steps"], summary["policy_mean_tir_pct"], marker="o", label="Shielded discrete policy")
    axis.plot(summary["checkpoint_steps"], summary["fixed_mean_tir_pct"], linestyle="--", label="Fixed reference")
    axis.set_title("Validation Mean Time in Range Across DQN Checkpoints")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean time in range (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_tir_by_checkpoint.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(summary["checkpoint_steps"], summary["policy_mean_risk"], marker="o", label="Shielded discrete policy")
    axis.plot(summary["checkpoint_steps"], summary["fixed_mean_risk"], linestyle="--", label="Fixed reference")
    axis.set_title("Validation Mean Simulated Risk Across DQN Checkpoints")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean simulated risk")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_risk_by_checkpoint.png", dpi=190)
    plt.close(figure)


def select_checkpoint(summary: pd.DataFrame) -> pd.Series:
    """Select only if a candidate passes the validation gate.

    Gate: no severe lows, no more below-range exposure than the fixed reference,
    and either better mean risk or better mean TIR. Otherwise step zero is kept.
    """
    step_zero = summary.loc[summary["checkpoint_steps"] == 0].iloc[0]
    candidates = summary.copy()
    passes_gate = (
        (candidates["policy_max_very_low_pct"] <= step_zero["policy_max_very_low_pct"] + 1e-9)
        & (candidates["policy_max_below_pct"] <= step_zero["policy_max_below_pct"] + 1e-9)
        & (
            (candidates["policy_mean_risk"] < step_zero["policy_mean_risk"] - 1e-6)
            | (candidates["policy_mean_tir_pct"] > step_zero["policy_mean_tir_pct"] + 1e-6)
        )
    )
    valid = candidates.loc[passes_gate]
    if valid.empty:
        return step_zero
    return valid.sort_values(
        by=["policy_max_very_low_pct", "policy_max_below_pct", "policy_mean_risk", "policy_mean_tir_pct"],
        ascending=[True, True, True, False],
    ).iloc[0]


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0 or args.checkpoint_every <= 0:
        raise ValueError("total-timesteps and checkpoint-every must be positive.")
    if args.total_timesteps % args.checkpoint_every != 0:
        raise ValueError("total-timesteps must be divisible by checkpoint-every.")

    output_dir = ROOT / "outputs" / "model_selection" / args.run_name
    detail_dir = output_dir / "checkpoint_validation"
    model_dir = ROOT / "models" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    check_environment = DiscreteShieldedGlucoseControlEnv(selection_seed=args.seed)
    check_env(check_environment, warn=True, skip_render_check=True)
    check_environment.close()

    env = Monitor(
        DiscreteShieldedGlucoseControlEnv(selection_seed=args.seed),
        filename=str(output_dir / "training_monitor"),
        info_keywords=("patient_name", "scenario_name", "simulator_seed"),
    )
    model = build_model(env, args.seed)

    aggregate_rows: list[dict[str, float | int]] = []
    neutral_policy = ConstantDiscretePolicy()
    neutral_result, _ = evaluate_candidate(neutral_policy, steps=0, output_dir=detail_dir)
    aggregate_rows.append(neutral_result)
    print("Validation checkpoint completed: step 0 (neutral discrete shield).")
    print(
        f"  TIR={neutral_result['policy_mean_tir_pct']:.2f}% | "
        f"mean risk={neutral_result['policy_mean_risk']:.4f}"
    )

    completed = 0
    while completed < args.total_timesteps:
        model.learn(total_timesteps=args.checkpoint_every, reset_num_timesteps=False)
        completed = int(model.num_timesteps)
        no_suffix = model_dir / f"{args.run_name}_step-{completed:06d}"
        model.save(str(no_suffix))
        result, _ = evaluate_candidate(model, steps=completed, output_dir=detail_dir)
        aggregate_rows.append(result)
        print(f"Validation checkpoint completed: step {completed}.")
        print(
            f"  TIR={result['policy_mean_tir_pct']:.2f}% | "
            f"delta={result['mean_tir_delta_pct_points']:+.2f} pp | "
            f"mean risk={result['policy_mean_risk']:.4f}"
        )

    env.close()

    summary = pd.DataFrame(aggregate_rows).sort_values("checkpoint_steps").reset_index(drop=True)
    summary_path = output_dir / "validation_checkpoint_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_progress(summary, output_dir)

    selected = select_checkpoint(summary)
    selected_steps = int(selected["checkpoint_steps"])
    selected_target = ROOT / "models" / f"{args.run_name}_selected_best.zip"
    if selected_steps > 0:
        selected_source = model_dir / f"{args.run_name}_step-{selected_steps:06d}.zip"
        shutil.copy2(selected_source, selected_target)

    selection_report = output_dir / "selected_checkpoint.txt"
    selected_model_text = str(selected_target) if selected_steps > 0 else "neutral_discrete_shield_policy_no_model_file"
    selection_report.write_text(
        "\n".join(
            [
                "Validation-only shielded discrete DQN checkpoint selection",
                f"run_name: {args.run_name}",
                f"selected_steps: {selected_steps}",
                f"selected_model: {selected_model_text}",
                f"mean_tir_pct: {selected['policy_mean_tir_pct']:.4f}",
                f"mean_tir_delta_pct_points: {selected['mean_tir_delta_pct_points']:.4f}",
                f"max_below_pct: {selected['policy_max_below_pct']:.4f}",
                f"max_very_low_pct: {selected['policy_max_very_low_pct']:.4f}",
                f"mean_risk: {selected['policy_mean_risk']:.4f}",
                f"mean_shield_intervention_pct: {selected['mean_shield_intervention_pct']:.4f}",
                "ranking: validation gate first; then minimum severe lows, minimum below range, minimum risk, maximum TIR",
                "final_held_out_suite_used: no",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nShielded discrete DQN checkpoint-selection run completed.")
    print(f"Training patients: {', '.join(TRAINING_PATIENTS)}")
    print(f"Training schedules: {', '.join(TRAINING_SCENARIOS)}")
    print(f"Validation checkpoints evaluated: {len(summary)} (including step zero neutral shield)")
    print(f"Validation summary saved to: {summary_path}")
    print(f"Selected checkpoint: {selected_steps} timesteps")
    print(f"Selected validation mean TIR: {selected['policy_mean_tir_pct']:.2f}%")
    print(f"Selected validation mean TIR delta: {selected['mean_tir_delta_pct_points']:+.2f} percentage points")
    print(f"Selected validation mean risk: {selected['policy_mean_risk']:.4f}")
    if selected_steps == 0:
        print("No learned DQN checkpoint passed the validation gate.")
    else:
        print(f"Selected model copied to: {selected_target}")
    print("Do not run the final held-out suite yet unless you explicitly lock this design.")
    print(f"Selection report saved to: {selection_report}")


if __name__ == "__main__":
    main()
