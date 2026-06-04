"""Evaluate a residual PPO model against the fixed baseline on validation cases.

Validation cases are intentionally distinct from the final held-out suite, so
controller design can be improved without repeatedly tuning on final results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import TARGET_HIGH, TARGET_LOW, summarize_episode  # noqa: E402
from glucopilot_rl.protocol import (  # noqa: E402
    TUNED_FIXED_ACTION,
    VALIDATION_PATIENTS,
    VALIDATION_SCENARIOS,
    validation_cases,
)
from glucopilot_rl.rl_experiment import run_policy_episode  # noqa: E402
from glucopilot_rl.scenarios import STANDARD_DAY_HOURS, get_scenario_meals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual PPO on the validation suite.")
    parser.add_argument(
        "--model", default="models/ppo_residual_smoke_model.zip",
        help="Residual PPO model path relative to the project root or absolute path.",
    )
    return parser.parse_args()


def plot_heatmap(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    table = summary.pivot(index="patient_name", columns="scenario_name", values=metric)
    table = table.reindex(index=VALIDATION_PATIENTS, columns=VALIDATION_SCENARIOS)
    figure, axis = plt.subplots(figsize=(10, 3.8))
    image = axis.imshow(table.to_numpy(dtype=float), aspect="auto")
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel("Validation meal scenario")
    axis.set_ylabel("Validation virtual patient")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=metric.replace("_", " "))
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            axis.text(column, row, f"{float(table.iloc[row, column]):.1f}", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def plot_case_comparison(
    fixed_trace: pd.DataFrame,
    ppo_trace: pd.DataFrame,
    patient_name: str,
    scenario_name: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.fill_between(
        ppo_trace["elapsed_hours"], TARGET_LOW, TARGET_HIGH, alpha=0.16,
        label="Target range (70–180 mg/dL)",
    )
    axis.plot(fixed_trace["elapsed_hours"], fixed_trace["cgm_mg_dl"], linewidth=2, label="Fixed action")
    axis.plot(ppo_trace["elapsed_hours"], ppo_trace["cgm_mg_dl"], linewidth=2, label="Residual PPO")
    top = min(max(float(fixed_trace["cgm_mg_dl"].max()), float(ppo_trace["cgm_mg_dl"].max())) + 3.0, 300.0)
    for meal_hour, meal_grams in get_scenario_meals(scenario_name):
        axis.axvline(meal_hour, linestyle=":", linewidth=1)
        axis.text(meal_hour + 0.08, top, f"{meal_grams} g", fontsize=8)
    axis.set_title(f"Validation Comparison — {patient_name}, {scenario_name}")
    axis.set_xlabel("Simulated time (hours)")
    axis.set_ylabel("CGM glucose (mg/dL)")
    axis.set_xlim(0, STANDARD_DAY_HOURS)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Residual PPO model not found: {model_path}. Train it first.")
    if model_path.name == "ppo_smoke_model.zip":
        raise ValueError("The Phase 4 native-action smoke model is incompatible with the residual environment.")

    output_dir = ROOT / "outputs" / "validation"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path))

    rows: list[dict[str, float | int | str]] = []
    comparison_traces: dict[tuple[str, str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for patient_name, scenario_name, seed in validation_cases():
        fixed_trace = run_constant_action_episode(
            TUNED_FIXED_ACTION, patient_name=patient_name, scenario_name=scenario_name, seed=seed
        )
        ppo_trace = run_policy_episode(
            model, patient_name=patient_name, scenario_name=scenario_name, seed=seed
        )
        fixed_trace.to_csv(traces_dir / f"fixed_{patient_name}_{scenario_name}_seed-{seed}.csv", index=False)
        ppo_trace.to_csv(traces_dir / f"ppo_{patient_name}_{scenario_name}_seed-{seed}.csv", index=False)
        fixed_metrics = summarize_episode(fixed_trace, "fixed_action")
        ppo_metrics = summarize_episode(ppo_trace, "residual_ppo")
        row: dict[str, float | int | str] = {
            "patient_name": patient_name,
            "scenario_name": scenario_name,
            "seed": seed,
        }
        row.update({f"{key}_fixed": value for key, value in fixed_metrics.items()})
        row.update({f"{key}_ppo": value for key, value in ppo_metrics.items()})
        row["tir_delta_ppo_minus_fixed"] = (
            float(ppo_metrics["time_in_range_pct"]) - float(fixed_metrics["time_in_range_pct"])
        )
        row["risk_delta_ppo_minus_fixed"] = float(ppo_metrics["mean_risk"]) - float(fixed_metrics["mean_risk"])
        rows.append(row)
        comparison_traces[(patient_name, scenario_name, seed)] = (fixed_trace, ppo_trace)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "residual_ppo_vs_fixed_validation_summary.csv"
    summary.to_csv(summary_path, index=False)
    ppo_view = summary.rename(columns={"time_in_range_pct_ppo": "ppo_time_in_range_pct"})
    plot_heatmap(
        ppo_view, "ppo_time_in_range_pct", "Residual PPO Validation — Time in Range (%)",
        output_dir / "ppo_validation_time_in_range_heatmap.png",
    )
    plot_heatmap(
        summary, "tir_delta_ppo_minus_fixed",
        "Residual PPO Minus Fixed Action — Validation TIR Difference (percentage points)",
        output_dir / "ppo_minus_fixed_validation_tir_delta_heatmap.png",
    )
    worst = summary.sort_values(
        by=["time_very_low_pct_ppo", "time_below_range_pct_ppo", "mean_risk_ppo", "time_in_range_pct_ppo"],
        ascending=[False, False, False, True],
    ).iloc[0]
    key = (str(worst["patient_name"]), str(worst["scenario_name"]), int(worst["seed"]))
    plot_case_comparison(
        *comparison_traces[key], key[0], key[1], output_dir / "worst_validation_comparison_trace.png"
    )

    print("Residual PPO validation evaluation completed.")
    print(f"Model: {model_path}")
    print(f"Validation episodes evaluated: {len(summary)}")
    print(f"Residual PPO mean time in range: {summary['time_in_range_pct_ppo'].mean():.2f}%")
    print(f"Fixed-action mean time in range: {summary['time_in_range_pct_fixed'].mean():.2f}%")
    print(
        "Mean TIR difference (residual PPO - fixed): "
        f"{summary['tir_delta_ppo_minus_fixed'].mean():.2f} percentage points"
    )
    print(f"Residual PPO mean risk: {summary['mean_risk_ppo'].mean():.4f}")
    print(f"Fixed-action mean risk: {summary['mean_risk_fixed'].mean():.4f}")
    print(
        "Worst residual PPO validation case: "
        f"{worst['patient_name']} / {worst['scenario_name']} / "
        f"TIR={worst['time_in_range_pct_ppo']:.2f}% / "
        f"below={worst['time_below_range_pct_ppo']:.2f}% / "
        f"mean risk={worst['mean_risk_ppo']:.4f}"
    )
    print(f"Comparison summary saved to: {summary_path}")
    print("The final held-out suite was not used in this Phase 5 validation check.")


if __name__ == "__main__":
    main()
