"""Final-only evaluation of a residual PPO model on the frozen held-out suite.

Do not use this script while designing or tuning the controller. Use
``evaluate_ppo_validation.py`` during iteration instead.
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

from glucopilot_rl.metrics import TARGET_HIGH, TARGET_LOW, summarize_episode  # noqa: E402
from glucopilot_rl.protocol import (  # noqa: E402
    HELD_OUT_PATIENTS,
    HELD_OUT_SCENARIOS,
    held_out_cases,
)
from glucopilot_rl.rl_experiment import run_policy_episode  # noqa: E402
from glucopilot_rl.scenarios import STANDARD_DAY_HOURS, get_scenario_meals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO against the frozen held-out suite.")
    parser.add_argument(
        "--model",
        default="models/ppo_residual_final_model.zip",
        help="Locked residual model path relative to project root or an absolute path.",
    )
    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="Required acknowledgement that this runs the final held-out comparison.",
    )
    return parser.parse_args()


def plot_heatmap(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    table = summary.pivot(index="patient_name", columns="scenario_name", values=metric)
    table = table.reindex(index=HELD_OUT_PATIENTS, columns=HELD_OUT_SCENARIOS)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    image = axis.imshow(table.to_numpy(dtype=float), aspect="auto")
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel("Frozen held-out meal scenario")
    axis.set_ylabel("Frozen held-out virtual patient")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=metric.replace("_", " "))
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            value = float(table.iloc[row, column])
            axis.text(column, row, f"{value:.1f}", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def plot_worst_trace(trace: pd.DataFrame, output_path: Path) -> None:
    patient_name = str(trace.iloc[0]["patient_name"])
    scenario_name = str(trace.iloc[0]["scenario_name"])
    plt.figure(figsize=(12, 5.5))
    plt.fill_between(
        trace["elapsed_hours"], TARGET_LOW, TARGET_HIGH, alpha=0.16,
        label="Target range (70–180 mg/dL)",
    )
    plt.plot(trace["elapsed_hours"], trace["cgm_mg_dl"], linewidth=2, label="PPO CGM glucose")
    y_label = min(float(trace["cgm_mg_dl"].max()) + 3.0, 260.0)
    for meal_hour, meal_grams in get_scenario_meals(scenario_name):
        plt.axvline(meal_hour, linestyle=":", linewidth=1)
        plt.text(meal_hour + 0.08, y_label, f"{meal_grams} g", fontsize=8)
    plt.title(f"Worst Held-Out PPO Episode — {patient_name}, {scenario_name}")
    plt.xlabel("Simulated time (hours)")
    plt.ylabel("CGM glucose (mg/dL)")
    plt.xlim(0, STANDARD_DAY_HOURS)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.confirm_final_evaluation:
        raise RuntimeError(
            "Held-out evaluation is now closed during model development. "
            "Run scripts/evaluate_ppo_validation.py instead. "
            "Use --confirm-final-evaluation only after locking a final model."
        )
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}. Train it first.")
    if model_path.name == "ppo_smoke_model.zip":
        raise ValueError("The Phase 4 native-action smoke model is incompatible with the residual environment.")

    baseline_path = ROOT / "outputs" / "generalization" / "fixed_action_generalization_summary.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Baseline generalization results not found. Run scripts/evaluate_baseline_generalization.py first."
        )

    output_dir = ROOT / "outputs" / "ppo_evaluation"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path))

    rows: list[dict[str, float | int | str]] = []
    traces: dict[tuple[str, str, int], pd.DataFrame] = {}
    for patient_name, scenario_name, seed in held_out_cases():
        trace = run_policy_episode(
            model, patient_name=patient_name, scenario_name=scenario_name, seed=seed
        )
        trace.to_csv(traces_dir / f"ppo_{patient_name}_{scenario_name}_seed-{seed}.csv", index=False)
        metrics = summarize_episode(trace, "ppo_policy")
        metrics.update({"patient_name": patient_name, "scenario_name": scenario_name, "seed": seed})
        rows.append(metrics)
        traces[(patient_name, scenario_name, seed)] = trace

    ppo_summary = pd.DataFrame(rows)
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["split"] == "held_out"].copy()
    merged = ppo_summary.merge(
        baseline,
        on=["patient_name", "scenario_name", "seed"],
        suffixes=("_ppo", "_fixed"),
    )
    merged["tir_delta_ppo_minus_fixed"] = (
        merged["time_in_range_pct_ppo"] - merged["time_in_range_pct_fixed"]
    )
    merged["risk_delta_ppo_minus_fixed"] = merged["mean_risk_ppo"] - merged["mean_risk_fixed"]

    summary_path = output_dir / "ppo_vs_fixed_held_out_summary.csv"
    merged.to_csv(summary_path, index=False)
    plot_heatmap(
        ppo_summary,
        "time_in_range_pct",
        "PPO Held-Out Evaluation — Time in Range (%)",
        output_dir / "ppo_held_out_time_in_range_heatmap.png",
    )
    plot_heatmap(
        merged.rename(columns={"tir_delta_ppo_minus_fixed": "tir_delta"}),
        "tir_delta",
        "PPO Minus Fixed Action — Time-in-Range Difference (percentage points)",
        output_dir / "ppo_minus_fixed_tir_delta_heatmap.png",
    )
    worst = ppo_summary.sort_values(
        by=["time_very_low_pct", "time_below_range_pct", "mean_risk", "time_in_range_pct"],
        ascending=[False, False, False, True],
    ).iloc[0]
    key = (str(worst["patient_name"]), str(worst["scenario_name"]), int(worst["seed"]))
    plot_worst_trace(traces[key], output_dir / "worst_held_out_ppo_trace.png")

    print("PPO held-out evaluation completed.")
    print(f"Model: {model_path}")
    print(f"Frozen held-out episodes evaluated: {len(ppo_summary)}")
    print(f"PPO mean time in range: {ppo_summary['time_in_range_pct'].mean():.2f}%")
    print(f"Fixed-action mean time in range: {merged['time_in_range_pct_fixed'].mean():.2f}%")
    print(f"Mean TIR difference (PPO - fixed): {merged['tir_delta_ppo_minus_fixed'].mean():.2f} percentage points")
    print(f"PPO mean risk: {ppo_summary['mean_risk'].mean():.4f}")
    print(f"Fixed-action mean risk: {merged['mean_risk_fixed'].mean():.4f}")
    print(
        "Worst PPO case: "
        f"{worst['patient_name']} / {worst['scenario_name']} / "
        f"TIR={worst['time_in_range_pct']:.2f}% / below={worst['time_below_range_pct']:.2f}% / "
        f"mean risk={worst['mean_risk']:.4f}"
    )
    print(f"Comparison summary saved to: {summary_path}")
    print("Final held-out evaluation completed for the locked residual model.")


if __name__ == "__main__":
    main()
