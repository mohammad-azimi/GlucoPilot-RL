"""Evaluate a neutral discrete policy with safety shield on validation cases."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.discrete_experiment import ConstantDiscretePolicy, run_discrete_policy_episode  # noqa: E402
from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import summarize_episode  # noqa: E402
from glucopilot_rl.protocol import TUNED_FIXED_ACTION, validation_cases  # noqa: E402


def plot_validation_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["patient_name"] + "\n" + summary["scenario_name"]
    x = range(len(summary))

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar([i - 0.2 for i in x], summary["time_in_range_pct_fixed"], width=0.4, label="Fixed reference")
    axis.bar([i + 0.2 for i in x], summary["time_in_range_pct_shield"], width=0.4, label="Neutral + safety shield")
    axis.set_title("Validation Time in Range: Fixed vs Neutral Safety Shield")
    axis.set_ylabel("Time in range (%)")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "shielded_neutral_validation_tir_bars.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar([i - 0.2 for i in x], summary["mean_risk_fixed"], width=0.4, label="Fixed reference")
    axis.bar([i + 0.2 for i in x], summary["mean_risk_shield"], width=0.4, label="Neutral + safety shield")
    axis.set_title("Validation Mean Simulated Risk: Fixed vs Neutral Safety Shield")
    axis.set_ylabel("Mean simulated risk")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "shielded_neutral_validation_risk_bars.png", dpi=190)
    plt.close(figure)


def main() -> None:
    output_dir = ROOT / "outputs" / "discrete_shield_validation"
    traces_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    policy = ConstantDiscretePolicy()
    rows: list[dict[str, float | int | str]] = []

    for patient_name, scenario_name, seed in validation_cases():
        fixed_trace = run_constant_action_episode(
            TUNED_FIXED_ACTION,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )
        shield_trace = run_discrete_policy_episode(
            policy,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )

        fixed_trace.to_csv(traces_dir / f"fixed_{patient_name}_{scenario_name}.csv", index=False)
        shield_trace.to_csv(traces_dir / f"shield_{patient_name}_{scenario_name}.csv", index=False)

        fixed_metrics = summarize_episode(fixed_trace, "fixed_action")
        shield_metrics = summarize_episode(shield_trace, "neutral_shield")
        row: dict[str, float | int | str] = {
            "patient_name": patient_name,
            "scenario_name": scenario_name,
            "seed": seed,
        }
        row.update({f"{key}_fixed": value for key, value in fixed_metrics.items()})
        row.update({f"{key}_shield": value for key, value in shield_metrics.items()})
        row["tir_delta_shield_minus_fixed"] = (
            float(shield_metrics["time_in_range_pct"]) - float(fixed_metrics["time_in_range_pct"])
        )
        row["risk_delta_shield_minus_fixed"] = (
            float(shield_metrics["mean_risk"]) - float(fixed_metrics["mean_risk"])
        )
        row["shield_intervention_pct"] = float(
            100.0 * (shield_trace["shield_reason"] != "unchanged").mean()
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "shielded_neutral_validation_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_validation_bars(summary, output_dir)

    mean_fixed_tir = float(summary["time_in_range_pct_fixed"].mean())
    mean_shield_tir = float(summary["time_in_range_pct_shield"].mean())
    mean_fixed_risk = float(summary["mean_risk_fixed"].mean())
    mean_shield_risk = float(summary["mean_risk_shield"].mean())
    mean_intervention = float(summary["shield_intervention_pct"].mean())

    print("Discrete safety-shield validation completed.")
    print(f"Validation episodes evaluated: {len(summary)}")
    print(f"Fixed mean time in range: {mean_fixed_tir:.2f}%")
    print(f"Shielded-neutral mean time in range: {mean_shield_tir:.2f}%")
    print(f"Mean TIR delta: {mean_shield_tir - mean_fixed_tir:+.2f} percentage points")
    print(f"Fixed mean risk: {mean_fixed_risk:.4f}")
    print(f"Shielded-neutral mean risk: {mean_shield_risk:.4f}")
    print(f"Mean risk delta: {mean_shield_risk - mean_fixed_risk:+.4f}")
    print(f"Mean shield intervention rate: {mean_intervention:.2f}% of steps")
    print(f"Summary saved to: {summary_path}")
    print(f"Charts saved to: {output_dir}")
    print("The final held-out suite was not used.")


if __name__ == "__main__":
    main()
