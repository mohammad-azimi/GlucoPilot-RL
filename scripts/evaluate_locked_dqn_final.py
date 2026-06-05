"""One-time final held-out evaluation for the locked shielded DQN.

Run this only after a model is explicitly locked using validation results.
The held-out cases are the final comparison suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.discrete_experiment import run_discrete_policy_episode  # noqa: E402
from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import summarize_episode  # noqa: E402
from glucopilot_rl.protocol import HELD_OUT_PATIENTS, HELD_OUT_SCENARIOS, TUNED_FIXED_ACTION, held_out_cases  # noqa: E402
from glucopilot_rl.scenarios import get_scenario_meals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a locked shielded DQN on the final held-out suite.")
    parser.add_argument(
        "--model",
        default=r"models\dqn_discrete_shield_locked_010240.zip",
        help="Locked model path relative to the project root.",
    )
    parser.add_argument(
        "--run-name",
        default="dqn_discrete_shield_locked_010240",
        help="Output subdirectory name under outputs/final_held_out/.",
    )
    return parser.parse_args()


def heatmap(
    values: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    output_path: Path,
    fmt: str = ".1f",
) -> None:
    patients = HELD_OUT_PATIENTS
    scenarios = HELD_OUT_SCENARIOS
    matrix = values.pivot(index="patient_name", columns="scenario_name", values="value").loc[patients, scenarios]

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    image = axis.imshow(matrix.values, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Final held-out meal scenario")
    axis.set_ylabel("Final held-out virtual patient")
    axis.set_xticks(np.arange(len(scenarios)))
    axis.set_xticklabels(scenarios, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(patients)))
    axis.set_yticklabels(patients)

    for row_index in range(len(patients)):
        for col_index in range(len(scenarios)):
            axis.text(
                col_index,
                row_index,
                format(matrix.values[row_index, col_index], fmt),
                ha="center",
                va="center",
                color="black",
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def plot_worst_case(trace: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.axhspan(70.0, 180.0, alpha=0.18, label="Target range (70–180 mg/dL)")
    axis.plot(trace["elapsed_hours"], trace["cgm_mg_dl"], label="Locked DQN CGM glucose")
    for meal_hour, grams in get_scenario_meals(str(trace["scenario_name"].iloc[0])):
        axis.axvline(meal_hour, linestyle=":", linewidth=1.0)
        axis.text(meal_hour + 0.06, axis.get_ylim()[1] * 0.92, f"{grams:g} g", fontsize=8)
    axis.set_title(title)
    axis.set_xlabel("Simulated time (hours)")
    axis.set_ylabel("CGM glucose (mg/dL)")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model_path = ROOT / args.model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Locked model not found: {model_path}\n"
            "Run scripts\\lock_dqn_checkpoint.py first, or pass --model to a locked model."
        )

    output_dir = ROOT / "outputs" / "final_held_out" / args.run_name
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    model = DQN.load(str(model_path))

    rows: list[dict[str, float | int | str]] = []
    worst_trace: pd.DataFrame | None = None
    worst_row: dict[str, float | int | str] | None = None

    for patient_name, scenario_name, seed in held_out_cases():
        fixed_trace = run_constant_action_episode(
            TUNED_FIXED_ACTION,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )
        dqn_trace = run_discrete_policy_episode(
            model,
            patient_name=patient_name,
            scenario_name=scenario_name,
            seed=seed,
        )

        fixed_trace.to_csv(trace_dir / f"fixed_{patient_name}_{scenario_name}.csv", index=False)
        dqn_trace.to_csv(trace_dir / f"dqn_{patient_name}_{scenario_name}.csv", index=False)

        fixed_metrics = summarize_episode(fixed_trace, "fixed_action")
        dqn_metrics = summarize_episode(dqn_trace, "locked_shielded_dqn")
        row: dict[str, float | int | str] = {
            "patient_name": patient_name,
            "scenario_name": scenario_name,
            "seed": seed,
        }
        row.update({f"{key}_fixed": value for key, value in fixed_metrics.items()})
        row.update({f"{key}_dqn": value for key, value in dqn_metrics.items()})
        row["tir_delta_dqn_minus_fixed"] = (
            float(dqn_metrics["time_in_range_pct"]) - float(fixed_metrics["time_in_range_pct"])
        )
        row["risk_delta_dqn_minus_fixed"] = (
            float(dqn_metrics["mean_risk"]) - float(fixed_metrics["mean_risk"])
        )
        row["below_delta_dqn_minus_fixed"] = (
            float(dqn_metrics["time_below_range_pct"]) - float(fixed_metrics["time_below_range_pct"])
        )
        row["very_low_delta_dqn_minus_fixed"] = (
            float(dqn_metrics["time_very_low_pct"]) - float(fixed_metrics["time_very_low_pct"])
        )
        row["shield_intervention_pct"] = float(100.0 * (dqn_trace["shield_reason"] != "unchanged").mean())
        rows.append(row)

        if worst_row is None or float(row["mean_risk_dqn"]) > float(worst_row["mean_risk_dqn"]):
            worst_row = row
            worst_trace = dqn_trace

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "locked_dqn_vs_fixed_held_out_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = {
        "final_held_out_episodes": len(summary),
        "fixed_mean_tir_pct": float(summary["time_in_range_pct_fixed"].mean()),
        "dqn_mean_tir_pct": float(summary["time_in_range_pct_dqn"].mean()),
        "mean_tir_delta_pct_points": float(summary["tir_delta_dqn_minus_fixed"].mean()),
        "fixed_mean_risk": float(summary["mean_risk_fixed"].mean()),
        "dqn_mean_risk": float(summary["mean_risk_dqn"].mean()),
        "mean_risk_delta": float(summary["risk_delta_dqn_minus_fixed"].mean()),
        "fixed_max_below_pct": float(summary["time_below_range_pct_fixed"].max()),
        "dqn_max_below_pct": float(summary["time_below_range_pct_dqn"].max()),
        "fixed_max_very_low_pct": float(summary["time_very_low_pct_fixed"].max()),
        "dqn_max_very_low_pct": float(summary["time_very_low_pct_dqn"].max()),
        "dqn_mean_shield_intervention_pct": float(summary["shield_intervention_pct"].mean()),
    }
    aggregate_path = output_dir / "final_held_out_aggregate_metrics.txt"
    aggregate_path.write_text(
        "\n".join([f"{key}: {value}" for key, value in aggregate.items()])
        + "\nresearch_simulation_only: not for real medical decisions\n",
        encoding="utf-8",
    )

    heatmap(
        summary.assign(value=summary["time_in_range_pct_dqn"]),
        title="Final Held-Out Locked DQN — Time in Range (%)",
        colorbar_label="time in range pct",
        output_path=output_dir / "locked_dqn_held_out_time_in_range_heatmap.png",
    )
    heatmap(
        summary.assign(value=summary["tir_delta_dqn_minus_fixed"]),
        title="Final Held-Out DQN Minus Fixed — Time-in-Range Delta",
        colorbar_label="TIR delta percentage points",
        output_path=output_dir / "locked_dqn_minus_fixed_tir_delta_heatmap.png",
    )
    heatmap(
        summary.assign(value=summary["risk_delta_dqn_minus_fixed"]),
        title="Final Held-Out DQN Minus Fixed — Mean Risk Delta",
        colorbar_label="mean risk delta",
        output_path=output_dir / "locked_dqn_minus_fixed_risk_delta_heatmap.png",
    )

    if worst_trace is not None and worst_row is not None:
        plot_worst_case(
            worst_trace,
            output_dir / "worst_locked_dqn_held_out_trace.png",
            (
                "Worst Final Held-Out Locked DQN Episode — "
                f"{worst_row['patient_name']}, {worst_row['scenario_name']}"
            ),
        )

    print("Final held-out locked DQN evaluation completed.")
    print(f"Locked model: {model_path}")
    print(f"Final held-out episodes evaluated: {len(summary)}")
    print(f"Fixed mean time in range: {aggregate['fixed_mean_tir_pct']:.2f}%")
    print(f"Locked DQN mean time in range: {aggregate['dqn_mean_tir_pct']:.2f}%")
    print(f"Mean TIR delta: {aggregate['mean_tir_delta_pct_points']:+.2f} percentage points")
    print(f"Fixed mean risk: {aggregate['fixed_mean_risk']:.4f}")
    print(f"Locked DQN mean risk: {aggregate['dqn_mean_risk']:.4f}")
    print(f"Mean risk delta: {aggregate['mean_risk_delta']:+.4f}")
    print(f"Fixed max below range: {aggregate['fixed_max_below_pct']:.2f}%")
    print(f"Locked DQN max below range: {aggregate['dqn_max_below_pct']:.2f}%")
    print(f"Fixed max very low: {aggregate['fixed_max_very_low_pct']:.2f}%")
    print(f"Locked DQN max very low: {aggregate['dqn_max_very_low_pct']:.2f}%")
    print(f"Mean shield intervention rate: {aggregate['dqn_mean_shield_intervention_pct']:.2f}%")
    print(f"Summary saved to: {summary_path}")
    print(f"Aggregate metrics saved to: {aggregate_path}")
    print("This is the one-time final held-out comparison for the locked model.")
    print("Research simulation only; not for real medical decision-making.")


if __name__ == "__main__":
    main()
