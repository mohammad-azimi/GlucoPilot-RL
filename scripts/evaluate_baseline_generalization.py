"""Test the tuned fixed-action baseline on held-out virtual patients and meals."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import TARGET_HIGH, TARGET_LOW, summarize_episode  # noqa: E402
from glucopilot_rl.scenarios import (  # noqa: E402
    STANDARD_DAY_HOURS,
    STANDARD_DAY_STEPS,
    available_scenarios,
    get_scenario_meals,
)

# This value was selected using only the development case:
# adult#001 + standard-day meals + seed 42.
# It is evaluated here without retuning on held-out episodes.
TUNED_FIXED_ACTION = 0.0450
REFERENCE_CASE = ("adult#001", "standard-day", 42)
HELD_OUT_PATIENTS = ["adult#002", "adult#003", "adult#004", "adult#005"]
HELD_OUT_SCENARIOS = list(available_scenarios())


def plot_heatmap(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    table = summary.pivot(index="patient_name", columns="scenario_name", values=metric)
    table = table.reindex(index=HELD_OUT_PATIENTS, columns=HELD_OUT_SCENARIOS)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    image = axis.imshow(table.to_numpy(dtype=float), aspect="auto")
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel("Held-out meal scenario")
    axis.set_ylabel("Held-out virtual patient")
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
    meals = get_scenario_meals(scenario_name)

    plt.figure(figsize=(12, 5.5))
    plt.fill_between(
        trace["elapsed_hours"], TARGET_LOW, TARGET_HIGH, alpha=0.16,
        label="Target range (70–180 mg/dL)",
    )
    plt.plot(trace["elapsed_hours"], trace["cgm_mg_dl"], linewidth=2, label="CGM glucose")
    y_label = min(float(trace["cgm_mg_dl"].max()) + 3.0, 260.0)
    for meal_hour, meal_grams in meals:
        plt.axvline(meal_hour, linestyle=":", linewidth=1)
        plt.text(meal_hour + 0.08, y_label, f"{meal_grams} g", fontsize=8)
    plt.title(
        "Worst Held-Out Fixed-Action Episode — "
        f"{patient_name}, {scenario_name} (action={TUNED_FIXED_ACTION:.4f})"
    )
    plt.xlabel("Simulated time (hours)")
    plt.ylabel("CGM glucose (mg/dL)")
    plt.xlim(0, STANDARD_DAY_HOURS)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close()


def evaluate_case(
    patient_name: str,
    scenario_name: str,
    seed: int,
    split: str,
    traces_dir: Path,
) -> tuple[dict[str, float | int | str], Path]:
    trace = run_constant_action_episode(
        TUNED_FIXED_ACTION,
        patient_name=patient_name,
        scenario_name=scenario_name,
        episode_steps=STANDARD_DAY_STEPS,
        seed=seed,
    )
    trace_path = traces_dir / f"{split}_{patient_name}_{scenario_name}_seed-{seed}.csv"
    trace.to_csv(trace_path, index=False)
    metrics = summarize_episode(trace, "fixed_action_tuned_on_reference")
    metrics.update(
        {
            "split": split,
            "patient_name": patient_name,
            "scenario_name": scenario_name,
            "seed": int(seed),
        }
    )
    return metrics, trace_path


def main() -> None:
    output_dir = ROOT / "outputs" / "generalization"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    trace_paths: dict[tuple[str, str, int], Path] = {}

    patient_name, scenario_name, seed = REFERENCE_CASE
    metrics, trace_path = evaluate_case(
        patient_name, scenario_name, seed, "development_reference", traces_dir
    )
    rows.append(metrics)
    trace_paths[(patient_name, scenario_name, seed)] = trace_path

    for patient_index, held_patient in enumerate(HELD_OUT_PATIENTS):
        for scenario_index, held_scenario in enumerate(HELD_OUT_SCENARIOS):
            held_seed = 1000 + patient_index * 100 + scenario_index
            metrics, trace_path = evaluate_case(
                held_patient, held_scenario, held_seed, "held_out", traces_dir
            )
            rows.append(metrics)
            trace_paths[(held_patient, held_scenario, held_seed)] = trace_path

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "fixed_action_generalization_summary.csv"
    summary.to_csv(summary_path, index=False)

    held_out = summary[summary["split"] == "held_out"].copy()
    worst = held_out.sort_values(
        by=["time_very_low_pct", "time_below_range_pct", "mean_risk", "time_in_range_pct"],
        ascending=[False, False, False, True],
    ).iloc[0]
    worst_key = (str(worst["patient_name"]), str(worst["scenario_name"]), int(worst["seed"]))
    worst_trace = pd.read_csv(trace_paths[worst_key])

    tir_heatmap_path = output_dir / "held_out_time_in_range_heatmap.png"
    risk_heatmap_path = output_dir / "held_out_mean_risk_heatmap.png"
    worst_trace_path = output_dir / "worst_held_out_trace.png"
    plot_heatmap(
        held_out,
        "time_in_range_pct",
        "Fixed Action Generalization — Time in Range (%)",
        tir_heatmap_path,
    )
    plot_heatmap(
        held_out,
        "mean_risk",
        "Fixed Action Generalization — Mean Simulated Risk",
        risk_heatmap_path,
    )
    plot_worst_trace(worst_trace, worst_trace_path)

    reference = summary[summary["split"] == "development_reference"].iloc[0]
    print("Fixed-action generalization evaluation passed.")
    print(f"Tuned action from development case: {TUNED_FIXED_ACTION:.4f}")
    print(
        "Development reference: "
        f"{reference['patient_name']} / {reference['scenario_name']} / "
        f"TIR={reference['time_in_range_pct']:.2f}% / mean risk={reference['mean_risk']:.4f}"
    )
    print(f"Held-out episodes evaluated: {len(held_out)}")
    print(f"Held-out mean time in range: {held_out['time_in_range_pct'].mean():.2f}%")
    print(f"Held-out minimum time in range: {held_out['time_in_range_pct'].min():.2f}%")
    print(f"Held-out mean risk: {held_out['mean_risk'].mean():.4f}")
    print(
        "Held-out worst case: "
        f"{worst['patient_name']} / {worst['scenario_name']} / "
        f"TIR={worst['time_in_range_pct']:.2f}% / "
        f"below={worst['time_below_range_pct']:.2f}% / "
        f"above={worst['time_above_range_pct']:.2f}% / mean risk={worst['mean_risk']:.4f}"
    )
    print(f"Summary saved to: {summary_path}")
    print(f"Time-in-range heatmap saved to: {tir_heatmap_path}")
    print(f"Mean-risk heatmap saved to: {risk_heatmap_path}")
    print(f"Worst-case chart saved to: {worst_trace_path}")


if __name__ == "__main__":
    main()
