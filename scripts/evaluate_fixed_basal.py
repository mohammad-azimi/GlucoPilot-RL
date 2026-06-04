"""Evaluate fixed-action baselines for one virtual adult patient over 24 hours."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.metrics import TARGET_HIGH, TARGET_LOW, summarize_episode  # noqa: E402
from glucopilot_rl.scenarios import (  # noqa: E402
    SENSOR_SAMPLE_MINUTES,
    STANDARD_DAY_HOURS,
    STANDARD_DAY_MEALS,
    STANDARD_DAY_STEPS,
)

PATIENT_NAME = "adult#001"
SEED = 42
# Extended after the first dry-run selected the upper boundary (0.0300).
# All values are simulator experiment actions only, never dosing guidance.
CANDIDATE_ACTIONS = np.array(
    [0.0000, 0.0100, 0.0200, 0.0250, 0.0300, 0.0350, 0.0375,
     0.0400, 0.0425, 0.0450, 0.0475, 0.0500, 0.0550],
    dtype=float,
)


def plot_best_trace(trace: pd.DataFrame, best_action: float, output_path: Path) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.fill_between(
        trace["elapsed_hours"], TARGET_LOW, TARGET_HIGH, alpha=0.16,
        label="Target range (70–180 mg/dL)",
    )
    plt.plot(trace["elapsed_hours"], trace["cgm_mg_dl"], linewidth=2, label="CGM glucose")
    y_label = min(float(trace["cgm_mg_dl"].max()) + 3.0, 245.0)
    for meal_hour, meal_grams in STANDARD_DAY_MEALS:
        plt.axvline(meal_hour, linestyle=":", linewidth=1)
        plt.text(meal_hour + 0.08, y_label, f"{meal_grams} g", fontsize=8)
    plt.title(f"Best Fixed-Action Baseline — {PATIENT_NAME} (action={best_action:.4f})")
    plt.xlabel("Simulated time (hours)")
    plt.ylabel("CGM glucose (mg/dL)")
    plt.xlim(0, STANDARD_DAY_HOURS)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close()


def plot_candidate_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("basal_action")
    plt.figure(figsize=(11, 5.5))
    plt.plot(ordered["basal_action"], ordered["time_in_range_pct"], marker="o", label="In range")
    plt.plot(ordered["basal_action"], ordered["time_below_range_pct"], marker="o", label="Below range")
    plt.plot(ordered["basal_action"], ordered["time_above_range_pct"], marker="o", label="Above range")
    plt.title(f"Fixed-Action Baseline Search — {PATIENT_NAME} (24 simulated hours)")
    plt.xlabel("Native simulator basal action")
    plt.ylabel("Episode time (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close()


def main() -> None:
    output_dir = ROOT / "outputs" / "baseline"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | int | str]] = []
    trace_paths: dict[float, Path] = {}

    for basal_action in CANDIDATE_ACTIONS:
        action_value = float(basal_action)
        label = f"fixed_action_{action_value:.4f}"
        trace = run_constant_action_episode(
            action_value,
            patient_name=PATIENT_NAME,
            episode_steps=STANDARD_DAY_STEPS,
            seed=SEED,
        )
        trace_path = traces_dir / f"{label}.csv"
        trace.to_csv(trace_path, index=False)
        trace_paths[action_value] = trace_path
        summaries.append(summarize_episode(trace, label))

    summary = pd.DataFrame(summaries)
    # Selection is safety-first: avoid very low/below-range glucose before
    # minimizing risk or rewarding in-range time.
    ranked = summary.sort_values(
        by=[
            "time_very_low_pct",
            "time_below_range_pct",
            "mean_risk",
            "time_above_range_pct",
            "time_in_range_pct",
        ],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    summary_path = output_dir / "fixed_basal_summary.csv"
    ranked.to_csv(summary_path, index=False)

    best_action = float(ranked.iloc[0]["basal_action"])
    best_trace = pd.read_csv(trace_paths[best_action])
    best_trace_path = output_dir / "best_fixed_basal_trace.csv"
    shutil.copyfile(trace_paths[best_action], best_trace_path)

    comparison_chart = output_dir / "fixed_basal_comparison.png"
    best_trace_chart = output_dir / "best_fixed_basal_trace.png"
    plot_candidate_comparison(ranked, comparison_chart)
    plot_best_trace(best_trace, best_action, best_trace_chart)

    best = ranked.iloc[0]
    duration_hours = len(best_trace) * SENSOR_SAMPLE_MINUTES / 60.0
    print("Corrected fixed-action baseline evaluation passed.")
    print(f"Patient: {PATIENT_NAME}")
    print(f"Seed: {SEED}")
    print(f"Simulated duration: {duration_hours:.1f} hours ({len(best_trace)} steps at {SENSOR_SAMPLE_MINUTES} minutes/step)")
    print(f"Candidates evaluated: {len(CANDIDATE_ACTIONS)}")
    print(f"Best safety-first simulator action: {best_action:.4f}")
    print(f"Time in range: {best['time_in_range_pct']:.2f}%")
    print(f"Time below range: {best['time_below_range_pct']:.2f}%")
    print(f"Time above range: {best['time_above_range_pct']:.2f}%")
    print(f"Mean risk: {best['mean_risk']:.4f}")
    print(f"Summary saved to: {summary_path}")
    print(f"Best trace saved to: {best_trace_path}")
    print(f"Comparison chart saved to: {comparison_chart}")
    print(f"Best trace chart saved to: {best_trace_chart}")


if __name__ == "__main__":
    main()
