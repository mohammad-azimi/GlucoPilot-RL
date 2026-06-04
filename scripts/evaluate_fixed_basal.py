"""Evaluate reproducible fixed-action baselines for one virtual adult patient."""

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
from glucopilot_rl.scenarios import STANDARD_DAY_MEALS  # noqa: E402

PATIENT_NAME = "adult#001"
EPISODE_STEPS = 288
SEED = 42
CANDIDATE_ACTIONS = np.linspace(0.0, 0.030, 13)


def plot_best_trace(trace: pd.DataFrame, best_action: float, output_path: Path) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.fill_between(
        trace["step"], TARGET_LOW, TARGET_HIGH, alpha=0.16, label="Target range (70–180 mg/dL)"
    )
    plt.plot(trace["step"], trace["cgm_mg_dl"], linewidth=2, label="CGM glucose")
    for meal_hour, meal_grams in STANDARD_DAY_MEALS:
        meal_step = meal_hour * 12
        plt.axvline(meal_step, linestyle=":", linewidth=1)
        plt.text(meal_step + 1, trace["cgm_mg_dl"].max(), f"{meal_grams} g", fontsize=8)
    plt.title(f"Best Fixed-Action Baseline — {PATIENT_NAME} (action={best_action:.4f})")
    plt.xlabel("Simulation step")
    plt.ylabel("CGM glucose (mg/dL)")
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
    plt.title(f"Fixed-Action Baseline Search — {PATIENT_NAME}")
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
            episode_steps=EPISODE_STEPS,
            seed=SEED,
        )
        trace_path = traces_dir / f"{label}.csv"
        trace.to_csv(trace_path, index=False)
        trace_paths[action_value] = trace_path
        summaries.append(summarize_episode(trace, label))

    summary = pd.DataFrame(summaries)
    # Simglucose defines its default reward from the risk-index change; mean
    # risk is therefore used as the primary reproducible baseline criterion.
    ranked = summary.sort_values(
        by=["mean_risk", "time_below_range_pct", "time_in_range_pct"],
        ascending=[True, True, False],
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
    print("Fixed-action baseline evaluation passed.")
    print(f"Patient: {PATIENT_NAME}")
    print(f"Candidates evaluated: {len(CANDIDATE_ACTIONS)}")
    print(f"Best native simulator action: {best_action:.4f}")
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
