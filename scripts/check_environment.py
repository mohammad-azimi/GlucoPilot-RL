"""Run a small non-learning experiment to verify that simglucose works locally."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl import make_simglucose_env  # noqa: E402


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    env = make_simglucose_env(patient_name="adult#001", episode_steps=48)
    observation, _ = env.reset(seed=42)
    records: list[dict[str, float | int]] = []

    for step in range(48):
        # This is only a wiring test, not a medical controller or a trained policy.
        basal_action = np.array([0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(basal_action)
        records.append(
            {
                "step": step + 1,
                "cgm_mg_dl": float(observation[0]),
                "reward": float(reward),
                "risk": float(info["risk"]),
                "basal_insulin": float(basal_action[0]),
            }
        )
        if terminated or truncated:
            break

    env.close()
    results = pd.DataFrame(records)
    csv_path = output_dir / "environment_check.csv"
    chart_path = output_dir / "environment_check_glucose_trace.png"
    results.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(results["step"], results["cgm_mg_dl"], linewidth=2)
    plt.axhline(70, linestyle="--", linewidth=1, label="Target lower bound (70 mg/dL)")
    plt.axhline(180, linestyle="--", linewidth=1, label="Target upper bound (180 mg/dL)")
    plt.title("Virtual Patient CGM Trace — Environment Check")
    plt.xlabel("Simulation step")
    plt.ylabel("CGM glucose (mg/dL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=180)
    plt.close()

    print("Environment check passed.")
    print(f"Steps recorded: {len(results)}")
    print(f"Initial CGM: {results.iloc[0]['cgm_mg_dl']:.2f} mg/dL")
    print(f"Final CGM: {results.iloc[-1]['cgm_mg_dl']:.2f} mg/dL")
    print(f"CSV saved to: {csv_path}")
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
