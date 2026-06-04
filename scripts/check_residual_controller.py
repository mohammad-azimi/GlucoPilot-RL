"""Verify that zero residual action reproduces the frozen fixed-action baseline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.protocol import REFERENCE_CASE, TUNED_FIXED_ACTION  # noqa: E402
from glucopilot_rl.rl_env import ResidualGlucoseControlEnv, residual_to_simulator_action  # noqa: E402
from glucopilot_rl.experiment import run_constant_action_episode  # noqa: E402
from glucopilot_rl.scenarios import STANDARD_DAY_STEPS  # noqa: E402


def main() -> None:
    patient_name, scenario_name, seed = REFERENCE_CASE
    fixed_trace = run_constant_action_episode(
        TUNED_FIXED_ACTION, patient_name=patient_name, scenario_name=scenario_name, seed=seed
    )
    env = ResidualGlucoseControlEnv(
        patient_names=[patient_name], scenario_names=[scenario_name], fixed_simulator_seed=seed,
        selection_seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    residual_values: list[float] = []
    glucose_values: list[float] = []
    for _ in range(STANDARD_DAY_STEPS):
        observation, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        residual_values.append(float(info["basal_action"]))
        glucose_values.append(float(info["raw_cgm_mg_dl"]))
        if terminated or truncated:
            break
    env.close()

    same_action = np.allclose(residual_values, TUNED_FIXED_ACTION)
    same_glucose = np.allclose(glucose_values, fixed_trace["cgm_mg_dl"].to_numpy())
    if not same_action or not same_glucose:
        raise RuntimeError("Zero residual did not reproduce the frozen fixed-action episode.")
    print("Residual controller sanity check passed.")
    print(f"Reference case: {patient_name} / {scenario_name} / seed={seed}")
    print(
        "Action mapping: "
        f"-1.0 -> {residual_to_simulator_action(-1.0):.4f}, "
        f"0.0 -> {residual_to_simulator_action(0.0):.4f}, "
        f"+1.0 -> {residual_to_simulator_action(1.0):.4f}"
    )
    print("A neutral residual policy exactly reproduces the fixed-action baseline.")


if __name__ == "__main__":
    main()
