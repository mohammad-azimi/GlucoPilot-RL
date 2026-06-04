"""Episode runners used to evaluate a trained Stable-Baselines3 policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .rl_env import AdaptiveGlucoseControlEnv
from .scenarios import SENSOR_SAMPLE_MINUTES, STANDARD_DAY_STEPS


def run_policy_episode(
    model: Any,
    *,
    patient_name: str,
    scenario_name: str,
    seed: int,
    episode_steps: int = STANDARD_DAY_STEPS,
) -> pd.DataFrame:
    """Run a deterministic PPO policy on one fixed simulated test episode."""
    env = AdaptiveGlucoseControlEnv(
        patient_names=[patient_name],
        scenario_names=[scenario_name],
        episode_steps=episode_steps,
        selection_seed=seed,
        fixed_simulator_seed=seed,
    )
    observation, info = env.reset(seed=seed)
    records: list[dict[str, float | int | str]] = []
    for step in range(episode_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": step + 1,
                "elapsed_hours": (step + 1) * SENSOR_SAMPLE_MINUTES / 60.0,
                "patient_name": patient_name,
                "scenario_name": scenario_name,
                "seed": int(seed),
                "cgm_mg_dl": float(info["raw_cgm_mg_dl"]),
                "reward": float(reward),
                "risk": float(info["risk"]),
                "basal_action": float(info["basal_action"]),
            }
        )
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(records)
