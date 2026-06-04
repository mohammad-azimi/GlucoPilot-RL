"""Reusable simulation runners for baseline and RL comparisons."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from .env import make_simglucose_env
from .scenarios import SENSOR_SAMPLE_MINUTES, STANDARD_DAY_STEPS, make_scenario


def run_constant_action_episode(
    basal_action: float,
    *,
    patient_name: str = "adult#001",
    scenario_name: str = "standard-day",
    episode_steps: int = STANDARD_DAY_STEPS,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one deterministic virtual-patient episode with a fixed action.

    ``basal_action`` is a native simulator action value. It is used only for
    controlled in-silico experimentation and is not a real dosing instruction.
    """
    env: gym.Env = make_simglucose_env(
        patient_name=patient_name,
        episode_steps=episode_steps,
        custom_scenario=make_scenario(scenario_name),
        scenario_tag=scenario_name,
        simulator_seed=seed,
    )
    observation, info = env.reset()
    sample_minutes = float(info["sample_time"])
    if not np.isclose(sample_minutes, SENSOR_SAMPLE_MINUTES):
        env.close()
        raise RuntimeError(
            "Unexpected simglucose sample time: "
            f"{sample_minutes} minutes; expected {SENSOR_SAMPLE_MINUTES}."
        )

    records: list[dict[str, float | int | str]] = []
    for step in range(episode_steps):
        action = np.array([basal_action], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": step + 1,
                "elapsed_hours": (step + 1) * sample_minutes / 60.0,
                "simulation_time": info["time"].isoformat(),
                "patient_name": patient_name,
                "scenario_name": scenario_name,
                "seed": int(seed),
                "cgm_mg_dl": float(observation[0]),
                "reward": float(reward),
                "risk": float(info["risk"]),
                "basal_action": float(basal_action),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return pd.DataFrame(records)
