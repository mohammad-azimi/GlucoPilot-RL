"""Reusable simulation runners for baseline and RL comparisons."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from .env import make_simglucose_env
from .scenarios import make_standard_day_scenario


def run_constant_action_episode(
    basal_action: float,
    *,
    patient_name: str = "adult#001",
    episode_steps: int = 288,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one deterministic virtual-patient episode with a fixed action.

    ``basal_action`` is a native simulator action value. It is used only for
    controlled in-silico experimentation and is not a real dosing instruction.
    """
    env: gym.Env = make_simglucose_env(
        patient_name=patient_name,
        episode_steps=episode_steps,
        custom_scenario=make_standard_day_scenario(),
        scenario_tag="standard-day",
    )
    observation, _ = env.reset(seed=seed)
    records: list[dict[str, float | int]] = []

    for step in range(episode_steps):
        action = np.array([basal_action], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": step + 1,
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
