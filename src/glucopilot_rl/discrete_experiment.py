"""Episode runners for discrete shielded residual policies."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from .discrete_env import DiscreteShieldedGlucoseControlEnv, NEUTRAL_ACTION_INDEX
from .scenarios import SENSOR_SAMPLE_MINUTES, STANDARD_DAY_STEPS


class PredictsAction(Protocol):
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        ...


class ConstantDiscretePolicy:
    """Small adapter with a Stable-Baselines-like predict method."""

    def __init__(self, action_index: int = NEUTRAL_ACTION_INDEX) -> None:
        self.action_index = int(action_index)

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
        del observation, deterministic
        return self.action_index, None


def run_discrete_policy_episode(
    model: PredictsAction,
    *,
    patient_name: str,
    scenario_name: str,
    seed: int,
    episode_steps: int = STANDARD_DAY_STEPS,
) -> pd.DataFrame:
    """Run deterministic discrete-policy inference on a fixed simulation case."""
    env = DiscreteShieldedGlucoseControlEnv(
        patient_names=[patient_name],
        scenario_names=[scenario_name],
        episode_steps=episode_steps,
        selection_seed=seed,
        fixed_simulator_seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    records: list[dict[str, float | int | str]] = []
    for step in range(episode_steps):
        action_index, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action_index)
        records.append(
            {
                "step": step + 1,
                "elapsed_hours": (step + 1) * SENSOR_SAMPLE_MINUTES / 60.0,
                "simulation_time": info["time"].isoformat(),
                "patient_name": patient_name,
                "scenario_name": scenario_name,
                "seed": int(seed),
                "cgm_mg_dl": float(info["raw_cgm_mg_dl"]),
                "reward": float(reward),
                "risk": float(info["risk"]),
                "basal_action": float(info["basal_action"]),
                "requested_residual_action": float(info["requested_residual_action"]),
                "residual_action": float(info["residual_action"]),
                "shield_reason": str(info["shield_reason"]),
            }
        )
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(records)
