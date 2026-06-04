"""Gymnasium environment helpers for GlucoPilot-RL.

The Gymnasium adapter shipped by simglucose defines a one-dimensional Box
space for actions, while its internal legacy simulator consumes a scalar basal
value. The wrapper below keeps the vector-shaped public API expected by
reinforcement-learning libraries and converts the action before it reaches the
simulator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry


class ScalarBasalActionWrapper(gym.ActionWrapper):
    """Convert Gymnasium/SB3 vector actions to simglucose scalar actions."""

    def action(self, action: np.ndarray | float) -> float:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 1:
            raise ValueError(
                "Expected one basal-insulin action value, "
                f"got shape {action_array.shape}."
            )
        return float(action_array[0])


def make_simglucose_env(
    patient_name: str = "adult#001",
    episode_steps: int = 480,
    *,
    custom_scenario: Any | None = None,
    scenario_tag: str = "default",
    simulator_seed: int | None = None,
    reward_fun: Callable[..., float] | None = None,
) -> gym.Env:
    """Create a Gymnasium-compatible simulated-patient environment.

    ``simulator_seed`` is supplied when the underlying simglucose object is
    constructed. This matters because its Gymnasium ``reset(seed=...)`` method
    does not reseed the internal sensor and virtual-patient streams.
    """
    safe_patient_id = patient_name.replace("#", "-")
    safe_scenario_tag = scenario_tag.replace("_", "-").replace(" ", "-")
    seed_tag = "none" if simulator_seed is None else str(simulator_seed)
    env_id = (
        f"GlucoPilotRL/{safe_patient_id}-{safe_scenario_tag}-"
        f"{episode_steps}-seed-{seed_tag}-v0"
    )

    kwargs: dict[str, Any] = {"patient_name": patient_name, "seed": simulator_seed}
    if custom_scenario is not None:
        kwargs["custom_scenario"] = custom_scenario
    if reward_fun is not None:
        kwargs["reward_fun"] = reward_fun

    if env_id not in registry:
        register(
            id=env_id,
            entry_point="simglucose.envs:T1DSimGymnaisumEnv",
            max_episode_steps=episode_steps,
            kwargs=kwargs,
        )

    raw_env = gym.make(env_id, disable_env_checker=True)
    return ScalarBasalActionWrapper(raw_env)
