"""Training and evaluation environments for the first PPO policy.

This module intentionally keeps the locked held-out suite separate from the
randomized development pool. All actions and results are simulation artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from simglucose.envs import T1DSimGymnaisumEnv

from .env import ScalarBasalActionWrapper
from .protocol import TRAINING_PATIENTS, TRAINING_SCENARIOS
from .scenarios import SENSOR_SAMPLE_MINUTES, STANDARD_DAY_STEPS, make_scenario

# PPO exploration is restricted to the experimental range already inspected in
# the baseline sweep. This is only a simulator bound, never dosing guidance.
MAX_SIMULATOR_ACTION = 0.0550
EARLY_FAILURE_PENALTY = 100.0


def safety_shaped_reward(bg_last_hour: Sequence[float]) -> float:
    """Return a reward emphasizing avoidance of low glucose in simulation.

    The built-in simulator reward measures change in risk. For the first agent
    we use a simple interpretable target-range reward with stronger penalties
    for simulated low glucose, because low-glucose failure dominated the locked
    baseline's worst case.
    """
    bg = float(bg_last_hour[-1])
    if bg < 54.0:
        return -12.0 - (54.0 - bg) / 4.0
    if bg < 70.0:
        return -4.0 - (70.0 - bg) / 8.0
    if bg <= 180.0:
        return 1.0 - abs(bg - 110.0) / 140.0
    if bg <= 250.0:
        return -1.0 - (bg - 180.0) / 35.0
    return -5.0 - (bg - 250.0) / 20.0


class AdaptiveGlucoseControlEnv(gym.Env[np.ndarray, np.ndarray]):
    """Feature-based basal-control environment for PPO.

    Each reset creates a fresh one-day simulated episode. During training, the
    episode is sampled from a development-only pool of virtual adults and meal
    schedules. During held-out evaluation, one fixed patient/scenario/seed is
    supplied and no random selection occurs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        patient_names: Sequence[str] | None = None,
        scenario_names: Sequence[str] | None = None,
        episode_steps: int = STANDARD_DAY_STEPS,
        selection_seed: int = 2026,
        fixed_simulator_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.patient_names = tuple(patient_names or TRAINING_PATIENTS)
        self.scenario_names = tuple(scenario_names or TRAINING_SCENARIOS)
        if not self.patient_names or not self.scenario_names:
            raise ValueError("At least one virtual patient and one scenario are required.")

        self.episode_steps = int(episode_steps)
        self.fixed_simulator_seed = fixed_simulator_seed
        self._rng = np.random.default_rng(selection_seed)
        self._inner_env: gym.Env | None = None
        self._step_count = 0
        self._previous_cgm = 110.0
        self._previous_action = 0.0
        self.current_patient_name = ""
        self.current_scenario_name = ""
        self.current_simulator_seed = -1

        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([MAX_SIMULATOR_ACTION], dtype=np.float32),
            dtype=np.float32,
        )
        # Features: normalized CGM, one-step trend, time sin/cos, previous action.
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -20.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1000.0 / 300.0, 20.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _create_inner_env(self) -> gym.Env:
        raw_env = T1DSimGymnaisumEnv(
            patient_name=self.current_patient_name,
            custom_scenario=make_scenario(self.current_scenario_name),
            reward_fun=safety_shaped_reward,
            seed=self.current_simulator_seed,
        )
        timed_env = gym.wrappers.TimeLimit(raw_env, max_episode_steps=self.episode_steps)
        return ScalarBasalActionWrapper(timed_env)

    def _features(self, cgm: float) -> np.ndarray:
        delta = (cgm - self._previous_cgm) / 50.0
        hours = self._step_count * SENSOR_SAMPLE_MINUTES / 60.0
        angle = 2.0 * np.pi * hours / 24.0
        features = np.array(
            [
                cgm / 300.0,
                np.clip(delta, -20.0, 20.0),
                np.sin(angle),
                np.cos(angle),
                self._previous_action / MAX_SIMULATOR_ACTION,
            ],
            dtype=np.float32,
        )
        self._previous_cgm = cgm
        return features

    def _augment_info(self, info: dict[str, Any], cgm: float) -> dict[str, Any]:
        augmented = dict(info)
        augmented.update(
            {
                "raw_cgm_mg_dl": float(cgm),
                "patient_name": self.current_patient_name,
                "scenario_name": self.current_scenario_name,
                "simulator_seed": int(self.current_simulator_seed),
            }
        )
        return augmented

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._inner_env is not None:
            self._inner_env.close()

        self.current_patient_name = str(self._rng.choice(self.patient_names))
        self.current_scenario_name = str(self._rng.choice(self.scenario_names))
        self.current_simulator_seed = (
            int(self.fixed_simulator_seed)
            if self.fixed_simulator_seed is not None
            else int(self._rng.integers(1, 2**31 - 1))
        )
        self._inner_env = self._create_inner_env()
        raw_observation, info = self._inner_env.reset()
        cgm = float(np.asarray(raw_observation).reshape(-1)[0])
        self._step_count = 0
        self._previous_cgm = cgm
        self._previous_action = 0.0
        observation = self._features(cgm)
        return observation, self._augment_info(info, cgm)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._inner_env is None:
            raise RuntimeError("Call reset() before step().")
        bounded_action = np.clip(np.asarray(action, dtype=np.float32), 0.0, MAX_SIMULATOR_ACTION)
        self._previous_action = float(bounded_action.reshape(-1)[0])
        raw_observation, reward, terminated, truncated, info = self._inner_env.step(bounded_action)
        self._step_count += 1
        cgm = float(np.asarray(raw_observation).reshape(-1)[0])
        if terminated and not truncated and self._step_count < self.episode_steps:
            reward = float(reward) - EARLY_FAILURE_PENALTY
        observation = self._features(cgm)
        info = self._augment_info(info, cgm)
        info["basal_action"] = self._previous_action
        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._inner_env is not None:
            self._inner_env.close()
            self._inner_env = None
        super().close()
