"""Discrete safety-shielded residual-control environment.

Phase 7 showed that continuous PPO moved away from the reference controller on
validation, even after adding virtual meal context. This phase switches from a
continuous action to a small, interpretable set of residual actions and adds a
hard safety shield that prevents obviously unsafe directions.

All controls are simulator actions for research only, never medical guidance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from simglucose.envs import T1DSimGymnaisumEnv

from .env import ScalarBasalActionWrapper
from .protocol import TRAINING_PATIENTS, TRAINING_SCENARIOS, TUNED_FIXED_ACTION
from .rl_env import (
    EARLY_FAILURE_PENALTY,
    MAX_SIMULATOR_ACTION,
    MIN_SIMULATOR_ACTION,
    REFERENCE_SIMULATOR_ACTION,
    RESIDUAL_EFFORT_PENALTY,
    residual_to_simulator_action,
    safety_shaped_reward,
)
from .scenarios import SENSOR_SAMPLE_MINUTES, STANDARD_DAY_STEPS, get_scenario_meals, make_scenario

DISCRETE_RESIDUAL_LEVELS = np.array(
    [-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 1.0],
    dtype=np.float32,
)
NEUTRAL_ACTION_INDEX = int(np.where(DISCRETE_RESIDUAL_LEVELS == 0.0)[0][0])


def apply_safety_shield(requested_residual: float, *, cgm: float, trend: float) -> tuple[float, str]:
    """Clamp discrete residual choices that are unsafe for the current CGM state.

    The shield is intentionally conservative:
    - when glucose is low or falling, it blocks extra insulin-like action;
    - when glucose is very high and rising, it blocks very low action;
    - otherwise it leaves the requested action unchanged.
    """
    requested = float(np.clip(requested_residual, -1.0, 1.0))
    if cgm < 54.0:
        return -1.0, "force-min-below-54"
    if cgm < 70.0:
        return min(requested, -0.50), "cap-low-below-70"
    if cgm < 85.0 and trend < -2.0:
        return min(requested, -0.25), "cap-falling-low"
    if cgm < 90.0:
        return min(requested, 0.0), "cap-near-low"
    if cgm > 300.0:
        return max(requested, 0.75), "floor-extreme-high"
    if cgm > 250.0:
        return max(requested, 0.50), "floor-very-high"
    if cgm > 180.0 and trend > 2.0:
        return max(requested, 0.25), "floor-rising-high"
    return requested, "unchanged"


class DiscreteShieldedGlucoseControlEnv(gym.Env[np.ndarray, int]):
    """Meal-aware discrete residual environment with a safety shield."""

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
        self._current_cgm = 110.0
        self._current_trend = 0.0
        self._previous_residual_action = 0.0
        self._previous_requested_residual_action = 0.0
        self._previous_simulator_action = REFERENCE_SIMULATOR_ACTION
        self._previous_shield_reason = "reset"
        self.current_patient_name = ""
        self.current_scenario_name = ""
        self.current_simulator_seed = -1

        self.action_space = gym.spaces.Discrete(len(DISCRETE_RESIDUAL_LEVELS))
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.25, -4.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([8.75, 4.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0], dtype=np.float32),
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

    def _meal_context_features(self, hours: float) -> tuple[float, float, float, float]:
        meals = sorted(get_scenario_meals(self.current_scenario_name), key=lambda item: item[0])
        if not meals:
            return 0.0, 1.0, 0.0, 1.0

        next_candidates = [(meal_hour, grams) for meal_hour, grams in meals if meal_hour >= hours]
        if next_candidates:
            next_hour, next_grams = next_candidates[0]
            hours_until_next = next_hour - hours
        else:
            first_hour, next_grams = meals[0]
            hours_until_next = (24.0 - hours) + first_hour

        previous_candidates = [(meal_hour, grams) for meal_hour, grams in meals if meal_hour <= hours]
        if previous_candidates:
            previous_hour, previous_grams = previous_candidates[-1]
            hours_since_previous = hours - previous_hour
        else:
            last_hour, previous_grams = meals[-1]
            hours_since_previous = hours + (24.0 - last_hour)

        return (
            float(np.clip(next_grams / 100.0, 0.0, 1.5)),
            float(np.clip(hours_until_next / 12.0, 0.0, 1.0)),
            float(np.clip(previous_grams / 100.0, 0.0, 1.5)),
            float(np.clip(hours_since_previous / 12.0, 0.0, 1.0)),
        )

    def _features(self, cgm: float) -> np.ndarray:
        cgm_scaled = np.clip((cgm - 125.0) / 100.0, -1.25, 8.75)
        delta = cgm - self._previous_cgm
        delta_scaled = np.clip(delta / 10.0, -4.0, 4.0)
        hours = self._step_count * SENSOR_SAMPLE_MINUTES / 60.0
        angle = 2.0 * np.pi * hours / 24.0
        meal_features = self._meal_context_features(hours)
        features = np.array(
            [
                cgm_scaled,
                delta_scaled,
                np.sin(angle),
                np.cos(angle),
                self._previous_residual_action,
                *meal_features,
            ],
            dtype=np.float32,
        )
        self._current_trend = float(delta)
        self._previous_cgm = float(cgm)
        self._current_cgm = float(cgm)
        return features

    def _augment_info(self, info: dict[str, Any], cgm: float) -> dict[str, Any]:
        augmented = dict(info)
        augmented.update(
            {
                "raw_cgm_mg_dl": float(cgm),
                "patient_name": self.current_patient_name,
                "scenario_name": self.current_scenario_name,
                "simulator_seed": int(self.current_simulator_seed),
                "requested_residual_action": float(self._previous_requested_residual_action),
                "residual_action": float(self._previous_residual_action),
                "basal_action": float(self._previous_simulator_action),
                "reference_basal_action": float(REFERENCE_SIMULATOR_ACTION),
                "shield_reason": self._previous_shield_reason,
            }
        )
        return augmented

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
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
        self._current_cgm = cgm
        self._current_trend = 0.0
        self._previous_residual_action = 0.0
        self._previous_requested_residual_action = 0.0
        self._previous_simulator_action = REFERENCE_SIMULATOR_ACTION
        self._previous_shield_reason = "reset"
        observation = self._features(cgm)
        return observation, self._augment_info(info, cgm)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._inner_env is None:
            raise RuntimeError("Call reset() before step().")

        action_index = int(np.asarray(action).reshape(-1)[0])
        requested_residual = float(DISCRETE_RESIDUAL_LEVELS[action_index])
        shielded_residual, shield_reason = apply_safety_shield(
            requested_residual,
            cgm=self._current_cgm,
            trend=self._current_trend,
        )
        simulator_action = residual_to_simulator_action(shielded_residual)

        self._previous_requested_residual_action = requested_residual
        self._previous_residual_action = shielded_residual
        self._previous_simulator_action = simulator_action
        self._previous_shield_reason = shield_reason

        native_action = np.array([simulator_action], dtype=np.float32)
        raw_observation, reward, terminated, truncated, info = self._inner_env.step(native_action)
        self._step_count += 1
        cgm = float(np.asarray(raw_observation).reshape(-1)[0])
        reward = float(reward) - RESIDUAL_EFFORT_PENALTY * abs(shielded_residual)
        if terminated and not truncated and self._step_count < self.episode_steps:
            reward -= EARLY_FAILURE_PENALTY
        observation = self._features(cgm)
        return observation, reward, bool(terminated), bool(truncated), self._augment_info(info, cgm)

    def close(self) -> None:
        if self._inner_env is not None:
            self._inner_env.close()
            self._inner_env = None
        super().close()
