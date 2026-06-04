"""Residual-control PPO environments for simulated glucose experiments.

Phase 4 verified that PPO can train/save/load, but its native action space was
``[0, 0.055]``. A newly initialized deterministic policy therefore acted close
to zero simulator action and produced avoidable hyperglycaemia during the smoke
test. This module changes the controller to a residual policy:

- PPO emits a normalized adjustment in ``[-1, 1]``;
- adjustment ``0`` reproduces the frozen fixed-action reference controller;
- negative adjustments can reduce simulated action down to ``0``;
- positive adjustments can increase it up to the inspected bound ``0.055``.

All values are simulator controls for research only, never medical guidance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from simglucose.envs import T1DSimGymnaisumEnv

from .env import ScalarBasalActionWrapper
from .protocol import TRAINING_PATIENTS, TRAINING_SCENARIOS, TUNED_FIXED_ACTION
from .scenarios import SENSOR_SAMPLE_MINUTES, STANDARD_DAY_STEPS, make_scenario

MIN_SIMULATOR_ACTION = 0.0
MAX_SIMULATOR_ACTION = 0.0550
REFERENCE_SIMULATOR_ACTION = TUNED_FIXED_ACTION
EARLY_FAILURE_PENALTY = 100.0
RESIDUAL_EFFORT_PENALTY = 0.02


def residual_to_simulator_action(residual_action: float | np.ndarray) -> float:
    """Map a normalized residual control signal to a native simulator action.

    Zero is intentionally the frozen fixed-action reference. The asymmetric map
    preserves a larger downward adjustment range because excessive fixed action
    caused severe low-glucose failure in one held-out virtual patient.
    """
    residual = float(np.clip(np.asarray(residual_action).reshape(-1)[0], -1.0, 1.0))
    if residual >= 0.0:
        native = REFERENCE_SIMULATOR_ACTION + residual * (
            MAX_SIMULATOR_ACTION - REFERENCE_SIMULATOR_ACTION
        )
    else:
        native = REFERENCE_SIMULATOR_ACTION + residual * (
            REFERENCE_SIMULATOR_ACTION - MIN_SIMULATOR_ACTION
        )
    return float(np.clip(native, MIN_SIMULATOR_ACTION, MAX_SIMULATOR_ACTION))


def safety_shaped_reward(bg_last_hour: Sequence[float]) -> float:
    """Return an interpretable simulation reward emphasizing low-CGM safety."""
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


class ResidualGlucoseControlEnv(gym.Env[np.ndarray, np.ndarray]):
    """Feature-based residual-control environment for PPO.

    A fresh one-day episode is sampled at every reset. Training uses only the
    development pool; validation and final evaluation provide fixed patient,
    schedule and simulator-seed combinations explicitly.
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
        self._previous_residual_action = 0.0
        self._previous_simulator_action = REFERENCE_SIMULATOR_ACTION
        self.current_patient_name = ""
        self.current_scenario_name = ""
        self.current_simulator_seed = -1

        # Symmetric normalized control is the recommended PPO interface.
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Features: scaled CGM, short trend, time sin/cos and prior residual.
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.25, -4.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([8.75, 4.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        cgm_scaled = np.clip((cgm - 125.0) / 100.0, -1.25, 8.75)
        delta_scaled = np.clip((cgm - self._previous_cgm) / 10.0, -4.0, 4.0)
        hours = self._step_count * SENSOR_SAMPLE_MINUTES / 60.0
        angle = 2.0 * np.pi * hours / 24.0
        features = np.array(
            [
                cgm_scaled,
                delta_scaled,
                np.sin(angle),
                np.cos(angle),
                self._previous_residual_action,
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
                "residual_action": float(self._previous_residual_action),
                "basal_action": float(self._previous_simulator_action),
                "reference_basal_action": float(REFERENCE_SIMULATOR_ACTION),
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
        self._previous_residual_action = 0.0
        self._previous_simulator_action = REFERENCE_SIMULATOR_ACTION
        observation = self._features(cgm)
        return observation, self._augment_info(info, cgm)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._inner_env is None:
            raise RuntimeError("Call reset() before step().")
        residual = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        simulator_action = residual_to_simulator_action(residual)
        self._previous_residual_action = residual
        self._previous_simulator_action = simulator_action
        native_action = np.array([simulator_action], dtype=np.float32)
        raw_observation, reward, terminated, truncated, info = self._inner_env.step(native_action)
        self._step_count += 1
        cgm = float(np.asarray(raw_observation).reshape(-1)[0])
        reward = float(reward) - RESIDUAL_EFFORT_PENALTY * abs(residual)
        if terminated and not truncated and self._step_count < self.episode_steps:
            reward -= EARLY_FAILURE_PENALTY
        observation = self._features(cgm)
        return observation, reward, bool(terminated), bool(truncated), self._augment_info(info, cgm)

    def close(self) -> None:
        if self._inner_env is not None:
            self._inner_env.close()
            self._inner_env = None
        super().close()


# Backward-compatible import alias for source files; old native-action PPO model
# files must not be evaluated with this residual-action environment.
AdaptiveGlucoseControlEnv = ResidualGlucoseControlEnv
