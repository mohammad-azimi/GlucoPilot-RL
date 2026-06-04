"""Experimental protocol constants for development, training and locked tests."""

from __future__ import annotations

from collections.abc import Iterator

# This fixed action was selected only on the original development reference case.
# It is frozen for comparison and is not a medical dose recommendation.
TUNED_FIXED_ACTION = 0.0450
REFERENCE_CASE = ("adult#001", "standard-day", 42)

# Locked comparison suite created before RL training.
# Do not use these virtual patients or scenario schedules for training/tuning.
HELD_OUT_PATIENTS = ["adult#002", "adult#003", "adult#004", "adult#005"]
HELD_OUT_SCENARIOS = ["standard-day", "high-carb-day", "late-dinner", "snack-heavy"]

# Development pool for domain-randomized PPO training. These are intentionally
# separate from the locked held-out virtual patients and meal schedules.
TRAINING_PATIENTS = ["adult#001", "adult#006", "adult#007", "adult#008"]
TRAINING_SCENARIOS = [
    "train-balanced-a",
    "train-balanced-b",
    "train-variable-a",
    "train-variable-b",
]

# Reserved for later hyperparameter decisions; not part of the locked final test.
VALIDATION_PATIENTS = ["adult#009", "adult#010"]
VALIDATION_SCENARIOS = ["validation-a", "validation-b"]


def held_out_cases() -> Iterator[tuple[str, str, int]]:
    """Yield the frozen held-out cases and their original simulator seeds."""
    for patient_index, patient_name in enumerate(HELD_OUT_PATIENTS):
        for scenario_index, scenario_name in enumerate(HELD_OUT_SCENARIOS):
            seed = 1000 + patient_index * 100 + scenario_index
            yield patient_name, scenario_name, seed
