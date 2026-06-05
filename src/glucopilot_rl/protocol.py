"""Experimental protocol constants for development, validation and locked tests.

The project distinguishes three roles:
- development/training cases: used to optimize the PPO policy;
- validation cases: used while iterating on controller design;
- held-out cases: reserved for the final locked comparison only.
"""

from __future__ import annotations

from collections.abc import Iterator

# This native simulator action was selected only on the original development
# case. It is a simulation reference controller, not dosing guidance.
TUNED_FIXED_ACTION = 0.0450
REFERENCE_CASE = ("adult#001", "standard-day", 42)

# Final comparison suite. It was already used once for the Phase 4 plumbing
# check; from Phase 5 onward it should remain closed until a model is locked.
HELD_OUT_PATIENTS = ["adult#002", "adult#003", "adult#004", "adult#005"]
HELD_OUT_SCENARIOS = ["standard-day", "high-carb-day", "late-dinner", "snack-heavy"]

# Domain-randomized PPO training pool. Patients are separated from validation
# and held-out groups so a model cannot memorize the final evaluation subjects.
TRAINING_PATIENTS = [
    "adult#001", "adult#006", "adult#007", "adult#008",
]
TRAINING_SCENARIOS = [
    "train-balanced-a", "train-balanced-b", "train-variable-a", "train-variable-b",
    "train-high-carb-a", "train-late-dinner-a", "train-snack-heavy-a", "train-small-meals-a",
]

# These cases may be used for smoke testing, design corrections and selection
# of a future training run. They are not part of the final locked comparison.
VALIDATION_PATIENTS = ["adult#009", "adult#010"]
VALIDATION_SCENARIOS = ["validation-a", "validation-b", "validation-high-carb", "validation-late-dinner"]


def held_out_cases() -> Iterator[tuple[str, str, int]]:
    """Yield final frozen comparison cases and deterministic simulator seeds."""
    for patient_index, patient_name in enumerate(HELD_OUT_PATIENTS):
        for scenario_index, scenario_name in enumerate(HELD_OUT_SCENARIOS):
            seed = 1000 + patient_index * 100 + scenario_index
            yield patient_name, scenario_name, seed


def validation_cases() -> Iterator[tuple[str, str, int]]:
    """Yield development-time validation cases and deterministic seeds."""
    for patient_index, patient_name in enumerate(VALIDATION_PATIENTS):
        for scenario_index, scenario_name in enumerate(VALIDATION_SCENARIOS):
            seed = 2000 + patient_index * 100 + scenario_index
            yield patient_name, scenario_name, seed
