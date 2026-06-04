"""Reproducible meal scenarios used in GlucoPilot-RL experiments."""

from __future__ import annotations

from datetime import datetime

from simglucose.simulation.scenario import CustomScenario

# The default Dexcom sensor exposed by simglucose reports a 3-minute sample time.
SENSOR_SAMPLE_MINUTES = 3
STANDARD_DAY_HOURS = 24
STANDARD_DAY_STEPS = STANDARD_DAY_HOURS * 60 // SENSOR_SAMPLE_MINUTES

# Development scenario used to tune the initial fixed-action baseline.
STANDARD_DAY_MEALS: list[tuple[int, int]] = [
    (7, 45),
    (12, 70),
    (16, 15),
    (18, 80),
    (23, 10),
]

# Frozen comparison scenarios. These are not used to train the PPO policy.
HELD_OUT_SCENARIO_MEALS: dict[str, list[tuple[int, int]]] = {
    "standard-day": STANDARD_DAY_MEALS,
    "high-carb-day": [
        (7, 75),
        (12, 100),
        (16, 25),
        (19, 105),
        (22, 20),
    ],
    "late-dinner": [
        (8, 40),
        (13, 65),
        (17, 15),
        (21, 100),
        (23, 15),
    ],
    "snack-heavy": [
        (7, 35),
        (10, 20),
        (13, 65),
        (16, 25),
        (19, 75),
        (22, 25),
    ],
}

# Training schedules are deliberately separate from the frozen comparison suite.
TRAINING_SCENARIO_MEALS: dict[str, list[tuple[int, int]]] = {
    "train-balanced-a": [(6, 35), (11, 55), (15, 12), (19, 65), (22, 8)],
    "train-balanced-b": [(8, 50), (12, 60), (16, 18), (20, 70)],
    "train-variable-a": [(7, 30), (10, 12), (13, 80), (18, 45), (21, 20)],
    "train-variable-b": [(6, 60), (12, 45), (14, 18), (19, 85), (22, 12)],
}

# Validation schedules are reserved for later iteration decisions only.
VALIDATION_SCENARIO_MEALS: dict[str, list[tuple[int, int]]] = {
    "validation-a": [(7, 40), (12, 85), (17, 20), (20, 55)],
    "validation-b": [(8, 55), (11, 15), (14, 55), (18, 75), (22, 15)],
}

ALL_SCENARIO_MEALS = {
    **HELD_OUT_SCENARIO_MEALS,
    **TRAINING_SCENARIO_MEALS,
    **VALIDATION_SCENARIO_MEALS,
}


def available_scenarios() -> tuple[str, ...]:
    """Return frozen baseline/held-out scenario names in stable display order."""
    return tuple(HELD_OUT_SCENARIO_MEALS.keys())


def available_training_scenarios() -> tuple[str, ...]:
    """Return schedules permitted for PPO training."""
    return tuple(TRAINING_SCENARIO_MEALS.keys())


def available_validation_scenarios() -> tuple[str, ...]:
    """Return schedules reserved for later validation."""
    return tuple(VALIDATION_SCENARIO_MEALS.keys())


def get_scenario_meals(scenario_name: str) -> list[tuple[int, int]]:
    """Get a copy of the meal schedule for a named experiment scenario."""
    try:
        meals = ALL_SCENARIO_MEALS[scenario_name]
    except KeyError as exc:
        valid = ", ".join(ALL_SCENARIO_MEALS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from: {valid}.") from exc
    return list(meals)


def make_scenario(scenario_name: str = "standard-day") -> CustomScenario:
    """Return a fresh fixed-meal scenario so experiments are comparable."""
    return CustomScenario(
        start_time=datetime(2018, 1, 1, 0, 0, 0),
        scenario=get_scenario_meals(scenario_name),
    )


def make_standard_day_scenario() -> CustomScenario:
    """Backward-compatible helper for the original development scenario."""
    return make_scenario("standard-day")
