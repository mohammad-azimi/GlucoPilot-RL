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

# Additional scenarios are deliberately kept separate from the original tuning run.
# They are used to check whether a controller generalizes beyond one easy episode.
SCENARIO_MEALS: dict[str, list[tuple[int, int]]] = {
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


def available_scenarios() -> tuple[str, ...]:
    """Return scenario names in stable display/evaluation order."""
    return tuple(SCENARIO_MEALS.keys())


def get_scenario_meals(scenario_name: str) -> list[tuple[int, int]]:
    """Get a copy of the meal schedule for a named experiment scenario."""
    try:
        meals = SCENARIO_MEALS[scenario_name]
    except KeyError as exc:
        valid = ", ".join(available_scenarios())
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
