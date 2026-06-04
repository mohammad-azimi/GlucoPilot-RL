"""Reproducible meal scenarios used in GlucoPilot-RL experiments."""

from __future__ import annotations

from datetime import datetime

from simglucose.simulation.scenario import CustomScenario

# A deterministic 24-hour meal schedule used only for in-silico evaluation.
STANDARD_DAY_MEALS: list[tuple[int, int]] = [
    (7, 45),
    (12, 70),
    (16, 15),
    (18, 80),
    (23, 10),
]


def make_standard_day_scenario() -> CustomScenario:
    """Return a fresh fixed-meal scenario so experiments are comparable."""
    return CustomScenario(
        start_time=datetime(2018, 1, 1, 0, 0, 0),
        scenario=STANDARD_DAY_MEALS,
    )
