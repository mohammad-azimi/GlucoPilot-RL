"""Evaluation metrics for simulated continuous-glucose traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_LOW = 70.0
TARGET_HIGH = 180.0
VERY_LOW = 54.0
VERY_HIGH = 250.0


def _percentage(mask: pd.Series) -> float:
    return float(mask.mean() * 100.0)


def summarize_episode(trace: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    """Summarize one simulated episode using glucose-safety metrics.

    The target thresholds are evaluation markers for virtual-patient research;
    they are not insulin-dosing instructions for real people.
    """
    if trace.empty:
        raise ValueError("Cannot summarize an empty episode trace.")

    cgm = trace["cgm_mg_dl"].astype(float)
    basal_values = trace["basal_action"].astype(float)
    constant_basal = basal_values.iloc[0] if basal_values.nunique() == 1 else np.nan

    return {
        "controller": label,
        "steps": int(len(trace)),
        "basal_action": float(constant_basal),
        "mean_cgm_mg_dl": float(cgm.mean()),
        "min_cgm_mg_dl": float(cgm.min()),
        "max_cgm_mg_dl": float(cgm.max()),
        "std_cgm_mg_dl": float(cgm.std(ddof=0)),
        "time_in_range_pct": _percentage((cgm >= TARGET_LOW) & (cgm <= TARGET_HIGH)),
        "time_below_range_pct": _percentage(cgm < TARGET_LOW),
        "time_above_range_pct": _percentage(cgm > TARGET_HIGH),
        "time_very_low_pct": _percentage(cgm < VERY_LOW),
        "time_very_high_pct": _percentage(cgm > VERY_HIGH),
        "mean_risk": float(trace["risk"].mean()),
        "total_reward": float(trace["reward"].sum()),
    }
