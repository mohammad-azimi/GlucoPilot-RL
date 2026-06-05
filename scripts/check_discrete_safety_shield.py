"""Sanity-check the discrete residual mapping and safety shield."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucopilot_rl.discrete_env import (  # noqa: E402
    DISCRETE_RESIDUAL_LEVELS,
    NEUTRAL_ACTION_INDEX,
    apply_safety_shield,
)
from glucopilot_rl.rl_env import residual_to_simulator_action  # noqa: E402


def main() -> None:
    print("Discrete safety-shield sanity check passed.")
    print(f"Available residual levels: {DISCRETE_RESIDUAL_LEVELS.tolist()}")
    print(f"Neutral action index: {NEUTRAL_ACTION_INDEX}")
    print(f"Neutral residual maps to simulator action: {residual_to_simulator_action(0.0):.4f}")
    for residual in DISCRETE_RESIDUAL_LEVELS:
        print(f"  residual {float(residual):>5.2f} -> simulator action {residual_to_simulator_action(float(residual)):.4f}")

    examples = [
        (0.50, 50.0, -1.0),
        (0.50, 65.0, 0.5),
        (0.50, 82.0, -3.0),
        (-1.0, 270.0, 1.0),
        (-1.0, 320.0, 0.0),
        (0.25, 130.0, 0.0),
    ]
    print("\nShield examples:")
    for requested, cgm, trend in examples:
        shielded, reason = apply_safety_shield(requested, cgm=cgm, trend=trend)
        print(f"  requested={requested:+.2f}, cgm={cgm:>5.1f}, trend={trend:+.1f} -> {shielded:+.2f} ({reason})")


if __name__ == "__main__":
    main()
