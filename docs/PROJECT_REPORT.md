# GlucoPilot-RL Project Report

## Summary

GlucoPilot-RL is a reinforcement learning simulation project for virtual blood glucose control. The goal was to test whether an adaptive controller could outperform a fixed-action baseline in a virtual Type 1 Diabetes environment.

The final locked model was a safety-shielded discrete DQN controller. It was selected using validation-only analysis and then evaluated once on the final held-out suite.

## Final held-out result

| Metric | Fixed baseline | Locked shielded DQN | Change |
|---|---:|---:|---:|
| Mean time in range | 73.65% | 76.65% | +3.01 percentage points |
| Mean simulated risk | 7.5494 | 5.5441 | -2.0053 |
| Worst below-range exposure | 66.88% | 43.75% | improved |
| Worst very-low exposure | 55.62% | 27.08% | improved |
| Mean safety-shield intervention rate | — | 28.22% | — |

## Research progression

1. A fixed-action controller was tuned on a development patient.
2. The fixed controller generalized poorly to some held-out patients.
3. PPO with continuous action control was tested and failed to outperform the baseline.
4. Residual PPO recovered baseline-like behavior but did not pass validation.
5. Meal-aware PPO still failed to improve validation performance.
6. A discrete residual action space and safety shield were introduced.
7. DQN with a safety shield produced a promising validation checkpoint.
8. The selected DQN checkpoint was locked and tested once on final held-out cases.

## Main finding

The final DQN improved average performance over the fixed baseline on the final held-out suite:

- Mean time in range improved by +3.01 percentage points.
- Mean simulated risk changed by -2.0053.
- Worst below-range and very-low exposure were reduced.

## Important limitation

The final worst-case trace still contains unsafe simulated glucose behavior. This project is therefore a research prototype and should not be described as a medical device or a real treatment controller.

## Medical disclaimer

This project is not for real medical use. It is not a clinical recommendation, not an insulin dosing system, and not a validated treatment tool.
