# GlucoPilot-RL

**Residual Reinforcement Learning for Simulated Blood Glucose Control in Virtual Patients**

GlucoPilot-RL is a portfolio-oriented research project that studies adaptive simulated insulin-control policies in a Type 1 Diabetes virtual-patient environment. It uses reproducible baselines, separate training and validation pools, and a final comparison suite.

> **Research and education only.** This repository uses virtual patients in simulation. It is not medical advice, not a clinical tool, and must never be used to make real insulin-dosing decisions.

## Current status

- [x] Validate `simglucose` locally on Windows with Python 3.12.6 and export a CGM trace.
- [x] Build a corrected 24-hour fixed-action baseline search.
- [x] Demonstrate the fixed controller's failure to generalize across virtual patients and meal schedules.
- [x] Verify the first PPO training/save/load/evaluation pipeline with a short smoke model.
- [x] Redesign PPO as a normalized **residual policy** anchored to the fixed-action baseline.
- [ ] Run the residual-policy smoke test on the validation suite.
- [ ] Train and select a longer model using validation only.
- [ ] Run one final locked comparison and prepare portfolio figures.

## Why adaptive control is needed

The frozen constant-action controller was selected only on `adult#001 / standard-day / seed=42`, where it achieved `100.00%` simulated time in range. Applied unchanged to 16 held-out patient-scenario combinations, its mean time in range dropped to `73.65%`, and its worst case fell to `33.12%` time in range with `66.88%` below range.

Phase 4 then confirmed that PPO trains and evaluates end to end. That short smoke policy scored only `35.24%` mean time in range versus the fixed controller's `73.65%`; this is **not a final performance result**. It exposed an action-design issue: a new PPO policy operating directly over `[0.0000, 0.0550]` predicts near zero by default, leading to very high simulated glucose before meaningful learning occurs.

## Phase 5 controller design: residual PPO

The corrected controller no longer asks PPO to output the entire native simulator action. Instead it learns an adjustment around the frozen baseline:

| PPO residual action | Native simulator action |
|---:|---:|
| `-1.0` | `0.0000` |
| `0.0` | `0.0450` — frozen reference |
| `+1.0` | `0.0550` |

This design has three advantages:

1. The PPO action space is symmetric and normalized (`[-1, 1]`).
2. An initially neutral deterministic policy reproduces the known baseline rather than behaving like a near-zero controller.
3. The agent can learn to reduce action strongly for more sensitive virtual patients while increasing it only within the previously inspected simulator bound.

All action values are simulation parameters only and are not dosing recommendations.

## Experimental protocol

### Fixed-action development reference

`scripts/evaluate_fixed_basal.py` searches only:

```text
adult#001 / standard-day / seed=42
```

The chosen reference action is frozen at `0.0450`.

### Training pool

The residual PPO policy trains only on a randomized development pool:

```text
Patients: adult#001, adult#006–adult#008, adult#011–adult#016
Meal schedules: eight development-only schedules
```

### Validation pool

During controller iteration, use only:

```text
Patients: adult#009, adult#010
Meal schedules: validation-a, validation-b, validation-high-carb, validation-late-dinner
```

### Final held-out comparison suite

Patients `adult#002`–`adult#005` and the four original held-out meal schedules are reserved for the final comparison. Phase 4 used them once to verify the original smoke-test inference pipeline; from Phase 5 onward they remain closed until a final residual model is locked.

## Technology stack

- Python 3.12.6 — verified on Windows 10
- `simglucose==0.2.11`
- `gymnasium==0.29.1`
- `stable-baselines3==2.8.0` and PyTorch
- NumPy, Pandas and Matplotlib

## Setup on Windows

```bat
cd "C:\Users\Rayan Service\Desktop\Projects\GlucoPilot-RL"
.venv\Scripts\activate
python -m pip install -r requirements.txt --disable-pip-version-check
```

If a system SOCKS proxy is active and `pip` reports missing SOCKS support, temporarily disable the proxy during dependency installation or install `PySocks` locally first.

## Phase 5 commands

First confirm that the neutral residual policy is identical to the frozen fixed-action controller:

```bat
python scripts\check_residual_controller.py
```

Then run a corrected short smoke training run and evaluate it on **validation**, not held-out cases:

```bat
python scripts\train_ppo_agent.py --timesteps 5000 --model-name ppo_residual_smoke_model
python scripts\evaluate_ppo_validation.py --model models\ppo_residual_smoke_model.zip
```

The previous model file `models\ppo_smoke_model.zip` used the old native-action meaning and must not be evaluated with the Phase 5 residual environment.

## Generated outputs

Existing baseline and held-out baseline outputs remain in `outputs/baseline/` and `outputs/generalization/`.

After Phase 5 residual PPO training and validation:

```text
models/ppo_residual_smoke_model.zip
outputs/training/ppo_residual_smoke_model_monitor.monitor.csv
outputs/training/ppo_residual_smoke_model_training_rewards.png
outputs/validation/residual_ppo_vs_fixed_validation_summary.csv
outputs/validation/ppo_validation_time_in_range_heatmap.png
outputs/validation/ppo_minus_fixed_validation_tir_delta_heatmap.png
outputs/validation/worst_validation_comparison_trace.png
outputs/validation/traces/*.csv
```

## Repository structure

```text
GlucoPilot-RL/
├── models/
├── outputs/
├── scripts/
│   ├── check_environment.py
│   ├── evaluate_fixed_basal.py
│   ├── evaluate_baseline_generalization.py
│   ├── check_residual_controller.py
│   ├── train_ppo_agent.py
│   ├── evaluate_ppo_validation.py
│   └── evaluate_ppo_generalization.py      # final-only gate
├── src/
│   └── glucopilot_rl/
│       ├── env.py
│       ├── experiment.py
│       ├── metrics.py
│       ├── protocol.py
│       ├── rl_env.py
│       ├── rl_experiment.py
│       └── scenarios.py
├── requirements.txt
└── README.md
```
