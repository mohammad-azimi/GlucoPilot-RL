# GlucoPilot-RL

**Reinforcement Learning for Simulated Blood Glucose Control in Virtual Patients**

GlucoPilot-RL is a portfolio-oriented research project that studies adaptive insulin-control policies in a simulated Type 1 Diabetes environment. It uses reproducible baseline experiments, a locked held-out test suite, and a reinforcement-learning policy trained only on separate development episodes.

> **Research and education only.** This repository uses virtual patients in simulation. It is not medical advice, not a clinical tool, and must not be used to make real insulin-dosing decisions.

## Current status

- [x] Validate the simulator locally and export a CGM glucose trace.
- [x] Add a corrected 24-hour deterministic meal scenario and safety-first fixed-action baseline search.
- [x] Evaluate the tuned fixed-action baseline on held-out virtual adults and meal scenarios.
- [x] Add the first Stable-Baselines3 PPO training and frozen-test evaluation pipeline.
- [ ] Run the PPO smoke test locally and verify model inference outputs.
- [ ] Train and compare a longer PPO experiment against the frozen baseline.
- [ ] Add a small visual dashboard for portfolio presentation.

## Motivation from the baseline experiment

A constant simulator action selected on `adult#001 / standard-day` scored `100.00%` time in range in its development episode. Once frozen and evaluated without retuning across 16 held-out combinations of other adult virtual patients and meal schedules, its mean time in range dropped to `73.65%`. Its worst episode was `adult#004 / late-dinner`, with only `33.12%` time in range and `66.88%` below range. This observed generalization failure motivates an adaptive policy.

These are simulator-only research measurements; they are not clinical performance claims.

## Experimental protocol

### Development baseline

`scripts/evaluate_fixed_basal.py` searches a fixed-action baseline on only one development case:

```text
adult#001 / standard-day / seed=42
```

The selected action is frozen at `0.0450` for comparison.

### Locked held-out comparison suite

`scripts/evaluate_baseline_generalization.py` evaluates the frozen fixed action on patients `adult#002`–`adult#005` across four fixed meal schedules. These 16 cases were defined before PPO training and must not be used to tune or train the agent.

### PPO training pool

`scripts/train_ppo_agent.py` trains PPO only on a separate randomized development pool:

```text
Patients: adult#001, adult#006, adult#007, adult#008
Meal schedules: train-balanced-a, train-balanced-b, train-variable-a, train-variable-b
```

At each training episode, a patient, a development-only meal schedule, and a simulator seed are sampled. The action is bounded to the simulator search range `0.0000–0.0550`. The observation supplied to the policy contains normalized current CGM, short-term trend, simulated time-of-day features, and the previous action.

The first 5,000-timestep run is a **pipeline smoke test**, not a final trained-performance result.

## Reinforcement-learning environment

The first agent uses:

- **Algorithm:** PPO from Stable-Baselines3
- **Action:** one continuous bounded simulator basal-control value
- **Observation features:** CGM, one-step CGM trend, cyclical time-of-day, previous action
- **Training environment:** randomized development patients and meals only
- **Reward:** an interpretable safety-shaped simulation reward that penalizes low glucose more severely than high glucose
- **Evaluation:** deterministic inference on the locked 16-case test suite, compared with the frozen fixed-action baseline

The simulator's built-in reward measures change in risk; this project introduces a safety-oriented custom reward for the first PPO experiment because severe simulated low glucose dominated the frozen baseline's worst held-out outcome.

## Technology stack

- Python 3.12.6 — tested on Windows 10 for simulator and baseline experiments
- simglucose 0.2.11
- Gymnasium 0.29.1
- Stable-Baselines3 2.8.0 and PyTorch — added for PPO training
- NumPy, Pandas and Matplotlib

## Setup on Windows

```bat
cd "C:\Users\Rayan Service\Desktop\Projects\GlucoPilot-RL"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel --disable-pip-version-check
python -m pip install -r requirements.txt --disable-pip-version-check
python scripts\check_environment.py
python scripts\evaluate_fixed_basal.py
python scripts\evaluate_baseline_generalization.py
```

If a system SOCKS proxy is active and `pip` reports missing SOCKS support, temporarily disable the proxy during dependency installation or install `PySocks` locally first.

## Phase 4: run the PPO pipeline smoke test

After copying the Phase 4 update into the project, install the newly added dependency and train a short first model:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt --disable-pip-version-check
python scripts\train_ppo_agent.py --timesteps 5000 --model-name ppo_smoke_model
python scripts\evaluate_ppo_generalization.py --model models\ppo_smoke_model.zip
```

Installing Stable-Baselines3 may take several minutes because it installs PyTorch. The 5,000-step model exists to confirm that training, saving, loading and frozen-test inference all work on the local Windows setup; its score must not be presented as the final project result.

## Generated outputs

After the environment check:

```text
outputs/environment_check.csv
outputs/environment_check_glucose_trace.png
```

After the fixed-action baseline evaluation:

```text
outputs/baseline/fixed_basal_summary.csv
outputs/baseline/best_fixed_basal_trace.csv
outputs/baseline/fixed_basal_comparison.png
outputs/baseline/best_fixed_basal_trace.png
outputs/baseline/traces/*.csv
```

After the fixed-action held-out generalization evaluation:

```text
outputs/generalization/fixed_action_generalization_summary.csv
outputs/generalization/held_out_time_in_range_heatmap.png
outputs/generalization/held_out_mean_risk_heatmap.png
outputs/generalization/worst_held_out_trace.png
outputs/generalization/traces/*.csv
```

After PPO training and evaluation:

```text
models/ppo_smoke_model.zip
outputs/training/ppo_smoke_model_monitor.monitor.csv
outputs/ppo_evaluation/ppo_vs_fixed_held_out_summary.csv
outputs/ppo_evaluation/ppo_held_out_time_in_range_heatmap.png
outputs/ppo_evaluation/ppo_minus_fixed_tir_delta_heatmap.png
outputs/ppo_evaluation/worst_held_out_ppo_trace.png
outputs/ppo_evaluation/traces/*.csv
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
│   ├── train_ppo_agent.py
│   └── evaluate_ppo_generalization.py
├── src/
│   └── glucopilot_rl/
│       ├── __init__.py
│       ├── env.py
│       ├── experiment.py
│       ├── metrics.py
│       ├── protocol.py
│       ├── rl_env.py
│       ├── rl_experiment.py
│       └── scenarios.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Technical notes

The simglucose Gymnasium adapter exposes a one-value vector action space, while its internal legacy simulator consumes a scalar basal value. `src/glucopilot_rl/env.py` contains an action wrapper that normalizes this interface before it reaches the simulator.

The locked-test split prevents an inflated claim: PPO training does not see patients `adult#002`–`adult#005` or the frozen meal schedules used in the comparison suite. Any later improvement must be measured against the already-frozen fixed-action outputs on those same cases.
