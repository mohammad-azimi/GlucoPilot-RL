# GlucoPilot-RL

**Reinforcement Learning for Simulated Blood Glucose Control in Virtual Patients**

GlucoPilot-RL is a portfolio-oriented research project that studies adaptive insulin-control policies in a simulated Type 1 Diabetes environment. The project begins with reproducible virtual-patient experiments and baseline evaluation, then introduces reinforcement learning agents and compares them using glucose-safety metrics.

> **Research and education only.** This repository uses virtual patients in simulation. It is not medical advice, not a clinical tool, and must not be used to make real insulin-dosing decisions.

## Current status

- [x] Validate the simulator locally and export a CGM glucose trace.
- [x] Add a corrected 24-hour deterministic meal scenario and safety-first fixed-action baseline search.
- [ ] Train a reinforcement-learning agent and compare it against the baseline.
- [ ] Evaluate across several virtual adults and scenario seeds.
- [ ] Add a small visual dashboard for portfolio presentation.

## Experiments

### 1. Environment check

`scripts/check_environment.py` runs a short non-learning episode on virtual patient `adult#001` and exports a CGM trace. It verifies that the simglucose and Gymnasium adapter work correctly on the local machine.

### 2. Fixed-action baseline search

`scripts/evaluate_fixed_basal.py` runs the same virtual adult patient under a deterministic 24-hour meal scenario. The default sensor interval reported by the simulator is 3 minutes, so a complete day is evaluated over 480 steps. It evaluates 13 constant native simulator actions and records:

- percentage of time in the evaluation target range of 70–180 mg/dL;
- percentage of time below and above that range;
- very-low and very-high glucose percentages;
- mean simulated risk and cumulative reward.

The best fixed-action baseline is selected with a safety-first ordering: minimize very-low and below-range glucose first, then minimize simulated mean risk and above-range time. This baseline becomes the comparison point for the future reinforcement-learning policy.

## Technology stack

- Python 3.12.6 — tested on Windows 10
- simglucose 0.2.11
- Gymnasium 0.29.1
- NumPy, Pandas and Matplotlib
- Stable-Baselines3 — planned for the reinforcement-learning milestone

## Setup on Windows

```bat
cd "C:\Users\Rayan Service\Desktop\Projects\GlucoPilot-RL"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel --disable-pip-version-check
python -m pip install -r requirements.txt --disable-pip-version-check
python scripts\check_environment.py
python scripts\evaluate_fixed_basal.py
```

If a system SOCKS proxy is active and `pip` reports missing SOCKS support, temporarily disable the proxy during dependency installation or install `PySocks` locally first.

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

## Repository structure

```text
GlucoPilot-RL/
├── outputs/
├── scripts/
│   ├── check_environment.py
│   └── evaluate_fixed_basal.py
├── src/
│   └── glucopilot_rl/
│       ├── __init__.py
│       ├── env.py
│       ├── experiment.py
│       ├── metrics.py
│       └── scenarios.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Technical notes

The simglucose Gymnasium adapter exposes a one-value vector action space, while its internal legacy simulator consumes a scalar action. `src/glucopilot_rl/env.py` contains an action wrapper that normalizes this interface before reinforcement-learning training is added.

The baseline evaluates native simulator action values under one reproducible in-silico scenario. The simulator seed is passed at environment construction time so that candidate actions are compared under the same sensor and patient randomness. These values and results are research artifacts inside the simulator only and must not be interpreted as real insulin recommendations.
