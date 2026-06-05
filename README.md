# GlucoPilot-RL

**Safety-shielded reinforcement learning for simulated blood glucose control with virtual Type 1 Diabetes patients.**

GlucoPilot-RL is a research and portfolio project that explores reinforcement learning for simulated blood glucose control. It compares a fixed-action baseline, continuous PPO variants, meal-aware PPO, and a final safety-shielded discrete DQN controller inside the `simglucose` virtual-patient simulator.

> **Medical safety notice**  
> This project is for research and educational simulation only. It is **not** a medical device, not a treatment recommendation, and must not be used for real insulin dosing or clinical decision-making.

---

## Final held-out result

The final DQN checkpoint was selected using validation results, locked, and then evaluated once on a final held-out suite of 16 unseen virtual-patient/scenario combinations.

| Metric | Fixed baseline | Locked shielded DQN | Change |
|---|---:|---:|---:|
| Mean time in range | `73.65%` | `76.65%` | `+3.01` percentage points |
| Mean simulated risk | `7.5494` | `5.5441` | `-2.0053` |
| Worst below-range exposure | `66.88%` | `43.75%` | Improved |
| Worst very-low exposure | `55.62%` | `27.08%` | Improved |
| Mean safety-shield intervention rate | `—` | `28.22%` | `—` |

The locked shielded DQN improved average time in range and reduced mean simulated risk on the final held-out suite. It also reduced the worst low-glucose exposure compared with the fixed baseline.

The result is promising, but the project should still be presented as a **research simulation**, not as a deployable medical controller.

---

## Final visualizations

### Final DQN time in range

![Final held-out time in range](docs/assets/final_time_in_range_heatmap.png)

### DQN minus fixed baseline: time-in-range delta

![Final held-out TIR delta](docs/assets/final_tir_delta_heatmap.png)

### DQN minus fixed baseline: mean-risk delta

![Final held-out risk delta](docs/assets/final_risk_delta_heatmap.png)

### Worst final held-out DQN episode

![Worst final held-out trace](docs/assets/final_worst_case_trace.png)

---

## Why this project is interesting

This repository documents a full experimental workflow instead of only showing a final model.

The project went through:

1. A reproducible fixed-action baseline.
2. Generalization testing across unseen virtual patients.
3. Continuous PPO experiments.
4. Residual PPO redesign.
5. Meal-aware PPO observations.
6. A discrete residual action space.
7. A rule-based safety shield.
8. DQN training with validation-only checkpoint selection.
9. A locked final model.
10. One-time final held-out evaluation.

This makes the project suitable for a machine learning portfolio because it shows experimentation, failure analysis, model selection discipline, safety-aware design, and honest reporting of limitations.

---

## Method overview

### Simulator

- Environment: `simglucose`
- Task: simulated 24-hour blood glucose control
- Patient type: adult virtual Type 1 Diabetes patients
- Observation interval: 3 simulated minutes
- Evaluation: validation suite and final held-out suite

### Final controller

The final controller uses:

- Discrete DQN
- Residual action design
- Safety-shielded action correction
- Validation-only checkpoint selection
- A locked model before final held-out evaluation

The final action space is a set of residual choices around a tuned fixed reference action:

```text
[-1.00, -0.75, -0.50, -0.25, 0.00, +0.25, +0.50, +1.00]
```

The neutral action reproduces the fixed-action baseline. The safety shield can override clearly unsafe choices, such as increasing insulin-like action when simulated glucose is already low.

---

## Validation before final testing

Before opening the final held-out suite, the DQN checkpoint at 10,240 timesteps was selected from validation results.

| Validation metric | Neutral shield | DQN checkpoint 10,240 |
|---|---:|---:|
| Mean time in range | `75.23%` | `79.79%` |
| Mean TIR delta | `-0.03` pp | `+4.53` pp |
| Mean simulated risk | `5.0101` | `3.7104` |
| Mean risk delta | `+0.0044` | `-1.2952` |
| Max below range | `2.71%` | `5.00%` |
| Max very low | `0.00%` | `0.00%` |

The validation result showed a trade-off: the DQN improved average time in range and mean risk, but slightly increased below-range exposure in some validation cases. This is why the final report avoids medical claims.

![DQN validation TIR by checkpoint](docs/assets/dqn_validation_tir_by_checkpoint.png)

![DQN validation risk by checkpoint](docs/assets/dqn_validation_risk_by_checkpoint.png)

---

## Repository structure

```text
GlucoPilot-RL/
├── docs/
│   ├── assets/
│   ├── results/
│   ├── PROJECT_REPORT.md
│   └── RESUME_SNIPPET.md
├── models/
├── outputs/
├── scripts/
├── src/
│   └── glucopilot_rl/
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

Create and activate a virtual environment:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## Reproduce the workflow

Run the environment check:

```bat
python scripts\check_environment.py
```

Evaluate the fixed baseline:

```bat
python scripts\evaluate_fixed_basal.py
```

Evaluate baseline generalization:

```bat
python scripts\evaluate_baseline_generalization.py
```

Run the discrete safety-shield validation:

```bat
python scripts\check_discrete_safety_shield.py
python scripts\evaluate_discrete_shield_validation.py
```

Train the shielded discrete DQN with validation checkpoint selection:

```bat
python scripts\train_dqn_discrete_with_validation_selection.py --total-timesteps 51200 --checkpoint-every 10240 --run-name dqn_discrete_shield_selection_51k
```

Lock the selected checkpoint:

```bat
python scripts\lock_dqn_checkpoint.py --source models\dqn_discrete_shield_selection_51k\dqn_discrete_shield_selection_51k_step-010240.zip --locked-name dqn_discrete_shield_locked_010240
```

Run the final held-out evaluation once:

```bat
python scripts\evaluate_locked_dqn_final.py --model models\dqn_discrete_shield_locked_010240.zip --run-name dqn_discrete_shield_locked_010240
```

---

## Key outputs

Important final artifacts are stored in:

```text
docs/assets/
docs/results/
docs/PROJECT_REPORT.md
docs/RESUME_SNIPPET.md
```

Generated local training artifacts are ignored by Git:

```text
models/
outputs/
```

---

## Limitations

- This is a simulation-only project.
- The policy is not clinically validated.
- The reward function and safety shield are manually designed.
- The final worst-case trace still contains unsafe simulated glucose behavior.
- The model must not be used outside the virtual-patient simulator.
- No real patient data is used.

---

## Portfolio summary

**GlucoPilot-RL** is a safety-aware reinforcement learning simulation for virtual blood glucose control. The final shielded DQN improved final held-out mean time in range from `73.65%` to `76.65%` and reduced mean simulated risk from `7.5494` to `5.5441`, while still being clearly documented as a research-only simulation.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
