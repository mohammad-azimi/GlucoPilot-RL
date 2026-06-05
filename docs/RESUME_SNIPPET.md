# Resume and Portfolio Snippet

## Resume bullet

**GlucoPilot-RL — Safety-Shielded Reinforcement Learning for Simulated Blood Glucose Control**  
Built a research simulation for virtual Type 1 Diabetes blood glucose control using `simglucose`, reinforcement learning, and validation-only model selection. Developed fixed-action baselines, PPO experiments, residual policy variants, a discrete DQN controller, and a rule-based safety shield. The locked DQN improved final held-out mean time-in-range from 73.65% to 76.65% and reduced mean simulated risk from 7.5494 to 5.5441. Research simulation only; not for medical use.

## Short portfolio description

GlucoPilot-RL is a reinforcement learning research project for simulated blood glucose control in virtual Type 1 Diabetes patients. The project compares fixed-action control, PPO variants, and a final safety-shielded discrete DQN controller. The final locked model was evaluated once on held-out virtual patients and improved mean time in range by +3.01 percentage points while reducing mean simulated risk by 2.0053. The project emphasizes validation discipline, safety-aware design, and honest reporting of limitations.

## GitHub repository description

Safety-shielded reinforcement learning for simulated blood glucose control with virtual Type 1 Diabetes patients.

## Suggested GitHub topics

```text
reinforcement-learning
dqn
ppo
healthcare-ai
medical-simulation
simglucose
gymnasium
stable-baselines3
python
safety-ai
```

## LinkedIn project post draft

I built **GlucoPilot-RL**, a reinforcement learning research simulation for virtual blood glucose control.

The project started with a fixed-action baseline, then tested multiple RL designs including continuous PPO, residual PPO, meal-aware observations, and finally a safety-shielded discrete DQN controller.

The final locked DQN model was evaluated once on held-out virtual patients. Compared with the fixed baseline, it improved mean time in range from 73.65% to 76.65% and reduced mean simulated risk from 7.5494 to 5.5441.

The most valuable part of this project was not only the final result, but the research workflow: failed experiments, validation-only model selection, safety-shield design, and honest reporting of limitations.

Important note: this is only a research simulation using virtual patients, not a medical tool or treatment recommendation.
