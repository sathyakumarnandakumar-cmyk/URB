# Bandit REINFORCE Route Choice Algorithm

The Contextual Bandit REINFORCE algorithm has been successfully integrated into the URB experiment framework as a standalone script!

## Algorithm Overview

Unlike independent Q-learning or UCB where each agent has its own "brain", the Bandit approach uses a single **Shared Neural Network** (`RouteScorer`). All Autonomous Vehicles (AVs) query this shared network, passing in 8 features about their specific route. The network outputs scores, which are converted to probabilities via softmax, and actions are sampled stochastically.

At the end of the episode, the entire network is updated simultaneously via REINFORCE using a **shared global reward** (the negative mean travel time of the entire AV fleet).

### The 8 Features
1. `free_flow_time` (Static)
2. `length_km` (Static - Mocked to 0.0 for compatibility)
3. `motorway_share` (Static - Mocked to 0.0 for compatibility)
4. `n_left_turns` (Static - Mocked to 0.0 for compatibility)
5. `mean_speed_ratio` (Dynamic via TraCI)
6. `mean_occupancy` (Dynamic via TraCI)
7. `mean_cav_link_load` (Fleet Memory tracking AV commitments)
8. `departure_time_normalized` (Temporal Context)

---

## Files

| File | Description |
|------|-------------|
| `scripts/bandit_reinforce.py` | Main experiment script |
| `config/algo_config/bandit_reinforce/config1.json` | Default hyperparameters |
| `config/algo_config/bandit_reinforce/test.json` | Short test config (10 training eps) |

## Hyperparameters (`algo_config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_eps` | 500 | Number of episodes for training |
| `lr` | 3e-4 | Learning rate for the Adam optimizer |
| `entropy_coeff` | 0.05 | Entropy bonus to prevent premature convergence (mode collapse) |
| `temperature` | 1.0 | Softmax temperature for stochastic sampling |
| `baseline_alpha` | 0.05 | Smoothing factor for the Exponential Moving Average baseline |
| `hidden_size` | 64 | Number of neurons in the hidden layers of the MLP |

---

## How to run it

### Quick Test Run (10 episodes)
```bash
python scripts/bandit_reinforce.py \
  --id bandit_test \
  --alg-conf test \
  --env-conf config1 \
  --task-conf test \
  --net ingolstadt_custom \
  --env-seed 42
```

### Full Production Run (500 episodes)
```bash
# Using tmux to run in the background
tmux new -s bandit_run

source urbenv/bin/activate
export SUMO_HOME=/home/sathyakumarnandakumar/URB/urbenv/lib/python3.12/site-packages/sumo

python scripts/bandit_reinforce.py \
  --id bandit_production \
  --alg-conf config1 \
  --env-conf config1 \
  --task-conf config1 \
  --net ingolstadt_custom \
  --env-seed 42

# Press Ctrl+B then D to detach
```

## Loss Tracking
The script tracks several metrics in `losses.csv`:
- `loss`: The final REINFORCE policy gradient loss.
- `reward`: The negative mean travel time of the fleet.
- `ema_reward`: The exponential moving average used as the baseline advantage.
- `entropy`: The mean entropy of the stochastic policy across all agents.
