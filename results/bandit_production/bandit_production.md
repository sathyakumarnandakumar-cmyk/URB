# Bandit Production Run Summary (`bandit_production`)

This directory contains the results of the 500-episode production run for the Contextual Bandit REINFORCE algorithm on the Ingolstadt custom network.

## Run Configuration
This run represents **Combination #1** from the hyperparameter sweep space.

### General Setup
- **Algorithm:** Bandit REINFORCE
- **Network:** `ingolstadt_custom`
- **Environment Seed:** `42`
- **Total Episodes:** 800 (`200` human + `500` training + `100` test)

### Hyperparameters
- **Learning Rate (`lr`):** `3e-4` (0.0003)
- **Entropy Coefficient:** `0.05`
- **Softmax Temperature:** `1.0`
- **Baseline Alpha (EMA):** `0.05`
- **Hidden Layer Size:** `64`

## Expected Output
Once the run finishes, you should see the following key files in this directory:
- `losses/losses.csv`: Tracks the REINFORCE loss, total fleet reward (negative mean travel time), and mean entropy per episode.
- `metrics/BenchmarkMetrics.csv`: Contains the final performance metrics (`t_test`, `t_CAV`, etc.) evaluating how well the trained Bandit policy decongested the network compared to the human baseline.
- `plots/`: Contains visualizations of the training curves and traffic distributions.
