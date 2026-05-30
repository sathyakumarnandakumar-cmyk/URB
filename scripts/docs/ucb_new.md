# UCB — Upper Confidence Bound for AV Route Choice

Tabular UCB (Upper Confidence Bound) algorithm adapted for the URB experiment framework. This script assigns a UCB agent to each autonomous vehicle (AV) after the human-learning mutation phase.

## Algorithm

UCB balances exploration and exploitation using the formula:

```
action = argmax [ Q(s, a) + β · √( ln(t) / N(s, a) ) ]
```

| Symbol | Meaning |
|--------|---------|
| `Q(s,a)` | Estimated value of action `a` in state `s` |
| `β` | Exploration bonus coefficient |
| `t` | Global step counter |
| `N(s,a)` | Visit count for state-action pair |

The continuous observations from `env.observation_space()` are discretised (integer-rounding) into hashable keys for a dictionary-backed Q-table.

## Files

| File | Description |
|------|-------------|
| `scripts/ucb_new.py` | Main experiment script (follows `iql.py` topology) |
| `config/algo_config/ucb/config1.json` | Default hyperparameters |
| `config/algo_config/ucb/test.json` | Short test config (10 training eps) |

## Hyperparameters (`algo_config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_eps` | 4000 | Number of AV training episodes |
| `alpha` | 0.1 | Q-value learning rate (step size) |
| `beta` | 3.0 | UCB exploration bonus coefficient |
| `update_every` | 1 | Learning frequency (no-op for UCB, kept for loop compatibility) |

## Usage

```bash
# Activate the project environment
source urbenv/bin/activate

# Test run (quick validation)
python scripts/ucb_new.py \
  --id ucb_test \
  --alg-conf test \
  --env-conf config1 \
  --task-conf test \
  --net ingolstadt_custom \
  --env-seed 42

# Full experiment
python scripts/ucb_new.py \
  --id ucb_ingolstadt_custom \
  --alg-conf config1 \
  --env-conf config1 \
  --task-conf config1 \
  --net ingolstadt_custom \
  --env-seed 42
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--id` | ✅ | — | Experiment identifier (results saved to `results/<id>/`) |
| `--alg-conf` | ✅ | — | Algorithm config name (from `config/algo_config/ucb/`) |
| `--env-conf` | — | `config1` | Environment config name |
| `--task-conf` | ✅ | — | Task config name |
| `--net` | ✅ | — | Traffic network name |
| `--env-seed` | — | `42` | Random seed |

## Pipeline

1. **Human learning** — Humans stabilise route choices over `human_learning_episodes`
2. **Mutation** — A fraction of humans are replaced by AVs
3. **AV training** — Each AV uses UCB to learn optimal routes via `act()` → `push(reward)` → online Q-update
4. **Testing** — Exploration disabled (`β=0`), AVs act greedily
5. **Finalization** — Plots, loss CSV (`|ΔQ|` per update), SUMO cleanup, metrics analysis

## Design Notes

- **Dictionary-backed Q-tables** (`defaultdict`) handle arbitrary observation spaces without pre-allocation
- **`push(reward)`** performs the Q-update immediately (no replay buffer); `learn()` is a no-op
- **Testing phase** sets `β=0` to disable exploration, analogous to `ε=0` in DQN
- **Loss tracking** records `|ΔQ|` per update as a convergence proxy
