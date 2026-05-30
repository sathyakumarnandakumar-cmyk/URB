# Bandit Routing for URB/Ingolstadt — Summary

## Problem

Control 40% of ~1000 vehicles (CAVs) in Ingolstadt to minimize their mean
travel time. The remaining 60% (HDVs) follow fixed routes. Each CAV chooses
once at departure from 4 precomputed routes. SUMO microsimulation evaluates
the joint assignment. No gradient through the simulator.

Key properties:
- One decision per agent per episode (bandit-flavoured, not full RL)
- Sequential departures: earlier CAVs affect network state seen by later ones
- ~400 OD pairs, ~1 CAV per OD pair on average
- Heavy link overlap across OD pairs → link-level signals are dense
- Shared reward: minimize mean CAV travel time

---

## Why existing MARL fails

All baselines (IPPO, MAPPO, IQL, QMIX) produce t_CAV > t_pre in Ingolstadt —
worse than if CAVs drove like humans. Root causes:

1. Observation is OD-local only (counts of same-OD earlier agents) — no
   real-time network state, no coordination signal
2. Sequential MDP wrapper is artificial — each agent decides once, not many
   times per episode
3. Shared reward + independent learners → herding onto same route (tragedy
   of the commons)
4. Reward delayed to episode end with no intermediate signal

---

## Proposed approach: Contextual Bandit REINFORCE

### Core idea
One shared MLP scorer that scores routes by their features. All 400 CAVs use
the same network. Stochastic sampling from softmax naturally load-balances
the fleet without explicit coordination.

### Observation (8 features per route, 32 total for 4 routes)

| Feature | Source | Why |
|---|---|---|
| free_flow_travel_time | static (JanuX) | baseline attractiveness |
| total_length_km | static | route cost proxy |
| motorway_share | static | speed quality |
| n_left_turns | static | urban complexity |
| mean_speed_ratio | TraCI at departure | realized HDV congestion |
| mean_occupancy | TraCI at departure | link load |
| mean_cav_link_load | fleet memory (Python counter) | within-episode CAV coordination |
| departure_time_normalized | agent.start_time | temporal context |

Fleet memory is link-level (not OD-level) to handle ~1 CAV per OD pair.

### Policy
```
scorer = MLP(8 → 64 → 64 → 1)          # scores one route
scores = [scorer(φ(r)) for r in routes] # 4 scores
π = softmax(scores / temperature)        # stochastic policy
a ~ π                                    # sample route
```

### Training (REINFORCE)
```
R = -mean_travel_time                    # episode reward
advantage = R - baseline                 # baseline = EMA of R
loss = -(advantage × Σ log π(aᵢ)) - β×entropy
```

Entropy bonus β prevents mode collapse (all agents taking same route).
EMA baseline reduces gradient variance (~10× reduction vs raw R).

---

## Files

| File | Location | Purpose |
|---|---|---|
| `bandit_reinforce.py` | `URB/scripts/` | Core module: features, scorer, trainer |
| `bandit_config1.json` | `URB/config/algo_config/bandit/` | Starting hyperparameters |
| `submit_bandit_sweep.py` | `URB/scripts/` | Submits 54 Slurm jobs |

### Wiring into base_script.py
After `env.mutation(...)`:
```python
from bandit_reinforce import BanditTrainer
trainer = BanditTrainer(env, params, records_folder)
trainer.train(n_episodes=500)
trainer.test(n_episodes=50)
```

---

## Hyperparameter sweep (54 jobs)

```
lr:            [3e-4, 1e-3]
entropy_coeff: [0.01, 0.05, 0.10]
temperature:   [0.5, 1.0, 2.0]
seeds:         [42, 123, 7]
→ 2 × 3 × 3 × 3 = 54 parallel Slurm jobs
```

---

## Expectations and caveats

**Target**: beat Greedy baseline (t_CAV = 4.24s in Ingolstadt).

**Likely episodes to convergence**: 300–500. Run 100 first and check:
- Is baseline (EMA of mean_tt) decreasing? If flat → signal too noisy
- Is entropy decaying slowly? If it collapses fast → increase β

**No global optimality guarantee.** This is a local search over a non-convex
policy space with a black-box simulator. It is a strong heuristic, not a
solver. The combinatorial optimum (4^400 joint assignments) is not tractable.

**Prerequisite check before first run**:
- Does `paths.csv` have an `edges` column with SUMO edge IDs per route?
  (If not: TraCI features fall back to neutral defaults — static + fleet
  memory still work)
- Is `env.simulator.sumo` the correct TraCI handle?
  (If not: set traci_conn=None, graceful fallback)
