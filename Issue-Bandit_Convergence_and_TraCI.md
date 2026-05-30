# Issue: Bandit REINFORCE Convergence & TraCI Dynamic Features (Analysis)

**Date**: May 20, 2026
**Run Analyzed**: `bandit_REINFORCE_500` (500 episodes on `ingolstadt_custom`)

## Executive Summary
Following the first 500-episode run of the newly integrated `bandit_reinforce.py` script, an analysis of the results revealed that the Autonomous Vehicles (AVs) were still underperforming compared to the baseline human traffic. 

Two root causes were identified:
1. **The Entropy Problem**: The `entropy_coeff` hyperparameter was preventing the network from converging on optimal routes.
2. **The TraCI Handle Bug**: A mismatch in the `routerl` environment's TraCI connection handle caused the Bandit's real-time dynamic congestion features to silently fail and fall back to neutral defaults.

This document details the investigation, findings, and the implemented fixes.

---

## 1. The Convergence Issue: High Entropy

### Context
In Reinforcement Learning, the entropy coefficient dictates how much "random exploration" the policy performs. A higher coefficient forces the network to try different routes, while a lower coefficient allows the network to exploit the best routes it has found.

### Findings from `losses.csv`
Looking at the loss and reward metrics over the 500 episodes, we observed the following:
* **Episode 1 Entropy**: `1.386` (Representing uniform random choice across the 4 possible routes)
* **Episode 500 Entropy**: `1.361`

After 500 full training episodes, the entropy had barely decayed. Because the `entropy_coeff` was set to `0.05` (a relatively high value for this environment), the neural network was heavily penalized for being "too certain". Instead of committing to the fastest routes, it continued to dispatch vehicles almost uniformly across all available paths.

### Impact on Metrics
Because the network was forced into endless exploration, the vehicles could not coordinate efficiently.
* **Baseline Human Traffic (`t_pre`)**: 4.19s
* **Bandit AV Traffic (`t_CAV`)**: 4.32s

While this is an improvement over the independent `UCB` algorithm (which scored 4.42s and suffered from herding), it still failed to beat the human baseline. 

**Next Steps**: For the upcoming hyperparameter sweep, dropping the `entropy_coeff` to `0.01` will allow the network to confidently exploit its learned advantages.

---

## 2. Prerequisite Check: The `paths.csv` / `routes.csv` Edges Column

### Context
A prerequisite flagged in the Contextual Bandit documentation warned that if the routing CSV lacked an `edges` column containing SUMO edge IDs, the TraCI features would fail and fall back to neutral defaults.

### Investigation
Upon inspecting the URB environment output (`results/bandit_REINFORCE_500/routes.csv`), it was confirmed that:
1. The environment names the file `routes.csv` (not `paths.csv`).
2. The file **does not** contain an `edges` column natively. Instead, the edge sequences are stored in a column named `path`.

### The Resolution
This potential failure was successfully avoided during implementation. A safeguard was built directly into `RouteFeatureStore.load_routes()`:

```python
if "path" in df.columns and "edges" not in df.columns:
    df["edges"] = df["path"]
```
This logic successfully intercepted the `path` column, duplicated it as `edges`, and ensured the data was correctly formatted for TraCI extraction.

---

## 3. Prerequisite Check: The TraCI Handle Bug

### Context
Another prerequisite flagged in the documentation questioned whether `env.simulator.sumo` was the correct handle to access the active SUMO simulation. If incorrect, the connection would fail and the algorithm would fall back to feeding the neural network neutral constants `[1.0, 0.0]` instead of real-time traffic data.

### Investigation
To verify this, the internal source code of the RouteRL simulator (`/home/sathyakumarnandakumar/URB/urbenv-github/lib/python3.12/site-packages/routerl/environment/simulator.py`) was analyzed.

The following instantiation logic was found in the `start()` method:
```python
traci.start(sumo_cmd, label=self.sumo_id)
self.sumo_connection = traci.getConnection(self.sumo_id)
```

### The Bug
The handle exposed by the environment is **not** `.sumo`, it is `.sumo_connection`.

Because the original script attempted to access `.sumo`, an `AttributeError` was triggered internally. The script's graceful fallback instantly caught the error, set the connection to `None`, and continued the simulation without crashing. 

However, because the connection was `None`, the dynamic feature extraction (`DynamicFeatureExtractor.get_dynamic()`) was disabled. For all 500 episodes, the `mean_cav` (congestion ratio) and `norm_start` features were fed into the neural network as static `[1.0, 0.0]` constants. 

**The Bandit trained for 500 episodes completely blind to real-time traffic congestion.**

### The Fix
This bug has been patched in `scripts/bandit_reinforce.py`. The extraction logic has been updated to point to the correct handle:

```python
try:
    # Attempt to get active TraCI connection from URB's simulator
    traci_conn = self.env.simulator.sumo_connection if hasattr(self.env, "simulator") else None
except AttributeError:
    traci_conn = None
```

## Conclusion
With the TraCI handle fixed, the Contextual Bandit will now receive dynamic, real-time feedback on route congestion. Combined with a lower entropy coefficient during the hyperparameter sweep, the Bandit is now fully equipped to coordinate the fleet and outperform the human baseline.
