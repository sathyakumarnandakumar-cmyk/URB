# `scripts/` Module Reference

This directory contains the main execution files that bind the URB multi-agent traffic environment to learning policies (Deep Reinforcement Learning algorithms and heuristics).

## Dependencies & Imports

```python
import torch
from routerl import TrafficEnvironment, Keychain as kc
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
```
- **External Frameworks:** Utilizes `routerl` to instantiate the `TrafficEnvironment`. MARL scripts heavily rely on the PyTorch `torchrl` ecosystem to define multi-agent architectures (`MultiAgentMLP`), memory buffers (`LazyTensorStorage`), and loss computations (`ClipPPOLoss`).
- **Standard Library:** Uses `json` for config parsing, `argparse` for CLI configuration.

---

## Data Flow & Architecture

1. **Setup Phase:** A script (e.g., `ippo_torchrl.py`) receives command line arguments (network name, random seed, task configurations), loads the corresponding `json` configurations from `config/`, and injects these into `TrafficEnvironment`.
2. **Human Stabilization Phase:** The script loops `env.step()` for `human_learning_episodes`, simulating the network solely with non-autonomous (human) drivers allowing the traffic model to stabilize into equilibrium.
3. **Mutation Phase:** The script triggers `env.mutation()`, converting a subset of the agents from "Human" logic to "Machine" (AV) logic based on the `task_config`.
4. **Environment Wrapping (RL Only):** The environment is wrapped in a `PettingZooWrapper` to convert outputs into standard RL shapes, and a `TransformedEnv` sums rewards into an `episode_reward`.
5. **Observation Encoding (RL Only):** An origin-destination (OD) embedding is prepended to the raw observation using `AppendODEmbedding` so policies are location-aware.
6. **Execution/Training Phase:** The script loops over epochs. Agents select actions (routes) from the actor network. At the end of an episode, `env.last()` emits the true travel times (negative rewards), which populate the `ReplayBuffer`. The algorithm computes Generalized Advantage Estimation (GAE) and backpropagates errors through the PPO/QMIX/VDN loss functions.
7. **Testing Phase:** Runs inference loops (policy evaluation mode) strictly to record deterministic baseline performances.

---

## Deep Class/Script References

### `base_script.py`
Serves as the skeleton/template for writing new experiments. It sets the execution order:
```python
# 1. Initialize environment
env = TrafficEnvironment(..., simulator_parameters={"sumo_type": "sumo", ...})

# 2. Human learning
for episode in range(human_learning_episodes):
    env.step()

# 3. Mutation
env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)

# 4. User-Defined Training block goes here!

# 5. Clean up and analyze
env.stop_simulation()
run_metrics_analysis(exp_id, results_folder="../results")
```

### `ippo_torchrl.py` (Independent PPO)
Implements an Independent PPO where each agent learns decentralized policies, but parameters might be shared.

**Core TorchRL Objects:**
- **Policy Network (Actor)**: Uses `MultiAgentMLP`. The state flows from `PettingZooWrapper` into `AppendODEmbedding`, then into the `MultiAgentMLP` generating route logits. `ProbabilisticActor` samples this distribution.
- **Critic Network**: Also a `MultiAgentMLP`. Crucially, because it is *Independent* PPO, `centralised=False` is passed to the critic initialization.
- **Collector**: `SyncDataCollector` orchestrates running the policy in the environment and collecting `TensorDict` transitions.
- **Loss Computation**:
```python
loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=normalize_advantage,
)
```
During training iterations, subdata is sampled from the `ReplayBuffer`. The algorithm calculates `loss_objective` (actor clip loss), `loss_critic` (MSE of travel times), and `loss_entropy`.

### `utils.py`
Provides cross-script functional utilities.
- **`AppendODEmbedding` (Class)**: A `nn.Module` that maps scalar integer OD pairs to learned dense vectors.
  ```python
  def forward(self, observation: torch.Tensor) -> torch.Tensor:
      od_embedding = self.embedding(self.od_ids)
      # Appends the embedding onto the end of the feature array
      return torch.cat([observation, od_embedding], dim=-1)
  ```
- **`clear_SUMO_files` (Function)**: Parses `SUMO_output/` dynamically. If an XML `<tripinfo>` file is completely empty (no vehicles routed during an episode), it deletes it to save disk space and re-indexes the remaining files so analysis scripts don't fail.
