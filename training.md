# URB Training Theory & Technical Implementation

This document serves as an extensive, theoretical, and technical companion to the main repository. It explicitly details the underlying mathematics, the Reinforcement Learning (RL) Markov Decision Process (MDP) formulation, and how exactly the codebase translates these concepts into execution.

**Links:**
- [GitHub Repository](https://github.com/COeXISTENCE-PROJECT/URB)
- [URB Paper (arXiv)](https://arxiv.org/abs/2505.17734)

---

## 1. Theoretical Formulation (The Routing Game)

The core premise of the Urban Routing Benchmark (URB) is treating city-wide traffic navigation as a massive Multi-Agent game. 

### The Core Utility: Travel Time ($t$)
In traffic assignment problems, the primary metric is the time it takes to go from an Origin to a Destination (OD pair).
- For **Human Drivers**, they act as "rational utility maximizers" trying to minimize their own travel time.
- For **CAVs (Connected Autonomous Vehicles)**, travel time serves as the fundamental negative reward signal for Reinforcement Learning. The goal of the RL algorithms is to minimize the expected aggregate travel time.

### User Equilibrium vs. System Optimum
Traffic theory relies heavily on Wardrop's Principles to describe network states:

1. **Wardrop's First Principle (User Equilibrium)**: How selfish behavior causes humans to migrate to faster roads until all routes equalize in travel time, eliminating the incentive to switch but creating suboptimal city-wide traffic.
2. **Wardrop's Second Principle (System Optimum)**: How minimizing the average travel time for everyone requires some individuals to take a "sacrifice" route so they don't block critical highways.

**The Price of Anarchy**: How URB uses Reinforcement Learning to bridge the gap between the selfish human baseline and the cooperative system optimum.

### The "Mutation" Concept
The benchmark runs a unique procedural phase called **Mutation**. 

**What is it?**
1. **Pre-mutation**: The simulation runs with 100% human drivers for many "days" (episodes). They explore routes and eventually settle into a User Equilibrium (UE)—a state where no single human can unilaterally switch routes to improve their travel time. This creates a realistic, baseline traffic jam.
2. **Mutation Event**: Suddenly, a defined percentage of humans "mutate" into CAVs. They delegate their routing choices to a machine algorithm (the RL policy).
3. **Training**: The CAVs now explore the environment collectively, destabilizing the old human equilibrium, and attempt to find a System Optimum (SO) where the fleet's actions minimize overall congestion.

**Why is this needed?**
In the real world, autonomous vehicles won't be deployed onto empty, pristine highways. They will be introduced into existing, highly congested urban networks where human commuters already have deeply ingrained, selfish routing habits (User Equilibrium). If an RL environment simply spawned CAVs onto an empty map, it would be solving an artificially easy problem. By forcing a period of human-only equilibrium first, and then *mutating* a subset of those humans into CAVs, URB establishes a rigorous, realistic counterfactual baseline. This allows researchers to measure the *exact delta* (the real-world impact) the RL agents have on breaking an already established traffic jam.

**How is this achieved in the codebase?**
This is achieved via the `env.mutation()` command inside the main execution scripts (e.g., `scripts/ippo_torchrl.py`). 
1. The script reads the `ratio_machines` parameter from `task_config.json` (e.g., `0.4` for 40%).
2. Under the hood, the `routerl` environment iterates through the list of simulated agents. It randomly selects 40% of the agents that are currently flagged as "Human".
3. For these selected agents, their internal logic module is swapped out. The human probabilistic routing model (like the `gawron` heuristic) is deleted, and they are re-classified as "Machine" (AV) agents.
4. Crucially, their core physical properties—their Origin, Destination, and departure time—remain **identical**. The only thing that mutates is the *delegation of routing control*. From that episode onward, whenever those specific agents need to choose a route, they wait for the external PyTorch policy to output an action, rather than deciding for themselves.

---

## 2. Demystifying the Execution Pipeline

The execution scripts (like `scripts/ippo_torchrl.py`) follow a strict 7-phase architecture. Here is a simplified, non-cryptic breakdown of what those steps actually do:

1. **Setup Phase:**
   When you run the command, Python reads three JSON files from `config/` (algorithm math, environment physics, and the specific traffic scenario). It passes these rules to the `TrafficEnvironment` class so the simulator knows exactly how many cars to spawn and where they want to go.

2. **Human Stabilization Phase (`human_learning_episodes`):**
   The simulator runs for several "days" (episodes) using *only* human drivers. The humans try different routes and mathematically "learn" the fastest way to work. How do they achieve this? By using a mathematical behavioral model (typically **Gawron's Algorithm**):
   - **Initial Exploration:** On Day 1, humans pick routes somewhat randomly based on distance.
   - **Receiving Feedback:** At the end of the day, SUMO records exactly how many seconds it took each driver to reach their destination.
   - **Updating Probabilities:** Instead of immediately jumping to the absolute fastest route (which causes massive, oscillatory traffic jams when everyone jumps to the same road simultaneously), the `gawron` model updates a **probability distribution**. Humans compare the time they just experienced against their historical estimates for other routes. Using the `human_alpha` and `human_beta` configuration parameters, they tweak their route probabilities. If Route A was faster today, the *chance* they take Route A tomorrow increases slightly.
   - **Convergence (User Equilibrium):** By repeating this loop for 100 or 200 episodes, the thousands of tiny probabilistic adjustments slowly balance out. The humans organically distribute themselves across the city until the travel times on all used routes equalize. The traffic jam becomes a stable, realistic baseline. *(Note: There is also a `greedy` human model available, where humans simply pick the absolute fastest route from yesterday with 100% certainty, modeling highly reactive drivers).*

3. **Mutation Phase:**
   The script looks at the task configuration. If the config says "40% AVs", it randomly selects 40% of the human drivers and hands control of their steering wheels over to the AI (the PyTorch neural network). 

4. **Environment Wrapping:**
   Traffic simulators like SUMO don't naturally speak the language of Deep Learning. `PettingZooWrapper` acts as a translator. It takes SUMO's messy traffic data and formats it into clean matrices (Tensors) that PyTorch can mathematically process.

5. **Observation Encoding:**
   If an AI just sees "there are 5 cars ahead", it doesn't know where it is in the city. The `AppendODEmbedding` step gives the AI a "GPS coordinate". It takes the agent's start point and destination (OD pair) and converts it into a dense mathematical vector so the neural network knows *where* it is routing the car.

6. **Execution/Training Phase:**
   The AI drives the cars for hundreds of episodes. 
   - *Action*: The AI picks a route.
   - *Result*: The car reaches its destination. SUMO measures exactly how many seconds the trip took.
   - *Learning*: The true travel time is sent back to PyTorch as a "negative reward". If the trip was slow, the neural network adjusts its weights to discourage picking that route again.

7. **Testing Phase:**
   Training is frozen. The AI uses its finalized logic to drive the cars one last time. This deterministic run proves whether the AI actually solved the traffic jam or just got lucky during training.

---

## 3. How Task Complexity is Configured

The URB paper emphasizes extensive experimental flexibility. The environment can simulate thousands of interacting agents across varied real-world road networks (such as `gretz_armainvilliers`, `nangis`, `nemours`, `provins`, `saint_arnoult`, and `ingolstadt_custom`). **How does the codebase seamlessly orchestrate this complexity?** 
This is handled entirely through a highly modular JSON injection pattern located in `config/task_config/` and `config/env_config/`. The core execution scripts (`base_script.py`) parse these dictionaries and strictly enforce them during the PyTorch execution loop.

> [!IMPORTANT]
> **The Configuration Lifecycle**
> All settings described below (e.g., `av_behavior`, `human_model`, `ratio_machines`) are **strictly constant for the entire duration of a single training experiment.** They do *not* change dynamically between episodes or epochs. When you launch a script (e.g., `python ippo_torchrl.py --task-conf config4`), the environment locks in those rules. For example, if `"av_behavior"` is `"selfish"`, the PyTorch network will receive the selfish reward function for every single episode until the script finishes. To test a different configuration, you must launch a completely new, separate script execution. This strict separation ensures the Markov Decision Process remains scientifically valid.

### Full vs. Mixed Autonomy
Controlled via `"ratio_machines"`.
- `"ratio_machines": 1.0` means 100% of drivers mutate into CAVs (Full Autonomy). The network becomes a purely cooperative MARL problem.
- `"ratio_machines": 0.4` means 40% are CAVs, and they must share the road with 60% humans. This transforms the environment into a much harder **mixed-autonomy game**, where the RL agents must learn to navigate around unpredictable, selfish human drivers.

### CAV Behavior Profiles (Malicious, Altruistic, Selfish)
Controlled via `"av_behavior"` in the JSON config. 
**How is it implemented and when?** 
This is implemented dynamically during the **Execution Phase** when the `routerl` environment computes the reward scalar for the PyTorch agents. The reward function inherently alters the optimization landscape:
- `"av_behavior": "selfish"`: The environment extracts the specific `duration` of the trip for agent $i$ directly from SUMO's C++ bindings. It returns $R_i = -t_i$. The agent learns to cut off other cars to minimize its own trajectory.
- `"av_behavior": "altruistic"`: The environment intercepts the episode termination phase, sums the travel times of the *entire* fleet, and broadcasts this scalar. The reward returned to agent $i$ is $R_i = -\sum t$. To maximize this, an agent might purposely choose a slower peripheral route to free up the central highway, embodying true cooperative Multi-Agent optimization.
- `"av_behavior": "malicious"`: The environment actively inverts the reward signal to maximize the delay of *other* agents. The RL agent learns adversarial behaviors—seeking out and blocking critical intersections to cause gridlock.

### Human Behavior Models (Probabilistic vs Greedy)
Controlled via `"human_model"`.
**How is it implemented and when?**
Implemented during the **Setup Phase**. When `routerl` initializes, it instantiates polymorphic Python classes (e.g., `GawronModel` or `GreedyModel`) and assigns them to the internal logic blocks of non-mutated agents.
- `"human_model": "gawron"`: During the **Human Stabilization Phase**, when `env.step()` is called, the agent executes its `update()` method. It mathematically shifts its internal probability distribution (using the `human_alpha` learning rate) based on historical delays, preventing herd behavior.
- `"human_model": "greedy"`: The agent ignores continuous probabilities. At the end of the episode, its `update()` method executes a strict `argmin()` over historical delays. The agent picks the absolute fastest route for the next day, resulting in severe, oscillatory "herd" traffic jams as all greedy humans migrate to the same highway simultaneously.

### Human Adaptations
Controlled via `"should_humans_adapt"`.
**How is it implemented and when?**
This flag dictates the stationarity of the environment during the **Execution Phase** (after Mutation).
- `false`: When CAVs begin exploring via PyTorch, the human agents' probability tables are "locked" (their learning rate drops to zero). Humans blindly drive the exact same routes they settled on during stabilization. This provides a **stationary**, predictable background for the AI to learn against.
- `true`: The human `update()` function remains active. While the AI is learning new routes, the humans notice the traffic patterns shifting and probabilistically change their own routes in response. This creates a highly volatile, **non-stationary Markov Decision Process**, a notoriously difficult paradigm in MARL where the environment actively reacts to the learning agent.

---

## 4. The Markov Decision Process (MDP) in Depth

For the Reinforcement Learning algorithms implemented in URB (IPPO, MAPPO, QMIX, VDN, Centralized DQN), the environment must be rigorously formalized as an MDP tuple: $\langle S, A, P, R, \gamma \rangle$.

### State Space / Observations ($S$)
The observation space defines exactly what feature tensors the PyTorch neural network processes before making a routing decision. It is explicitly designed to be **decentralized**—agents only possess local and historical knowledge, mimicking the limited sensor/V2I communication of real-world CAVs. 

The specific observation schema is dictated by `config/env_config/config1.json`:
```json
"observations" : "previous_agents_plus_start_time",
"num_samples" : 10
```

**Exact Feature Array Example:**
Under these settings, the raw observation extracted from the SUMO simulator is a continuous array of size 11 (1 start time + 10 historical delays):
```python
raw_obs = [28800.0, 315.2, 320.1, 400.5, 395.0, 380.0, 310.2, 315.5, 305.0, 312.4, 314.1]
```
- **Index 0 (`28800.0`)**: The agent's own scheduled departure time in seconds (e.g., 08:00 AM).
- **Index 1 to 10 (`315.2, 320.1...`)**: The travel times (delays) experienced by the last 10 agents who recently completed trips anywhere in the network. This scalar array acts as a low-dimensional proxy for "current city-wide congestion".

**The OD-Embedding (Contextualizing the State):**
Knowing the congestion isn't enough; the agent must know *where* it is trying to go. However, passing discrete Origin and Destination node IDs directly to a neural network is mathematically unsound.
Instead, the PyTorch execution script utilizes the `AppendODEmbedding` module (found in `scripts/utils.py`). It uses a PyTorch `nn.Embedding` layer to convert the discrete OD pair into a continuous vector:

```python
# From scripts/utils.py
class AppendODEmbedding(nn.Module):
    def __init__(self, od_ids: list[int], num_od_pairs: int, embedding_dim: int):
        super().__init__()
        # Maps every possible Origin-Destination combination to a dense vector
        self.embedding = nn.Embedding(num_od_pairs, embedding_dim)
        self.register_buffer("od_ids", torch.as_tensor(od_ids, dtype=torch.long))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Expands the embedding to match batch size and concatenates it to features
        od_embedding = self.embedding(self.od_ids)
        return torch.cat([observation, od_embedding], dim=-1)
```

In `config/algo_config/ippo_torchrl/config1.json`, the `"od_embedding_dim"` is strictly set to `8`.
Therefore, the final input to the neural network is `Size 11 (raw_obs) + Size 8 (OD embedding) = Size 19 Feature Vector`. 

### Action Space ($A$)
The action space is **discrete** and bounded. 
Configured by `"number_of_paths": 4` in `env_config`.
When an agent spawns, `RouteRL` pre-calculates the 4 most logical physical routes (e.g., via Yen's K-Shortest Path algorithm) between the agent's origin and destination. 
The PyTorch Actor network outputs a Logit vector of size 4. A `Categorical` distribution is applied over these logits to sample an action $a \in \{0, 1, 2, 3\}$. The agent then strictly follows that predefined sequence of road edges in SUMO.

### State Transitions ($P$)

The transition dynamics $P(s_{t+1} | s_t, a_t)$ are fully governed by the microscopic physical simulation inside **SUMO**. 

**Important:** The micro-level physics in SUMO are **entirely pre-determined, fixed, and deterministic**. The RL agent does *not* learn how to press the gas pedal, steer the wheel, or avoid rear-ending the car in front of it. It operates purely at the macro-level of Routing.

Here is how the responsibilities are split:
1. **The PyTorch RL Agent (Learned/Dynamic):** Looks at the city congestion observation and decides, *"I will take Highway A instead of Downtown B."* (This is the discrete action $a \in \{0, 1, 2, 3\}$).
2. **SUMO Simulator (Fixed/Deterministic Physics):** Once the PyTorch agent selects "Highway A", the environment hands the car back to SUMO. SUMO uses strict mathematical formulas (specifically, models like the Krauss car-following model) to physically move the car along Highway A. SUMO automatically handles speeding up, braking for red lights, yielding at intersections, and preventing collisions using hard-coded C++ logic.

When the documentation mentions "complex, non-linear interactions," it refers to **emergence**. Even though SUMO's physics are fixed, when you put 5,000 cars on the same highway—each following those fixed rules—a tiny braking event by one car can cause a massive cascading shockwave traffic jam. The RL algorithm has to learn how to predict and route *around* these emergent shockwaves, but the physics causing them are completely hard-coded by the environment!

### Reward Formulation and Advantage ($R$ & $\gamma$)
The reward formulation is mathematically strict: $R_i = -t_i$. The environment emits the exact travel time (in seconds) it took for agent $i$ to complete the trip.

For advanced Actor-Critic architectures (like IPPO and MAPPO), the TorchRL library relies on incredibly specific hyperparameters to backpropagate this reward. Based directly on `config/algo_config/ippo_torchrl/config1.json`, here is exactly how the network optimizes the reward:

```json
"lr": 0.0002,
"clip_epsilon": 0.2,
"gamma": 0.97,
"lmbda": 0.9,
"policy_network_depth": 4,
"policy_network_num_cells": 256,
"critic_network_depth": 4,
"critic_network_num_cells": 256,
```

1. **Generalized Advantage Estimation (GAE):** 
TorchRL's `ValueEstimators.GAE` module calculates the Advantage scalar. The Advantage $\hat{A}_t$ answers the question: *Did taking this route perform better than the Critic network originally predicted?*
It is computed using the TD-residual $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$. 
Using the specific configurations, it computes the exponentially weighted average of these residuals:
$$ \hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $$
- **$\gamma = 0.97$ (Discount Factor):** Controls how much the agent cares about future traffic states versus immediate travel time. A high $0.97$ forces the agent to heavily weigh the long-term impact of its routing choice.
- **$\lambda = 0.9$ (GAE Smoothing):** Controls the bias-variance tradeoff in the advantage estimation.

2. **Actor-Critic Architecture:** 
Both the Policy $\pi_\theta$ (Actor) and Value function $V_\phi$ (Critic) networks are physically instantiated as massive 4-layer Multi-Layer Perceptrons (MLPs).
- `policy_network_depth: 4`, `policy_network_num_cells: 256`
They are optimized using Adam with a learning rate of `"lr": 0.0002`.

3. **PPO Clipping (The Surrogate Loss):** 
Once the Advantage $\hat{A}_t$ is computed, it is multiplied against the action log probabilities. However, traffic simulations are extremely volatile. If thousands of cars suddenly change routes based on a new policy, the entire city could gridlock. 
To prevent the policy from collapsing during these volatile traffic shifts, the PPO loss function restricts how much the neural network weights $\theta$ can update in a single step using `"clip_epsilon": 0.2`.
Let $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ be the probability ratio. The algorithm maximizes:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - 0.2, 1 + 0.2)\hat{A}_t) \right] $$
Because $\epsilon = 0.2$, the probability ratio is strictly clipped between $[0.8, 1.2]$. If a new route was incredibly successful, the network is not allowed to instantly shift 100% of the probability to that route (which would cause a massive herd traffic jam the next day). It forces a slow, stable convergence to the System Optimum.
