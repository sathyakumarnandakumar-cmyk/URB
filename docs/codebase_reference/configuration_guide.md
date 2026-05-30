# URB Configuration Architecture Guide

The Urban Routing Benchmark (URB) repository uses a modular, three-tier JSON configuration system. This allows researchers to mix and match different environment settings, tasks, and reinforcement learning algorithms without writing new code for each combination.

When running an experiment script (like `iql.py` or `ucb_new.py`), all three configurations are merged into a single dictionary that controls the entire pipeline.

---

## 1. Algorithm Config (`--alg-conf`)
**Location:** `config/algo_config/<algorithm_name>/<config_name>.json`

**Purpose:** Defines the hyperparameters specific to the learning algorithm being used (e.g., UCB, IQL, MAPPO). Crucially, this file also defines how long the AI agents are allowed to train.

**Common Parameters:**
*   `training_eps`: Number of episodes the Autonomous Vehicles (AVs) spend training and updating their policies.
*   `update_every`: How frequently (in episodes) the model weights/Q-tables are updated.

*Algorithm-specific parameters:*
*   **For UCB:** `alpha` (learning rate), `beta` (exploration coefficient).
*   **For Deep RL (IQL/MAPPO):** `lr` (learning rate), `batch_size`, `buffer_size`, `eps_init`, `eps_decay`, `widths` (neural network layer sizes), `num_epochs`.

> [!NOTE]
> **TorchRL Algorithms (`*_torchrl`):**
> Algorithms using the TorchRL library under the hood (like `ippo_torchrl` and `iql_torchrl`) handle training loops fundamentally differently than standard scripts. Instead of a flat `training_eps` variable, the training length is defined by collecting batches of data. It uses two parameters to calculate the total number of training episodes:
> 
> *   `agent_frames_per_batch`: How many interactions (episodes) each agent collects before the neural network updates.
> *   `n_iters`: How many times this batch-collection-and-update cycle repeats.
>
> The exact total is calculated as: `training_episodes = agent_frames_per_batch * n_iters`. So, for a config with `20` frames per batch and `200` iters, the total training episodes would be `4000`. To do a quick test run, simply lower `n_iters` (e.g., `n_iters: 2` for 40 episodes).

---

## 2. Task Config (`--task-conf`)
**Location:** `config/task_config/<config_name>.json`

**Purpose:** Defines the structure of the experiment phases (Human Learning -> Mutation -> Testing), the ratio of AVs, and how the baseline human agents behave.

**Common Parameters:**
*   `human_learning_episodes`: How many episodes humans simulate to stabilize traffic before AVs are introduced.
*   `test_eps`: How many episodes to run at the very end with exploration disabled (greedy policy) to evaluate performance.
*   `ratio_machines`: The percentage of the total vehicle population that will be converted into AVs (e.g., `0.4` means 40% AVs).
*   `should_humans_adapt`: Boolean controlling whether human agents continue to update their route choices during the AV training phase.
*   `av_behavior`: e.g., `"selfish"` (optimizing individual travel time) or `"cooperative"`.
*   `human_model`: The behavioral model for humans (e.g., `"gawron"`).
*   `human_alpha`, `human_beta`: Parameters specific to the human behavioral model.

---

## 3. Environment Config (`--env-conf`)
**Location:** `config/env_config/<config_name>.json`

**Purpose:** Defines simulator integration details, observation types, route generation settings, and logging/plotting frequencies.

**Common Parameters:**
*   `observations`: What the AI agents can "see" (e.g., `"previous_agents_plus_start_time"`).
*   `number_of_paths`: How many alternative routes are generated per Origin-Destination (OD) pair for the agents to choose from (e.g., `4`).
*   `path_gen_workers`, `num_samples`, `path_gen_beta`: Parameters for the stochastic path generation algorithm.
*   `plot_every`: How often (in episodes) to generate and update the visual plots.
*   `save_every`: How often to save checkpoints/metrics.
*   `smooth_by`: The rolling average window used for smoothing output plots.
*   `plot_choices`: Type of plots to generate (e.g., `"basic"`).

---

## Summary of the Timeline

The length of an experiment is determined by summing parameters across the **Task** and **Algorithm** configs:

1.  **Phase 1: Stabilization** (`human_learning_episodes` from Task Config)
2.  *Mutation Event*
3.  **Phase 2: Training** (`training_eps` from Algorithm Config)
4.  **Phase 3: Testing** (`test_eps` from Task Config)

Total Experiment Episodes = `human_learning_episodes` + `training_eps` + `test_eps`
