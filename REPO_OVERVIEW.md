# URB (Urban Routing Benchmark) Repository Overview

Welcome to the **URB** codebase! This document provides a high-level, comprehensive overview of the repository's architecture. `URB` is a benchmarking environment that unifies multi-agent reinforcement learning (MARL) evaluation across real-world traffic networks paired with realistic demand patterns. 

For an extensive explanation of the project context and setup instructions, refer to the main [README.md](README.md).

---

## System Architecture Pipeline

The diagram below illustrates how data flows through the URB framework—from initial configuration and environment simulation to final metric aggregation.

```mermaid
flowchart TD
    %% Define Styles
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef execute fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef analyze fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;

    %% Nodes
    subgraph Inputs ["1. Inputs & Configuration"]
        A[config/]:::input
        B[networks/]:::input
        A1[algo_config]
        A2[env_config]
        A3[task_config]
        A --- A1 & A2 & A3
    end

    subgraph Execution ["2. Execution Engine"]
        C[scripts/]:::execute
        D[baseline_models/]:::execute
        E[(SUMO Simulator)]
        C -- Uses heuristics --> D
        C -- Drives --> E
    end

    subgraph Outputs ["3. Telemetry Output"]
        F[results/]:::output
        F1[exp_config.json]
        F2[episodes/ (CSV)]
        F3[SUMO_output/ (XML)]
        F4[losses/ (CSV)]
        F --- F1 & F2 & F3 & F4
    end

    subgraph Analysis ["4. Analysis & Aggregation"]
        G[analysis/]:::analyze
        H[leaderboard/]:::analyze
        G1[metrics/BenchmarkMetrics.csv]
        H1[docs/leaderboard/index.html]
    end

    %% Connections
    Inputs -->|Hyperparameters & Demand Data| Execution
    Execution -->|Raw Simulation Logs| Outputs
    F2 & F3 -->|Raw Telemetry| G
    G -->|Extracts| G1
    F1 & G1 -->|Configs & Metrics| H
    H -->|Generates| H1
```

---

## Comprehensive Directory Map

To explore the codebase in extreme technical depth—including explicit code snippets, function signatures, and internal logic flows—please click the **Detailed Reference** links provided under each core directory below.

### 1. The Execution Engine
The directories responsible for actually running the simulations and training agents.
- **`training.md`**: 📖 **Start Here for RL Theory:** An extensively detailed document explaining the math, the MDP (State/Action/Reward), and how the execution pipeline works in plain English. ([Read training.md](training.md))
- **`scripts/`**: The core execution hub. Contains implementations of algorithms like IPPO, IQL, MAPPO, VDN, QMIX, and Centralized DQN. These scripts load configurations, mutate humans into AVs, and orchestrate the PyTorch/TorchRL learning loops.
  - 📖 **Detailed Reference:** [docs/codebase_reference/scripts.md](docs/codebase_reference/scripts.md)
- **`baseline_models/`**: Contains implementations of non-RL routing baseline models. These models are typically hard-coded heuristics or traditional routing algorithms (e.g., `AON`, `Random`, `Gawron`) used as standard benchmarks to evaluate complex RL approaches.
  - 📖 **Detailed Reference:** [docs/codebase_reference/baseline_models.md](docs/codebase_reference/baseline_models.md)

### 2. Configuration & Environments
The static files that dictate the parameters of a simulation.
- **`config/`**: Stores the highly modular JSON configuration scheme for the benchmark.
  - `algo_config/`: Hyperparameter settings for different RL algorithms (e.g., Learning Rates, PPO clipping).
  - `env_config/`: Configurations for the environment logic (e.g., path generation sampling).
  - `task_config/`: Parameters defining simulated scenarios (e.g., portion of AVs, human learning durations, AV behavior).
- **`networks/`**: Contains the physical traffic networks and associated demand data (e.g., `gretz_armainvilliers`, `nangis`, `nemours`, `provins`, `saint_arnoult`, `ingolstadt_custom`). These define the roads and the origins/destinations for all agents.

### 3. Data Output & Analytics
Where the simulation results are dumped and subsequently evaluated.
- **`results/`**: (Generated upon execution). This is where the output data from running the scripts are saved, uniquely organized by the `<exp_id>`. It holds raw SUMO XMLs, RouteRL episode CSVs, and training losses.
  - 📖 **Detailed Reference:** [docs/codebase_reference/results.md](docs/codebase_reference/results.md)
- **`analysis/`**: Contains the analytics engine (`metrics.py`). This script parses the massive raw telemetry dumped into `results/`, aligns the multi-agent data, and extracts final performance indicators (e.g., system travel time, CAV advantage, cost of learning, instability).
  - 📖 **Detailed Reference:** [docs/codebase_reference/analysis.md](docs/codebase_reference/analysis.md)
- **`leaderboard/`**: Contains automation logic to dynamically scan the aggregated metrics across all experiments and generate the interactive HTML leaderboard to display state-of-the-art results.
  - 📖 **Detailed Reference:** [docs/codebase_reference/leaderboard.md](docs/codebase_reference/leaderboard.md)

### 4. Utilities & Support
- **`tests/`**: Contains automated `pytest` suites validating the functionality of the baselines and RL scripts, ensuring the pipeline doesn't crash prior to deploying long-running training jobs.
  - 📖 **Detailed Reference:** [docs/codebase_reference/tests.md](docs/codebase_reference/tests.md)
- **`docs/`**: Contains visual assets, static documentation images, and the detailed codebase references.
- **`urbenv/`**: A local virtual environment directory containing Python dependencies.
