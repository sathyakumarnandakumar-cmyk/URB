# Last Run Tasks Summary

This file serves as a memory aid for the recent work and experiments completed on this VM before its deletion.

## Recent Objectives & Tasks Completed

1. **Migrated URB Documentation & Adapted IQL To RG_MFD**
   - Integrated the Independent Q-Learning (IQL) algorithm into the native `RG_MFD` simulation framework.
   - Refactored legacy DQN implementation into a Multi-Agent Reinforcement Learning (MARL) architecture.
   - Replaced legacy environment wrappers with native simulation callbacks using `MARLIQLAdapter`.

2. **Bandit REINFORCE Algorithm Setup & Hyperparameter Sweep**
   - Developed and integrated the Bandit-REINFORCE algorithm into the Urban Routing Benchmark (URB).
   - Conducted a robust 54-job hyperparameter sweep for Bandit-REINFORCE routing algorithm on the `ingolstadt_custom` network.
   - Executed a successful long-running 4000-episode Bandit REINFORCE training experiment.
   - Monitored training stability and analyzed metrics, including vector outputs and routing performance.
   - Resolved simulation issues, including patching persistent SUMO TraCI handle warnings.

3. **Development Environment Configuration**
   - Set up an isolated development environment using `uv`.
   - Managed dependencies for running RL models safely without conflicts.

4. **UCB Algorithm Transition**
   - Implemented the UCB (Upper Confidence Bound) algorithm for RouteRL.
   - Documented the transition from the base URB to the `UCB` learning model implementation.

## Key Changes
- `bandit_reinforce.py` implementation and script refactoring.
- Configurations for environment and tasks aligned (`config1.json`, `long_run.json`).
- `urbenv` virtual environment set up to isolate and manage dependencies.

*Note: Large virtual environment files (e.g. `urbenv/`) and heavy cached data were excluded from the final GitHub sync to save space.*
