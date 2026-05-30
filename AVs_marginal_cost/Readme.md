# Autonomous vehicles need social awareness to find optima in multi-agent reinforcement learning routing games.

> Previous work has shown that when multiple selfish Autonomous Vehicles (AVs) are introduced to future cities and start learning optimal routing strategies using Multi-Agent Reinforcement Learning (MARL), they may destabilize traffic systems, as they would require a significant amount of time to converge to the optimal solution, equivalent to years of real-world commuting.
> We demonstrate that moving beyond the selfish component in the reward significantly relieves this issue. If each AV, apart from minimizing its own travel time, aims to reduce its impact on the system, this will be beneficial not only for the system-wide performance but also for each individual player in this routing game.
> By introducing an intrinsic reward signal based on the marginal cost matrix, we significantly reduce training time and achieve convergence more reliably. Marginal cost quantifies the impact of each individual action (route-choice) on the system (total travel time). Including it as one of the components of the reward can reduce the degree of non-stationarity by aligning agents' objectives. Notably, the proposed counterfactual formulation preserves the system's equilibria and avoids oscillations.
> Our experiments show that training MARL algorithms with our novel reward formulation enables the agents to converge to the optimal solution, whereas the baseline algorithms fail to do so. We show these effects in both a toy network and the real-world network of Saint-Arnoult. Our results optimistically indicate that social awareness (i.e., including marginal costs in routing decisions) improves both the system-wide and individual performance of future urban systems with AVs.

**All experiments are conducted using the RouteRL framework, which simulates the routing decisions of Autonomous Vehicles (AVs) using Multi-Agent Reinforcement Learning (MARL) and of human drivers using state-of-the-art human learning models across different traffic networks. Check the documentation of RouteRL [here](https://coexistence-project.github.io/RouteRL/).**

# Contents



```
./
├── saint_arnoult/ 
│   └── ...
├── two_route_net/ 
│   └── ...
|
```

`saint_arnoult` directory contains the script utilizing the **UCB** algorithm to run experiments in the Saint Arnoult network using demand from the [URB benchmark](https://openreview.net/pdf?id=SDJ3Y1ZkNz).

`two_route_net` directory contains the scripts utilizing the **UCB, MAPPO, IDQN** algorithms to run experiments using the toy two-route network that contains two routes and a yield sign and was explicitly designed for its game-theoretical properties. SUMO videos from runs on this network are included [here](https://github.com/COeXISTENCE-PROJECT/Marginal-cost/blob/main/video_try_network.mp4).

## Run experiments

<!--In this work, we demonstrate that including in each agent’s reward the marginal impact it imposes on other agents can significantly accelerate convergence toward the optimal solution.--> 

The TrafficEnvironment() class from RouteRL accepts these flags: 
- `marginal_cost_calculation`: if set to True, the marginal cost is calculated and introduced in the reward of the AV agents, taking into account only the impact of the agent on the other AV agents (default value is False).
- `marginal_cost_calculation_machine_to_all`: if set to True, the marginal cost is calculated and introduced in the reward of the AV agents, taking into account the impact on both AV and human agents of the system (default value is False).
- `randomize_sumo_seed`: if set to True the random seed of SUMO is changing in every episode enabling non-deterministic traffic dynamics (default value is False).

*The marginal cost matrices are saved in the `training_records/marginal_cost_matrices` path for each episode of an experiment.*

The TrafficEnvironment() class from RouteRL accepts some additional parameters:
- `marginal_cost_coefficient_beta`: specifies the coefficient $\beta$ that scales how strongly an agent’s marginal cost contributes to its reward (default value is 0.5).

# Installation

- **Prerequisite**: Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).
- Install [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL) following:  
  ```
    pip3 install git+https://github.com/COeXISTENCE-PROJECT/RouteRL.git@dev  
  ```


