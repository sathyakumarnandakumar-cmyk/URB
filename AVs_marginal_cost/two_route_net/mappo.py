from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

from routerl import TrafficEnvironment
from routerl import MAPPO
from routerl import Keychain as kc
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#########################
## Hyperparameter setting
#########################

new_machines_after_mutation = 10
human_learning_episodes = 0
training_episodes = 1000
testing_episodes = 100

total_episodes = human_learning_episodes + training_episodes + testing_episodes

env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,
        "agents_csv_file_name": "agents.csv",
        "num_agents" : 22,

        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents",
            "marginal_cost_coefficient_beta": 0.3,
            "records_folder": "training_records",
        }
    },
    "simulator_parameters" : {
        "network_name" : "two_route_yield",
        "sumo_type" : "sumo",
    },  
    "environment_parameters": {
        "save_every": 1
    },
    "plotter_parameters" : {
        "phases" : [0, human_learning_episodes, int(training_episodes) + human_learning_episodes],
        "smooth_by" : 50,
        "phase_names" : [
            "Human learning", 
            "Mutation - Machine learning",
            "Testing phase"
        ],
        "plot_choices": "basic",
        "records_folder": "training_records",
        "plots_folder": "plots",
    },
    "path_generation_parameters":
    {
        "number_of_paths" : 2,
        "beta" : -3,
        "visualize_paths" : True
    }
}

## Environment initialization
env = TrafficEnvironment(seed=41, create_agents=False, create_paths=True, randomize_sumo_seed=False, marginal_cost_calculation=True, marginal_cost_calculation_machine_to_all=False, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

## Initialize the connection with SUMO
env.start()
env.reset()
    
pre_mutation_agents = env.all_agents.copy()

## Mutation -> a number of human agents switch to using AVs 
env.mutation_odd_id_agents()

## Humans choose the action that will collectively correspond to system optimal (route 0)
for human in env.human_agents:
    human.default_action = 0


## Instantiate MAPPO algorithm
num_agents=len(env.machine_agents)
state_size = 2

mappo = MAPPO(
    state_size=state_size,
    action_space_size=2,
    num_agents=num_agents,
    shared_policy=False,
    share_critic=True,
    policy_arch_kwargs={'num_hidden':2, 'widths':[32,64,32]},
    critic_arch_kwargs={'num_hidden':2, 'widths':[64,64,64]},
    lr_actor=3e-4,
    lr_critic=3e-4
)


machines = env.machine_agents.copy()
mutated_humans = dict()
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break

raw_ids = sorted(int(k) for k in mutated_humans.keys())   # [1,3,5,7,9,11,13,15,17,19]
id_to_idx = { raw_id: idx for idx, raw_id in enumerate(raw_ids) }


################
## Training loop
################

pbar = tqdm(total=total_episodes, desc="AV learning")
for episode in range(training_episodes):
    env.reset()
    
    states, actions, rewards, logps, next_states, dones, agent_idxs = [], [], [], [], [], [], []

    for agent in env.agent_iter():
        raw_id = int(agent)
        idx = id_to_idx[raw_id]

        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            
            last_obs = mappo.get_last_observation(idx)
            last_action = mappo.get_last_action(idx)
            last_logp = mappo.get_last_log_prob(idx)

            states.append(last_obs)
            actions.append(last_action)
            rewards.append(reward)
            logps.append(last_logp)
            next_states.append([0]*state_size)
            agent_idxs.append(idx)
            dones.append(1)
            action = None
        else:
            action = mappo.act(observation, idx)
        env.step(action)

    # update MAPPO
    mappo.learn(states, actions, rewards, logps, next_states, dones, agent_idxs)

    if mappo.loss_actor and mappo.loss_critic:
        pbar.set_description(f"Ep {episode} actor={mappo.loss_actor[-1]:.4f} critic={mappo.loss_critic[-1]:.4f}")
    else:
        pbar.set_description(f"Ep {episode} actor=N/A critic=N/A")
    pbar.update()


free_flows = env.get_free_flow_times()
for h_id, human in mutated_humans.items():
    initial_knowledge = free_flows[(human.origin, human.destination)]
    initial_knowledge = [0, 0]
    mutated_humans[h_id].model = mappo.get_policy(id_to_idx[int(h_id)])


###############
## Testing loop
###############

mappo.eval()

pbar.set_description("Testing")
for episode in range(testing_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = mappo.act(observation, id_to_idx[int(agent)])
        env.step(action)
    pbar.update()

pbar.close()

## Plot results
env.plot_results()

## Close the connection with SUMo
env.stop_simulation()