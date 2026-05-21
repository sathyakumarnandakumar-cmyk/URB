from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl import TrafficEnvironment
from routerl import DQN
from routerl import UCB
from routerl import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#########################
## Hyperparameter setting
#########################


new_machines_after_mutation = 10
human_learning_episodes = 0
training_episodes = 300
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
            "marginal_cost_coefficient_beta": 200,
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
env = TrafficEnvironment(seed=44, create_agents=False, create_paths=True, randomize_sumo_seed=False, marginal_cost_calculation=False, marginal_cost_calculation_machine_to_all=False, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

## Initialize the connection with SUMO
env.start()
env.reset()


## Mutation -> a number of human agents switch to using AVs 
pre_mutation_agents = env.all_agents.copy()

env.mutation_odd_id_agents()

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

## Humans choose the action that will collectively correspond to system optimal (route 0)
for human in env.human_agents:
    human.default_action = 0

machines = env.machine_agents.copy()
mutated_humans = dict()
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break

## Initialize UCB object for each agent
free_flows = env.get_free_flow_times()
for h_id, human in mutated_humans.items():
    num_actions = env.action_space_size
    num_states = pow(len(env.all_agents), num_actions) 
    alpha = 0.1
    beta = 3
    
    mutated_humans[h_id].model = UCB(num_states = num_states, num_actions = num_actions,
                                    num_agents = len(env.all_agents), alpha = alpha, beta=beta)


################
## Training loop
################

pbar = tqdm(total=total_episodes, desc="AV learning")
for episode in range(training_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
            last_action = mutated_humans[agent].last_action
            last_observation = mutated_humans[agent].last_obs

            mutated_humans[agent].learn(last_action, obs)
            action = None
        else:
            action = mutated_humans[agent].act(observation)
            mutated_humans[agent].last_action = action

        env.step(action)


    pbar.update()


###############
## Testing loop
###############

pbar.set_description("Testing")
for episode in range(testing_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = mutated_humans[agent].act(observation)
        env.step(action)
    pbar.update()

pbar.close()

## Plot results
env.plot_results()

## Close the connection with SUMo
env.stop_simulation()