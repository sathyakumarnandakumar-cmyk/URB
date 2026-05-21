from tqdm import tqdm
import os
import sys

from routerl import TrafficEnvironment
from routerl import DQN
from routerl import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#########################
## Hyperparameter setting
#########################

new_machines_after_mutation = 10
human_learning_episodes = 0
training_episodes = 500
testing_episodes = 100

total_episodes = human_learning_episodes + training_episodes + testing_episodes

env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,

        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents",
            "marginal_cost_coefficient_beta": 10,
            "records_folder": "training_records"
        }
    },
    "simulator_parameters" : {
        "network_name" : "two_route_yield",
        "sumo_type" : "sumo",
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
        "beta" : -1,
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

## Initialize IDQN object for each agent
free_flows = env.get_free_flow_times()
for h_id, human in mutated_humans.items():
    initial_knowledge = free_flows[(human.origin, human.destination)]
    initial_knowledge = [0, 0]
    state_size = 2

    mutated_humans[h_id].model = DQN(state_size, len(initial_knowledge), epsilon_decay_rate=0.01,
                                                                            memory_size=200,   
                                                                            batch_size=32,
                                                                            learning_rate=0.1)
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

            mutated_humans[agent].model.learn(observation, last_action, reward)
            action = None
        else:
            action = mutated_humans[agent].model.act(observation)
            mutated_humans[agent].last_action = action

        env.step(action)


    pbar.update()


###############
## Testing loop
###############

for h_id, human in mutated_humans.items():
    mutated_humans[h_id].model.eval()

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
