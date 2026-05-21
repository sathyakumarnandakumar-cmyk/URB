from tqdm import tqdm
import numpy as np
import os
import sys

from routerl import TrafficEnvironment
from routerl import DQN
from routerl import UCB
from routerl import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#########################
## Hyperparameter setting
#########################


new_machines_after_mutation = 10
human_learning_episodes = 10
training_episodes = 300
testing_episodes = 10

total_episodes = human_learning_episodes + training_episodes


origins = ['-42762428#0', '-1323419247', '71324347#2', '-100525445', '-282689981#0', '-101607967#0', '-101594809#0', '-659278038#3', '-101609498#5', '-120511302', '101416508#0', '-47374680#7', '336863934', '100475365#1', '-101611601', '-101604496#1', '101594836#1', '659281081#1', '-297823021', '-282689985#1', '526438862#2', '71324347#0', '297823019#4', '-47374665#1', '-297823024', '-100488715#4', '868087112', '75421017', '416409192#0', '-282689975#2', '120511932#0', '-101604513#1', '-100468675#0', '-416409190', '297823019#3', '-297823017', '-100468681#0', '-101600741#2', '-101608828', '-101594843', '-101787489', '-1323419243', '-101601559', '120511930', '659283000#1', '526439431#0', '-71324343#4', '-101607098#1', '101611600', '526438862#0', '101417074', '71324343#3', '-101411571#0', '-101749462', '-42762428#3', '71324347#5', '101611060', '-101415865#0', '-100525438#2', '-101418584#1', '71223791#0', '-71223800#7', '120509991', '-1006827381', '-416375813', '-71324343#5', '-101601551', '-416409191#0', '101606547#1', '-101746498#2', '-101601553#1', '71324347#1', '-101746516#2', '1164653913', '71324347#3', '-101415864#1', '71223791#1', '-679068326#3', '-101746498#0', '463830226#1', '-101765339#0', '-101787498', '101784499#1', '101418583', '-47374664', '-101749451', '47374680#2', '619185406', '-101746498#1', '-1200809291', '-282689986', '-619184994', '-101608960', '-100468681#2', '-100488715#6', '-659278038#0', '-100488715#3', '100525438#0', '101607969#1', '-101604497', '-336860316#0', '120829754', '1200809292', '659278036#2', '71223800#6', '526438862#1', '-101752975#1', '336863927#3', '101607964#0', '-101605951#3', '101594822', '-1323419242#1', '101607969#0', '-101606549', '-101605951#5', '416409192#5', '76867740', '-101765343', '-416375814', '-352797377', '-71223800#4', '-479315694', '-336860317#0', '-101606546', '101416508#1', '-101749465', '-282689981#1', '-282689975#3', '-336860317#1', '336863927#0', '1323419244', '120509990', '120511301', '-101746516#0', '101605951#0', '416405090', '-101767911', '-101752970#1', '-101754276#1', '101765339#2', '-416409191#1', '-101784793', '659281081#0', '-101749449#0', '-173532317#1']
destinations =  ['-100468681#2', '464265154', '416373452', '101606547#1', '-352797377', '-100468681#0', '-101787493', '-101752975#1', '282689983#1', '-372606529#4', '101418583', '526438862#2', '100475365#1', '336863934', '-101418584#1', '-101746498#2', '-101754276#1', '-101608959', '-1323419243', '-101787498', '101767915#1', '-101611601', '-100468675#0', '-416409191#0', '-101594836#0', '-101604513#1', '-71324343#0', '-659283000#3', '-336863927#1', '-619184994', '-101787489', '-101746516#0', '-101601832#0', '100525438#0', '-1200809291', '-101599076#0', '71324347#3', '-101749458#0', '463830226#1', '-47374664', '-1047412913', '-100525438#2', '-101415865#0', '101607963', '416428068', '-101746508#0', '-101601551', '120509990', '-100468681#1', '526438862#1', '-101418584#0', '868087112', '-1323419247', '71324343#3', '-416412034', '-71223800#1', '1047412912#0', '71223791#1', '120509991', '-120511144#2', '-101752970#2', '-282690560#0', '282479125#1', '-101600741#2', '-101752975#2', '71324347#5', '659281081#1', '-101609498#0', '1164659926', '-101608828', '-101604518', '-1323419242#1', '101608824', '101754278#1', '-23887076#2', '526438862#0', '120829754', '-101605951#5', '-416409192#7', '101417074', '-120511144#0', '-100488715#7', '-101609494#0', '-101606549', '-101767922', '-837309289#1', '-416375814', '-101606546', '-336860317#0', '-101411571#1', '1323419244', '-71223800#4', '-101746514', '-101418589']


env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,
        "agents_csv_file_name": "agents.csv",
        "num_agents" : 22,

        "human_parameters" :
        {
            "model" : "general_model",

            "noise_weight_agent" : 0,
            "noise_weight_path" : 0.8,
            "noise_weight_day" : 0.2,

            "beta" : -1,
            "beta_k_i_variability" : 0.1,
            "epsilon_i_variability" : 0.1,
            "epsilon_k_i_variability" : 0.1,
            "epsilon_k_i_t_variability" : 0.1,

            "greedy" : 0.9,
            "gamma_c" : 0.0,
            "gamma_u" : 0.0,
            "remember" : 1,

            "alpha_zero" : 0.8,
            "alphas" : [0.2]  
        },
        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents",
            "marginal_cost_coefficient_beta": 0.3,
            "records_folder": "training_records",
        }
    },
    "simulator_parameters" : {
        "network_name" : "saint_arnoult",
        "custom_network_folder" : "saint_arnoult",
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
        "origins" : origins,
        "destinations" : destinations,
        "number_of_paths" : 3,
        "beta" : -3,
        "visualize_paths" : True,        
    }
}

## Environment initialization
env = TrafficEnvironment(seed=41, create_agents=False, create_paths=False, randomize_sumo_seed=True, marginal_cost_calculation=True, marginal_cost_calculation_machine_to_all=True, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

## Initialize the connection with SUMO
env.start()
env.reset()

## Human learning phase
for episode in range(human_learning_episodes):
    env.step()

pre_mutation_agents = env.all_agents.copy()

env.mutation(mutation_start_percentile=5)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")



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

pbar = tqdm(total=total_episodes, desc="Human learning")

pbar.set_description("AV learning")
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
env.close()