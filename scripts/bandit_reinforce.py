import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from routerl import TrafficEnvironment
from tqdm import tqdm

from utils import clear_SUMO_files
from utils import print_agent_counts
from utils import run_metrics_analysis
from utils import save_loss_records
from utils import script_path_for_config


# ------------------------------------------------------------------
# Welford's online normalisation
# ------------------------------------------------------------------
class RunningNormalizer:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.M2 = np.zeros(shape, dtype=np.float32)

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x - self.mean
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance) + 1e-8
        return (x - self.mean) / std


# ------------------------------------------------------------------
# Neural Network Scorer
# ------------------------------------------------------------------
class RouteScorer(nn.Module):
    """
    Takes 8 features representing a route and outputs a scalar score.
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x is [4, 8] for 4 routes
        return self.net(x).squeeze(-1)  # [4]


# ------------------------------------------------------------------
# Feature Store & Extractor
# ------------------------------------------------------------------
class RouteFeatureStore:
    """
    Preloads static features from the dynamically generated routes.csv.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        
        if "path" in df.columns and "edges" not in df.columns:
            df["edges"] = df["path"]

        self.store = {}
        self.edge_store = {}

        for (o, d), group in df.groupby(["origins", "destinations"]):
            o_str, d_str = str(o), str(d)
            if (o_str, d_str) not in self.store:
                self.store[(o_str, d_str)] = {}
                self.edge_store[(o_str, d_str)] = {}

            for idx, row in enumerate(group.itertuples()):
                ff_time = getattr(row, "free_flow_time", getattr(row, "free_flow_tt", 0.0))
                length = getattr(row, "length_km", 0.0)
                mw_share = getattr(row, "motorway_share", 0.0)
                n_left = getattr(row, "n_left_turns", 0.0)

                features = np.array([
                    ff_time,
                    length,
                    mw_share,
                    n_left
                ], dtype=np.float32)

                self.store[(o_str, d_str)][idx] = features
                edges = str(row.edges).strip().split()
                self.edge_store[(o_str, d_str)][idx] = edges

    def get_static(self, origin: str, dest: str, path_idx: int) -> np.ndarray:
        return self.store[(origin, dest)][path_idx]

    def get_edges(self, origin: str, dest: str, path_idx: int) -> list:
        return self.edge_store[(origin, dest)][path_idx]


class DynamicFeatureExtractor:
    def __init__(self, traci_conn):
        self.traci = traci_conn
        self.link_cav_count = defaultdict(int)

    def get_dynamic(self, edges: list, start_time: float) -> np.ndarray:
        if not edges:
            return np.zeros(4, dtype=np.float32)

        edge_speeds = []
        edge_occ = []
        cav_load = 0

        for e in edges:
            cav_load += self.link_cav_count[e]
            if self.traci is not None:
                try:
                    speed = self.traci.edge.getLastStepMeanSpeed(e)
                    occ = self.traci.edge.getLastStepOccupancy(e)
                    edge_speeds.append(speed)
                    edge_occ.append(occ)
                except Exception:
                    pass

        mean_speed = np.mean(edge_speeds) if edge_speeds else 13.89
        mean_occ = np.mean(edge_occ) if edge_occ else 0.0
        mean_cav = cav_load / len(edges)

        max_start = 7200.0
        norm_start = start_time / max_start

        return np.array([
            mean_speed / 13.89,
            mean_occ,
            mean_cav,
            norm_start
        ], dtype=np.float32)

    def register_decision(self, edges: list):
        for e in edges:
            self.link_cav_count[e] += 1


def build_all_routes_features(agent, store, dyn_extractor, n_routes, normaliser=None):
    o_str, d_str = str(agent.origin), str(agent.destination)
    all_features = []
    
    for i in range(n_routes):
        try:
            static = store.get_static(o_str, d_str, i)
            edges = store.get_edges(o_str, d_str, i)
        except KeyError:
            static = np.zeros(4, dtype=np.float32)
            edges = []

        dynamic = dyn_extractor.get_dynamic(edges, float(agent.start_time))
        combined = np.concatenate([static, dynamic])

        if normaliser is not None:
            normaliser.update(combined)
            combined = normaliser.normalize(combined)
            
        all_features.append(combined)
        
    return torch.tensor(np.array(all_features), dtype=torch.float32)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class BanditTrainer:
    def __init__(self, env, params, records_folder):
        self.env = env
        self.n_routes = params.get("number_of_paths", 4)
        
        self.lr = params.get("lr", 3e-4)
        self.entropy_coeff = params.get("entropy_coeff", 0.05)
        self.temperature = params.get("temperature", 1.0)
        self.baseline_alpha = params.get("baseline_alpha", 0.05)
        
        routes_path = os.path.join(records_folder, "routes.csv")
        self.store = RouteFeatureStore(routes_path)
        self.normaliser = RunningNormalizer(shape=(8,))
        
        self.scorer = RouteScorer(hidden_size=params.get("hidden_size", 64))
        self.optimiser = optim.Adam(self.scorer.parameters(), lr=self.lr)
        
        self.ema_reward = None
        self.loss_history = []
        
    def _run_episode(self, explore=True):
        self.env.reset()
        try:
            # Attempt to get active TraCI connection from URB's simulator
            traci_conn = self.env.simulator.sumo_connection if hasattr(self.env, "simulator") else None
        except AttributeError:
            traci_conn = None
        self.dyn = DynamicFeatureExtractor(traci_conn)
        
        log_probs = []
        entropies = []
        travel_times = []
        
        agent_lookup = {str(a.id): a for a in self.env.machine_agents}
        
        for agent_id in self.env.agent_iter():
            obs, reward, term, trunc, info = self.env.last()
            
            if term or trunc:
                travel_times.append(-float(reward))
                self.env.step(None)
            else:
                agent = agent_lookup[agent_id]
                feats = build_all_routes_features(agent, self.store, self.dyn, self.n_routes, self.normaliser)
                
                scores = self.scorer(feats)
                scaled_scores = scores / self.temperature
                
                if explore:
                    pi = Categorical(logits=scaled_scores)
                    action = pi.sample()
                    log_probs.append(pi.log_prob(action))
                    entropies.append(pi.entropy())
                else:
                    action = torch.argmax(scores)
                    
                action_idx = action.item()
                
                origin, dest = str(agent.origin), str(agent.destination)
                edges = self.store.get_edges(origin, dest, action_idx)
                self.dyn.register_decision(edges)
                
                self.env.step(action_idx)
                
        if len(travel_times) == 0:
            return 0.0
            
        R = -float(np.mean(travel_times))
        
        if explore and len(log_probs) > 0:
            if self.ema_reward is None:
                self.ema_reward = R
            else:
                self.ema_reward = (1 - self.baseline_alpha) * self.ema_reward + self.baseline_alpha * R
                
            advantage = R - self.ema_reward
            
            log_probs_stack = torch.stack(log_probs)
            entropies_stack = torch.stack(entropies)
            mean_entropy = entropies_stack.mean()
            
            pg_loss = -(advantage * log_probs_stack.sum())
            loss = pg_loss - self.entropy_coeff * mean_entropy
            
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            
            self.loss_history.append({
                "iteration": len(self.loss_history) + 1,
                "agent_id": "fleet_shared",
                "loss": loss.item(),
                "reward": R,
                "ema_reward": self.ema_reward,
                "entropy": mean_entropy.item()
            })
            
        return R

    def train(self, n_episodes, pbar=None):
        for _ in range(n_episodes):
            self._run_episode(explore=True)
            if pbar: pbar.update()

    def test(self, n_episodes, pbar=None):
        for _ in range(n_episodes):
            self._run_episode(explore=False)
            if pbar: pbar.update()


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--entropy_coeff', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    args = parser.parse_args()
    
    ALGORITHM = "bandit_reinforce"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)

    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    if "desc" in params: del params["desc"]

    if args.lr is not None:
        params["lr"] = args.lr
    if args.entropy_coeff is not None:
        params["entropy_coeff"] = args.entropy_coeff
    if args.temperature is not None:
        params["temperature"] = args.temperature

    for key, value in params.items():
        globals()[key] = value
    
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        data = ast.literal_eval(f.read())
    origins = data['origins']
    destinations = data['destinations']

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config.update({
        "network": network,
        "env_seed": env_seed,
        "env_config": env_config,
        "task_config": task_config,
        "alg_config": alg_config,
        "script": script_path_for_config(__file__),
        "algorithm": ALGORITHM,
        "num_agents": num_agents,
        "num_machines": num_machines
    })
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : observations
            }
        },
        environment_parameters = {"save_every" : save_every},
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : 180
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "path_gen_workers" : path_gen_workers,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()

    # Mutation
    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)
    print_agent_counts(env)
    
    # Initialize BanditTrainer (reads routes.csv automatically generated by env.start())
    trainer = BanditTrainer(env, params, records_folder)
    
    ### Learning phase ###
    pbar.set_description("AV learning (Bandit REINFORCE)")
    os.makedirs(plots_folder, exist_ok=True)
    
    trainer.train(training_eps, pbar)
    
    ### Testing phase ###
    pbar.set_description("Testing (Greedy)")
    trainer.test(test_eps, pbar)
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    
    save_loss_records(
        records_folder,
        trainer.loss_history,
        columns=["iteration", "agent_id", "loss", "reward", "ema_reward", "entropy"],
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")
