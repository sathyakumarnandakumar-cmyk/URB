"""
Contextual Bandit REINFORCE for URB – Ingolstadt
=================================================
Drop-in replacement for the "User defined AV learning pipeline" section
in base_script.py.

Assumes:
  - env is a TrafficEnvironment after mutation (machine_agents populated)
  - paths.csv exists in records_folder with columns:
        origin, destination, path_id, edges (space-separated edge IDs),
        free_flow_tt, length_km, motorway_share, n_left_turns
  - SUMO/TraCI is accessible via env.simulator.sumo (traci connection)
  - Each machine agent has: .id, .start_time, .origin, .destination,
    .action_space_size (== number of routes, e.g. 4)
  - env.step(machine_action) steps one agent and returns
  - After all agents step, episode rewards come from ep CSV or agent.last_reward

Usage (in your experiment script after mutation):
    from bandit_reinforce import BanditTrainer
    trainer = BanditTrainer(env, params, records_folder)
    trainer.train(n_episodes=500)
    trainer.test(n_episodes=50)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import defaultdict
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Static route features  (loaded once from paths.csv)
# ---------------------------------------------------------------------------

class RouteFeatureStore:
    """
    Holds precomputed static features for every (origin, destination, path_id).
    Provides fast lookup at decision time.
    """

    STATIC_COLS = ["free_flow_tt", "length_km", "motorway_share", "n_left_turns"]

    def __init__(self, paths_csv: str):
        df = pd.read_csv(paths_csv)
        # Normalise column names to lowercase / underscore
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Build edge list per route for TraCI queries
        # Expect column 'edges' with space-separated edge IDs
        self.edges = {}       # (origin, dest, path_id) -> list[str]
        self.static = {}      # (origin, dest, path_id) -> np.array shape (4,)
        self.routes_for = defaultdict(list)  # (origin, dest) -> [path_id, ...]

        for _, row in df.iterrows():
            key = (row["origin"], row["destination"], int(row["path_id"]))
            od  = (row["origin"], row["destination"])

            # Edge list (may be missing – graceful fallback)
            if "edges" in df.columns and pd.notna(row.get("edges", None)):
                self.edges[key] = str(row["edges"]).split()
            else:
                self.edges[key] = []

            # Static feature vector
            feats = np.array([row.get(c, 0.0) for c in self.STATIC_COLS],
                             dtype=np.float32)
            self.static[key] = feats

            path_id = int(row["path_id"])
            if path_id not in self.routes_for[od]:
                self.routes_for[od].append(path_id)

        # Sort path_ids so index == action
        for od in self.routes_for:
            self.routes_for[od].sort()

    def get_static(self, origin, destination, path_id) -> np.ndarray:
        key = (origin, destination, int(path_id))
        return self.static.get(key, np.zeros(4, dtype=np.float32))

    def get_edges(self, origin, destination, path_id) -> list:
        key = (origin, destination, int(path_id))
        return self.edges.get(key, [])

    def n_routes(self, origin, destination) -> int:
        return len(self.routes_for[(origin, destination)])


# ---------------------------------------------------------------------------
# 2. Dynamic feature extraction (TraCI + fleet memory)
# ---------------------------------------------------------------------------

class DynamicFeatureExtractor:
    """
    At each agent departure computes:
      - mean_speed_ratio  : mean(current_speed / speed_limit) over route edges
      - mean_occupancy    : mean edge occupancy over route edges
      - mean_cav_link_load: mean committed-CAV traffic on route edges (fleet mem)
      - dep_time_norm     : departure time / max_departure_time

    Fleet memory is a dict {edge_id: int} counting how many CAVs already
    committed to routes containing that edge this episode.
    """

    def __init__(self, traci_conn, max_departure_time: float,
                 route_store: RouteFeatureStore):
        self.traci = traci_conn
        self.max_dep_time = max(max_departure_time, 1.0)
        self.store = route_store
        self.reset_episode()

    def reset_episode(self):
        """Call at the start of every episode before agents decide."""
        self.link_cav_count = defaultdict(int)  # edge_id -> n_cavs committed

    def register_decision(self, origin, destination, path_id):
        """Call after each CAV decides to update fleet memory."""
        edges = self.store.get_edges(origin, destination, path_id)
        for e in edges:
            self.link_cav_count[e] += 1

    def get_dynamic(self, origin, destination, path_id,
                    dep_time: float) -> np.ndarray:
        """
        Returns np.array of shape (4,):
          [mean_speed_ratio, mean_occupancy, mean_cav_link_load, dep_time_norm]
        """
        edges = self.store.get_edges(origin, destination, path_id)

        if edges and self.traci is not None:
            speed_ratios, occupancies, cav_loads = [], [], []
            for e in edges:
                try:
                    max_spd = self.traci.edge.getMaxSpeed(e)
                    cur_spd = self.traci.edge.getLastStepMeanSpeed(e)
                    occ     = self.traci.edge.getLastStepOccupancy(e)
                    ratio   = (cur_spd / max_spd) if max_spd > 0 else 1.0
                    speed_ratios.append(ratio)
                    occupancies.append(occ)
                    cav_loads.append(self.link_cav_count.get(e, 0))
                except Exception:
                    speed_ratios.append(1.0)
                    occupancies.append(0.0)
                    cav_loads.append(0)

            mean_sr   = float(np.mean(speed_ratios))
            mean_occ  = float(np.mean(occupancies))
            mean_cav  = float(np.mean(cav_loads))
        else:
            # No TraCI or no edge info – neutral defaults
            mean_sr, mean_occ, mean_cav = 1.0, 0.0, 0.0

        dep_norm = dep_time / self.max_dep_time

        return np.array([mean_sr, mean_occ, mean_cav, dep_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Feature assembly
# ---------------------------------------------------------------------------

N_STATIC  = 4   # free_flow_tt, length_km, motorway_share, n_left_turns
N_DYNAMIC = 4   # mean_speed_ratio, mean_occupancy, mean_cav_link_load, dep_norm
N_FEATURES_PER_ROUTE = N_STATIC + N_DYNAMIC   # = 8

def build_route_features(agent, path_id: int,
                          store: RouteFeatureStore,
                          dyn: DynamicFeatureExtractor) -> np.ndarray:
    """Returns feature vector of shape (8,) for one route."""
    static  = store.get_static(agent.origin, agent.destination, path_id)
    dynamic = dyn.get_dynamic(agent.origin, agent.destination, path_id,
                              agent.start_time)
    return np.concatenate([static, dynamic])   # (8,)


def build_all_routes_features(agent, store: RouteFeatureStore,
                               dyn: DynamicFeatureExtractor,
                               n_routes: int = 4) -> torch.Tensor:
    """
    Returns tensor of shape (n_routes, 8) – one row per route alternative.
    Routes are ordered by path_id (== action index).
    """
    od = (agent.origin, agent.destination)
    path_ids = store.routes_for[od]          # sorted list of path_ids
    rows = []
    for pid in path_ids:
        rows.append(build_route_features(agent, pid, store, dyn))
    # Pad if agent has fewer routes than max (shouldn't happen with fixed 4)
    while len(rows) < n_routes:
        rows.append(np.zeros(N_FEATURES_PER_ROUTE, dtype=np.float32))
    return torch.tensor(np.stack(rows[:n_routes]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Shared scorer network
# ---------------------------------------------------------------------------

class RouteScorer(nn.Module):
    """
    Scores ONE route given its feature vector.
    Applied independently to each of the k route alternatives,
    then softmax gives the routing policy.

    Input:  (batch, 8)
    Output: (batch, 1)  – scalar score
    """

    def __init__(self, n_features: int = N_FEATURES_PER_ROUTE,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (batch,) or scalar

    def score_routes(self, route_features: torch.Tensor) -> torch.Tensor:
        """
        route_features: (n_routes, 8)
        Returns: (n_routes,) scores
        """
        return self.forward(route_features)


# ---------------------------------------------------------------------------
# 5. Normaliser (running mean/std per feature)
# ---------------------------------------------------------------------------

class RunningNormaliser:
    """Online feature normalisation – updates incrementally."""

    def __init__(self, n_features: int):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2   = np.ones(n_features,  dtype=np.float64)

    def update(self, x: np.ndarray):
        """x shape: (n_routes, n_features) or (n_features,)"""
        x = x.reshape(-1, self.mean.shape[0])
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (row - self.mean)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.M2 / max(self.n, 1)) + 1e-8
        return ((x - self.mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Trainer
# ---------------------------------------------------------------------------

class BanditTrainer:
    """
    REINFORCE trainer for the contextual bandit routing problem.

    Parameters (via params dict):
      lr              : learning rate (default 3e-4)
      entropy_coeff   : entropy bonus coefficient β (default 0.05)
      baseline_alpha  : EMA smoothing for baseline (default 0.05)
      temperature     : softmax temperature (default 1.0)
      hidden_size     : MLP hidden units (default 64)
      n_routes        : routes per OD pair (default 4)
      save_every      : save checkpoint every N episodes (default 50)
    """

    def __init__(self, env, params: dict, records_folder: str):
        self.env = env
        self.records_folder = records_folder
        os.makedirs(records_folder, exist_ok=True)

        # Hyperparameters
        self.lr            = params.get("lr", 3e-4)
        self.beta          = params.get("entropy_coeff", 0.05)
        self.bl_alpha      = params.get("baseline_alpha", 0.05)
        self.temperature   = params.get("temperature", 1.0)
        self.hidden        = params.get("hidden_size", 64)
        self.n_routes      = params.get("n_routes", 4)
        self.save_every    = params.get("save_every", 50)

        # Route feature store
        paths_csv = os.path.join(records_folder, "paths.csv")
        self.store = RouteFeatureStore(paths_csv)

        # TraCI connection (may be None if not available)
        try:
            traci_conn = env.simulator.sumo
        except AttributeError:
            traci_conn = None

        max_dep = max(a.start_time for a in env.machine_agents)
        self.dyn = DynamicFeatureExtractor(traci_conn, max_dep, self.store)

        # Normaliser
        self.normaliser = RunningNormaliser(N_FEATURES_PER_ROUTE)

        # Model + optimiser
        self.scorer    = RouteScorer(N_FEATURES_PER_ROUTE, self.hidden)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=self.lr)

        # Baseline (EMA of episode reward)
        self.baseline = None

        # Logging
        self.history = []   # list of dicts per episode

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def _run_episode(self, training: bool = True):
        """
        Runs one episode. Returns (mean_cav_tt, log dict).
        If training=True, also accumulates gradients.
        """
        self.dyn.reset_episode()

        agents = sorted(self.env.machine_agents, key=lambda a: a.start_time)

        log_probs   = []   # one per agent
        entropies   = []
        feat_buffer = []   # for normaliser update

        # ---- Decision phase ----
        actions = {}
        for agent in agents:
            feats = build_all_routes_features(
                agent, self.store, self.dyn, self.n_routes
            )   # (n_routes, 8)

            # Normalise
            feats_np = feats.numpy()
            self.normaliser.update(feats_np)
            feats_norm = torch.tensor(
                self.normaliser.normalise(feats_np), dtype=torch.float32
            )
            feat_buffer.append(feats_norm)

            # Score routes
            with torch.set_grad_enabled(training):
                scores = self.scorer.score_routes(feats_norm)   # (n_routes,)
                scores = scores / self.temperature
                dist   = Categorical(logits=scores)
                action = dist.sample()

            actions[agent.id] = action.item()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            # Register decision in fleet memory
            od = (agent.origin, agent.destination)
            path_ids = self.store.routes_for[od]
            chosen_pid = path_ids[action.item()] if action.item() < len(path_ids) else 0
            self.dyn.register_decision(agent.origin, agent.destination, chosen_pid)

            # Step the environment for this agent
            self.env.step(machine_action=action.item())

        # ---- Reward ----
        # Collect individual travel times from agent rewards
        travel_times = []
        for agent in agents:
            # last_reward is negative travel time in RouteRL
            tt = -float(agent.last_reward)
            travel_times.append(tt)

        mean_tt = float(np.mean(travel_times))
        R = -mean_tt   # higher is better

        # ---- REINFORCE update ----
        if training:
            # Initialise baseline on first episode
            if self.baseline is None:
                self.baseline = R

            advantage = R - self.baseline

            # Update baseline (EMA)
            self.baseline = (1 - self.bl_alpha) * self.baseline + self.bl_alpha * R

            log_prob_sum = torch.stack(log_probs).sum()
            entropy_sum  = torch.stack(entropies).sum()

            loss = -(advantage * log_prob_sum) - self.beta * entropy_sum

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.scorer.parameters(), max_norm=1.0)
            self.optimiser.step()
        else:
            loss = torch.tensor(0.0)
            advantage = 0.0

        log = {
            "mean_cav_tt": mean_tt,
            "reward": R,
            "baseline": self.baseline if self.baseline is not None else R,
            "advantage": advantage,
            "loss": loss.item() if training else 0.0,
            "mean_entropy": float(torch.stack(entropies).mean().item()),
        }
        return mean_tt, log

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int):
        pbar = tqdm(total=n_episodes, desc="Bandit training")

        for ep in range(n_episodes):
            mean_tt, log = self._run_episode(training=True)
            log["episode"] = ep
            log["phase"] = "train"
            self.history.append(log)

            pbar.set_postfix({
                "t_CAV": f"{mean_tt:.1f}s",
                "baseline": f"{self.baseline:.1f}",
                "H": f"{log['mean_entropy']:.2f}",
            })
            pbar.update()

            if (ep + 1) % self.save_every == 0:
                self._save_checkpoint(ep)
                self._save_history()

        pbar.close()
        self._save_checkpoint(n_episodes - 1, tag="final")
        self._save_history()
        print(f"Training done. Final mean t_CAV: {self.history[-1]['mean_cav_tt']:.2f}s")

    # ------------------------------------------------------------------
    # Test loop (greedy: argmax instead of sample)
    # ------------------------------------------------------------------

    def test(self, n_episodes: int):
        # Switch to deterministic (temperature → 0 approximated by argmax)
        original_temperature = self.temperature
        self.temperature = 0.01   # near-greedy

        pbar = tqdm(total=n_episodes, desc="Bandit testing")
        test_tts = []

        for ep in range(n_episodes):
            mean_tt, log = self._run_episode(training=False)
            log["episode"] = ep
            log["phase"] = "test"
            self.history.append(log)
            test_tts.append(mean_tt)
            pbar.set_postfix({"t_CAV": f"{mean_tt:.1f}s"})
            pbar.update()

        pbar.close()
        self.temperature = original_temperature
        print(f"Test mean t_CAV: {np.mean(test_tts):.2f}s  "
              f"(std {np.std(test_tts):.2f}s)")
        self._save_history()
        return test_tts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, episode: int, tag: str = ""):
        fname = f"bandit_scorer_ep{episode}{('_'+tag) if tag else ''}.pt"
        path  = os.path.join(self.records_folder, fname)
        torch.save({
            "episode":    episode,
            "model":      self.scorer.state_dict(),
            "optimiser":  self.optimiser.state_dict(),
            "baseline":   self.baseline,
            "normaliser_mean": self.normaliser.mean,
            "normaliser_M2":   self.normaliser.M2,
            "normaliser_n":    self.normaliser.n,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.scorer.load_state_dict(ckpt["model"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.baseline = ckpt["baseline"]
        self.normaliser.mean = ckpt["normaliser_mean"]
        self.normaliser.M2   = ckpt["normaliser_M2"]
        self.normaliser.n    = ckpt["normaliser_n"]
        print(f"Loaded checkpoint from episode {ckpt['episode']}")

    def _save_history(self):
        path = os.path.join(self.records_folder, "bandit_history.csv")
        pd.DataFrame(self.history).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 7. Convenience: algo config JSON structure
# ---------------------------------------------------------------------------
# Create config/algo_config/bandit/config1.json with:
# {
#   "desc": "Contextual bandit REINFORCE baseline",
#   "lr": 0.0003,
#   "entropy_coeff": 0.05,
#   "baseline_alpha": 0.05,
#   "temperature": 1.0,
#   "hidden_size": 64,
#   "n_routes": 4,
#   "save_every": 50
# }


# ---------------------------------------------------------------------------
# 8. Example: how to wire into base_script.py
# ---------------------------------------------------------------------------
# After env.mutation(...) and print_agent_counts(env):
#
#   from bandit_reinforce import BanditTrainer
#
#   trainer = BanditTrainer(env, params, records_folder)
#   trainer.train(n_episodes=learning_episodes)
#   trainer.test(n_episodes=test_episodes)
#
# That's it. The rest of base_script.py (plot_results, stop_simulation,
# clear_SUMO_files, run_metrics_analysis) runs unchanged.
