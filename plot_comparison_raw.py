import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(scalars, weight=0.9):
    """
    EMA implementation similar to TensorBoard.
    weight between 0 and 1. Higher means smoother.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_combined_raw():
    bandit_file = "results/bandit_long_run_42/losses/losses.csv"
    iql_file = "results/iql_ing_4/metrics/VectorMetrics.csv"
    output_path = "results/combined_raw_travel_times.png"

    if not os.path.exists(bandit_file):
        print(f"Error: {bandit_file} not found.")
        return
    if not os.path.exists(iql_file):
        print(f"Error: {iql_file} not found.")
        return

    # 1. Load Bandit REINFORCE values
    # In bandit_long_run_42, 'reward' is -mean(travel_times) in minutes.
    # Therefore, mean travel time (minutes) = -reward
    df_bandit = pd.read_csv(bandit_file)
    bandit_episodes = df_bandit['iteration'].values
    bandit_tt = -df_bandit['reward'].values
    bandit_smoothed = smooth(bandit_tt, weight=0.95)

    # 2. Load IQL values
    # In iql_ing_4, we only have VectorMetrics.csv which tracks avg_time_lost (seconds).
    # travel_time (seconds) = free_flow_time + avg_time_lost
    # From BenchmarkMetrics, t_pre is ~4.21 min (252.6s), and human avg_time_lost is 103.3s.
    # Thus free_flow_time ~ 149.3 seconds.
    # travel_time (minutes) = (avg_time_lost + 149.3) / 60
    df_iql = pd.read_csv(iql_file)
    iql_episodes = df_iql['episode'].values
    iql_avg_time_lost = df_iql['avg_time_lost'].values
    iql_tt = (iql_avg_time_lost + 149.3) / 60.0
    iql_smoothed = smooth(iql_tt, weight=0.95)

    # 3. Plotting
    plt.figure(figsize=(12, 7))
    
    # Plot smoothed curves with high opacity
    plt.plot(bandit_episodes, bandit_smoothed, label="Bandit REINFORCE", color="#1f77b4", linewidth=2.5)
    plt.plot(iql_episodes, iql_smoothed, label="Independent Q-Learning (IQL)", color="#ff7f0e", linewidth=2.5)
    
    # Plot raw data as faint background noise
    plt.plot(bandit_episodes, bandit_tt, color="#1f77b4", alpha=0.15, linewidth=0.5)
    plt.plot(iql_episodes, iql_tt, color="#ff7f0e", alpha=0.15, linewidth=0.5)

    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Mean Travel Time (minutes)", fontsize=14)
    plt.title("Convergence Comparison: Bandit REINFORCE vs IQL", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    print(f"Successfully generated combined curve plot: {output_path}")

if __name__ == "__main__":
    plot_combined_raw()
