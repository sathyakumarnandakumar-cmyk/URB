import pandas as pd
import glob
import os

results = []
for f in glob.glob("results/sweep_bandit/*/metrics/BenchmarkMetrics.csv"):
    try:
        df = pd.read_csv(f)
        if df.empty: continue
        
        # Extract hyperparameters from folder name
        # Example: lr_3e-4_ent_0.01_temp_0.5_seed_42
        folder = os.path.basename(os.path.dirname(os.path.dirname(f)))
        parts = folder.split('_')
        lr = parts[1]
        ent = parts[3]
        temp = parts[5]
        seed = parts[7]
        
        row = df.iloc[0]
        results.append({
            'lr': lr,
            'ent': ent,
            'temp': temp,
            'seed': seed,
            't_CAV': row['t_CAV'],
            't_test': row['t_test']
        })
    except Exception as e:
        print(f"Error reading {f}: {e}")

df_results = pd.DataFrame(results)
print(f"Loaded {len(df_results)} successfully generated metrics files out of 54.")

# Group by hyperparameters and calculate mean over seeds
grouped = df_results.groupby(['lr', 'ent', 'temp']).agg(
    avg_t_CAV=('t_CAV', 'mean'),
    avg_t_test=('t_test', 'mean'),
    seeds_completed=('seed', 'count')
).reset_index()

# Sort by best (lowest) CAV travel time
grouped = grouped.sort_values('avg_t_CAV')

print("\n--- TOP 5 CONFIGURATIONS (Ranked by lowest CAV Travel Time) ---")
print(grouped.head(5).to_string(index=False))

print("\n--- BOTTOM 5 CONFIGURATIONS ---")
print(grouped.tail(5).to_string(index=False))

