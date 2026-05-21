#!/usr/bin/env python3
"""
Generate and submit Slurm jobs sweeping bandit hyperparameters.
Run from the URB/scripts/ directory:

    python submit_bandit_sweep.py --dry-run    # just print commands
    python submit_bandit_sweep.py              # actually sbatch
"""

import argparse
import itertools
import json
import os
import subprocess
import textwrap

# ── Sweep grid ────────────────────────────────────────────────────────────────
GRID = {
    "lr":            [3e-4, 1e-3],
    "entropy_coeff": [0.01, 0.05, 0.10],
    "temperature":   [0.5,  1.0,  2.0],
}
SEEDS = [42, 123, 7]

# Fixed settings (not swept)
FIXED = {
    "baseline_alpha": 0.05,
    "hidden_size":    64,
    "n_routes":       4,
    "save_every":     50,
}

NETWORK          = "ingolstadt"
ENV_CONF         = "config1"
TASK_CONF        = "task1"
LEARNING_EPS     = 500
TEST_EPS         = 50
PARTITION        = "gpu"         # adjust to your cluster
TIME             = "12:00:00"
MEM              = "8G"
CPUS_PER_TASK    = 4
ALGO_CONF_DIR    = "../config/algo_config/bandit"
SCRIPT           = "bandit_script.py"   # your filled-in base_script copy


def make_config(combo: dict) -> dict:
    cfg = {**FIXED, **combo,
           "desc": f"bandit sweep lr={combo['lr']} β={combo['entropy_coeff']} τ={combo['temperature']}"}
    return cfg


def exp_id(combo: dict, seed: int) -> str:
    return (f"bandit_ing"
            f"_lr{combo['lr']:.0e}"
            f"_ent{combo['entropy_coeff']}"
            f"_tau{combo['temperature']}"
            f"_s{seed}")


def slurm_script(exp: str, conf_path: str, seed: int) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={exp}
        #SBATCH --partition={PARTITION}
        #SBATCH --time={TIME}
        #SBATCH --mem={MEM}
        #SBATCH --cpus-per-task={CPUS_PER_TASK}
        #SBATCH --output=../results/{exp}/slurm_%j.out
        #SBATCH --error=../results/{exp}/slurm_%j.err

        mkdir -p ../results/{exp}

        python {SCRIPT} \\
            --id {exp} \\
            --alg-conf {os.path.basename(conf_path).replace('.json','')} \\
            --env-conf {ENV_CONF} \\
            --task-conf {TASK_CONF} \\
            --net {NETWORK} \\
            --env-seed {seed} \\
            --learning-episodes {LEARNING_EPS} \\
            --test-episodes {TEST_EPS}
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(ALGO_CONF_DIR, exist_ok=True)

    keys   = list(GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]

    submitted = 0
    for combo in combos:
        for seed in SEEDS:
            eid  = exp_id(combo, seed)
            cfg  = make_config(combo)
            conf_name = f"{eid}.json"
            conf_path = os.path.join(ALGO_CONF_DIR, conf_name)

            # Write algo config
            with open(conf_path, "w") as f:
                json.dump(cfg, f, indent=2)

            # Write & submit slurm script
            slurm_txt = slurm_script(eid, conf_path, seed)
            slurm_path = f"/tmp/{eid}.sh"
            with open(slurm_path, "w") as f:
                f.write(slurm_txt)

            if args.dry_run:
                print(f"[DRY] sbatch {slurm_path}  ({eid})")
            else:
                result = subprocess.run(["sbatch", slurm_path],
                                        capture_output=True, text=True)
                print(result.stdout.strip() or result.stderr.strip())

            submitted += 1

    print(f"\n{'Would submit' if args.dry_run else 'Submitted'} "
          f"{submitted} jobs  "
          f"({len(combos)} combos × {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
