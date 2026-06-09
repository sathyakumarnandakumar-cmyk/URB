#!/bin/bash
# scripts/sweep_bandit.sh

# Hyperparameter search space
LRS=(3e-4 1e-3)
ENTROPIES=(0.01 0.05 0.10)
TEMPERATURES=(0.5 1.0 2.0)
SEEDS=(42 123 7)
PARALLEL_JOBS=6

# Verify execution context
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" && -z "$SUMO_HOME" ]]; then
    echo "Warning: Ensure you are in the correct virtual environment and SUMO_HOME is set!"
fi

# Create a temporary file to store all our commands
CMD_FILE=$(mktemp)

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for ent in "${ENTROPIES[@]}"; do
      for temp in "${TEMPERATURES[@]}"; do
        
        # Format the nested run ID
        run_id="sweep_bandit/lr_${lr}_ent_${ent}_temp_${temp}_seed_${seed}"
        
        # Write the execution command to the temporary file
        echo "echo '--> Starting run: $run_id' && python scripts/bandit_reinforce.py --id \"$run_id\" --task-conf config1 --alg-conf config1 --net ingolstadt_custom --env-seed \"$seed\" --lr \"$lr\" --entropy_coeff \"$ent\" --temperature \"$temp\"" >> "$CMD_FILE"
            
      done
    done
  done
done

echo "=========================================================="
echo "Generated 54 jobs. Launching with $PARALLEL_JOBS parallel workers..."
echo "=========================================================="

# Execute commands from the file in parallel using xargs
xargs -P "$PARALLEL_JOBS" -I {} bash -c "{}" < "$CMD_FILE"

# Clean up
rm "$CMD_FILE"

echo "=========================================================="
echo "Hyperparameter Sweep Completed!"
echo "=========================================================="
