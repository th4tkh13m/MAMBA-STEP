
#!/bin/bash

conda activate m1 # change to your YOUR_ENV_NAME

# Print environment information for debugging
echo "MASTER NODE HOST NAME: $MASTER_HOSTNAME"
echo "MASTER NODE ADDR: $MASTER_ADDR:$MASTER_PORT"
echo "NUM NODES: $NUM_NODES, CURRENT NODE RANK: $SLURM_NODEID"

# PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# if [ -n "$PIDS" ]; then
#     sudo kill -9 $PIDS
#     echo "Killed GPU processes: $PIDS"
# else
#     echo "No GPU processes found to kill."
# fi

# Run the accelerate launch command
accelerate launch \
    --num_machines=${NUM_NODES} \
    --num_processes=${NUM_PROCESSES} \
    --machine_rank=${SLURM_NODEID} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    -m axolotl.cli.train "${CFG}"
