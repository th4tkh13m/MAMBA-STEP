#!/bin/bash

#SBATCH --job-name=m1_sft                       # Job name
#SBATCH --nodes=PUT_YOUR_NUMBER_OF_NODE_HERE    # Number of nodes
#SBATCH --gres=gpu:8                            # Number of GPUs per node
#SBATCH --cpus-per-task=16                      # CPU cores per task
#SBATCH --nodelist=node[01-04]                  # PUT_YOUR_SLRUM_NODELIST_HERE
#SBATCH --output=m1_slurm/output_%j.log         # Standard output and error log

export MASTER_HOSTNAME=PUT_YOUR_MASTER_HOSTNAME
export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
export MASTER_PORT=9541

export NUM_NODES=$SLURM_NNODES
export NUM_PROCESSES=$(($NUM_NODES * 8))

export CFG="$@"

# Launch the training script using srun
srun bash -c 'bash ./launch_train.sh "${CFG}"'