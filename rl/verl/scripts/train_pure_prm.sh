#!/bin/bash
# Training script for MAMBA PURE-PRM
# Process Reward Model Only (No Verifiable Rewards)

set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
export VLLM_ATTENTION_BACKEND=XFORMERS
BASEDIR="/project/phan/kt477/MAMBA-STEP/M1/rl/verl/scripts"
# Set your PRM model path here
PRM_MODEL_PATH="jinachris/PURE-PRM-7B"

# Temperature for approximate min-form credit assignment
# Lower values (~0.1) -> sharper weighting on worst step
# Higher values (~1.0) -> more uniform weighting
CREDIT_TEMP=0.1

# Experiment name
EXP_NAME="mamba_pure_prm_temp${CREDIT_TEMP}"

echo "Starting PURE-PRM training with PRM model: $PRM_MODEL_PATH"
echo "Credit assignment temperature: $CREDIT_TEMP"

# Create output directory
mkdir -p ./checkpoints/${EXP_NAME}

# Train with PURE-PRM configuration
python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR} \
    --config-name config_pure_prm 

echo "Training completed. Checkpoints saved to: ./checkpoints/${EXP_NAME}"
