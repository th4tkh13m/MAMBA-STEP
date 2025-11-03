#!/bin/bash
# Training script for MAMBA PURE-PRM
# Process Reward Model Only (No Verifiable Rewards)

set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Set your PRM model path here
PRM_MODEL_PATH="<PATH_TO_YOUR_PRM_MODEL>"

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
    --config-name config_pure_prm \
    reward_model.model.path=$PRM_MODEL_PATH \
    reward_model.credit_assignment=$CREDIT_TEMP \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=./checkpoints/${EXP_NAME}

echo "Training completed. Checkpoints saved to: ./checkpoints/${EXP_NAME}"
