#!/bin/bash
# Training script for MAMBA PURE-PRM+VR
# Hybrid: Process Reward Model + Verifiable Rewards (SOTA)

set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Set your PRM model path here
PRM_MODEL_PATH="<PATH_TO_YOUR_PRM_MODEL>"

# Credit assignment configuration
CREDIT_TEMP=0.1  # Temperature for approximate min-form
# Alternative: CREDIT_METHOD="strict min-form"  # For exact minimum

# Reward coefficients
VR_COEF=1.0   # Weight for verifiable rewards
PRM_COEF=0.5  # Weight for process rewards

# Experiment name
EXP_NAME="mamba_pure_prm_vr_vr${VR_COEF}_prm${PRM_COEF}_temp${CREDIT_TEMP}"

echo "Starting PURE-PRM+VR training with PRM model: $PRM_MODEL_PATH"
echo "Credit assignment temperature: $CREDIT_TEMP"
echo "VR coefficient: $VR_COEF, PRM coefficient: $PRM_COEF"

# Create output directory
mkdir -p ./checkpoints/${EXP_NAME}

# Train with PURE-PRM+VR configuration
python3 -m verl.trainer.main_ppo \
    --config-name config_pure_prm_vr \
    reward_model.model.path=$PRM_MODEL_PATH \
    reward_model.credit_assignment=$CREDIT_TEMP \
    reward_model.verifiable_reward_coef=$VR_COEF \
    reward_model.modeling_reward_coef=$PRM_COEF \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=./checkpoints/${EXP_NAME}

echo "Training completed. Checkpoints saved to: ./checkpoints/${EXP_NAME}"

# Example usage with different configurations:
# 
# 1. Standard PURE-PRM+VR (balanced):
#    VR_COEF=1.0 PRM_COEF=0.5 bash scripts/train_pure_prm_vr.sh
#
# 2. More weight on PRM:
#    VR_COEF=0.5 PRM_COEF=1.0 bash scripts/train_pure_prm_vr.sh
#
# 3. Equal weights:
#    VR_COEF=1.0 PRM_COEF=1.0 bash scripts/train_pure_prm_vr.sh
#
# 4. Strict min-form (edit config to set credit_assignment: 'strict min-form')
