#!/usr/bin/env bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Usage:
# bash eval.sh MODEL_PATH "a,b,c" N top_p temperature LEN
MODEL_PATH="$1"

# Khiem: Modify the DATATYPES so that we can loop thru
# If $2 is "aime2025,aime,amc" this makes DATATYPES=(aime2025 aime amc)
IFS=',' read -r -a DATATYPES <<< "$2"



# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"


N=$3
top_p=$4
temperature=$5
LEN=$6
# Khiem: Output folder
MODEL_NAME=$7

for DATA_TYPE in "${DATATYPES[@]}"; do
    # build file paths
    DATA_FILE="/project/phan/kt477/MAMBA-STEP/M1/rl/verl/scripts/${DATA_TYPE}.parquet"
    OUT_FILE="${MODEL_NAME}/${DATA_TYPE}_t${temperature}_n${N}_topp_${top_p}_${LEN}.parquet"

    # print to verify what we are about to run
    echo "Running dataset: ${DATA_TYPE}"
    echo "  data.file -> ${DATA_FILE}"
    echo "  out.file  -> ${OUT_FILE}"

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=1 \
        data.path="'${DATA_FILE}'" \
        data.output_path="'${OUT_FILE}'" \
        data.batch_size=16 \
        model.path="'${MODEL_PATH}'" \
        model.mamba_inference=True \
        rollout.name=hf \
        rollout.prompt_length=1024 \
        rollout.micro_batch_size=8 \
        rollout.n=${N} \
        rollout.do_sample=True \
        rollout.temperature=${temperature} \
        rollout.response_length=${LEN} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1
done
