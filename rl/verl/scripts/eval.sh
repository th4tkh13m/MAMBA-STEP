set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="$1"
# Possible values: aime2025, aime, amc, math, olympiad_bench
DATATYPES=("$2")
OUTPUT_DIR="M1-3B/"  # Add default output directory

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

N=$3
top_p=$4
temperature=$5
LEN=$6

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=/home/junxiong/M1/rl/verl/data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}_t${temperature}_n${N}_topp_${top_p}_${LEN}.parquet \
        data.batch_size=32 \
        model.path=${MODEL_PATH} \
        model.mamba_inference=True \
        rollout.name=hf \
        rollout.prompt_length=1024 \
        rollout.micro_batch_size=4 \
        rollout.n=${N} \
        rollout.do_sample=True \
        rollout.temperature=${temperature} \
        rollout.response_length=${LEN} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1
done
