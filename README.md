# SymbolicAI class project

Wulver environment:

```bash
export PIP_NO_CACHE_DIR=off
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load CUDA/12.6.0
module load foss/2024a
```

Setup environment:

```bash
 conda create -p ./m1_pack python=3.10
 conda activate ./m1_pack
 
 # Install patched mamba
 pip install numpy
 pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
 git clone --branch varlen_mamba https://github.com/jxiw/varlen_mamba.git
 cd varlen_mamba/
 python setup.py install
 
 # Install conv1d (Compile since getting from pip is errornous)
 cd ..
 git clone https://github.com/Dao-AILab/causal-conv1d.git
 cd causal-conv1d/
 python setup.py install
 
 # Install training reqs
 cd ..
 git clone https://github.com/jxiw/M1.git
 cd M1/
 cd rl/verl/
 pip install -e .
 pip install -r deepscaler-requirements.txt
 pip install vllm==0.6.3.post1
 
 # Install flash attention 2.74post1 torch 2.4 py10
 #wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
 #   pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
 
 git clone https://github.com/Dao-AILab/flash-attention.git
 cd  flash-attention
 pip install flash-attn --no-build-isolation
 
 python tests/pack_mamba/test_mamba_layer.py
 python tests/pack_mamba/test_pack_hybrid.py
 pip install ray[default]=4.3.0
```

Take a look here also: https://github.com/jxiw/M1/issues/7

```jsx
bash eval.sh /scratch/phan/kt477/MAMBA-STEP/grpo_cg_packfix_4_8192_1_3e-6_0.01_0_0.9_2_new_veRL/global_step_290/actor/huggingface "aime2025,aime,amc,math,olympiad_bench" 5 1 0.7 24576 M1
bash eval.sh /scratch/phan/kt477/MAMBA-STEP/checkpoints/mamba_pure_prm-new/global_step_105/actor/huggingface "math,olympiad_bench" 5 1 0.7 24576 mamba-pure-new

bash eval.sh /scratch/phan/kt477/MAMBA-STEP/grpo_cg_packfix_4_8192_1_3e-6_0.01_0_0.9_2_new_veRL/global_step_290/actor/huggingface "amc,math,olympiad_bench" 5 1 0.7 24576 M1

```