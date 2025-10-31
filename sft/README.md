**Requirements**: Python >=3.10 and Pytorch >=2.1.1.

```bash
pip install -e .
pip install flash-attn==2.7.2.post1
pip install deepspeed==0.14.4
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

### Usage

Distillation over OpenMathInstruct

```bash
accelerate launch -m axolotl.cli.train math_config/distill.yaml
```

This produces a model similar to [this](https://huggingface.co/JunxiongWang/MambaInLlama3B_Distill_MATH)

SFT over OpenMathInstruct

```bash
accelerate launch -m axolotl.cli.train math_config/sft.yaml
```

This produces a model similar to [this](https://huggingface.co/JunxiongWang/MambaInLlama3B_SFT_MATH)

SFT over Reasoning

```bash
accelerate launch -m axolotl.cli.train reason_config/sft.yaml
```

This produces a model similar to [this](https://huggingface.co/JunxiongWang/M1-3B-SFT)

If you want to build your dataset, please refer `tokenized_dataset.py`, and you need to include a README.md for metadata like [this](https://huggingface.co/datasets/JunxiongWang/R1_Sythetic_SFT/blob/main/README.md).

Typically, data processing requires more than 30 minutes, so you might need to increase the `ddp_timeout` setting in the YAML configuration.

Check `multi_nodes/` if you want to run over multiple nodes over slurm.

Most of code is copied from [here](https://github.com/axolotl-ai-cloud/axolotl)