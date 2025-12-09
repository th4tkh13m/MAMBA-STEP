from transformers import AutoModel

model = AutoModel.from_pretrained(
    "/scratch/phan/kt477/MAMBA-STEP/mamba_pure_prm/global_step_110/test/huggingface",
    device_map="cpu",  # or "auto" if you want GPU
)
print(model)

