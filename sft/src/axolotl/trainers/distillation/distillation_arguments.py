from typing import Optional
from dataclasses import dataclass, field
from axolotl.core.trainer_builder import AxolotlTrainingArguments

@dataclass
class DistillationTrainingArguments(AxolotlTrainingArguments):
    distillation: bool = field(
        default=False,
        metadata={"help": "whether to use distillation"}
    )
    teacher_model: str = field(
        default=None,
        metadata={"help": "path to the target model for distillation"}
    )
    distillation_loss_type: Optional[str] = field(
        default="reverse_kl",
        metadata={"help": "distillation loss type to use"},
    )
    kl_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "kl weight to use"},
    )
