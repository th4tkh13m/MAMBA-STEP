from typing import Dict
import logging
import torch
import deepspeed
from copy import deepcopy

from axolotl.core.trainer_builder import AxolotlTrainer
from axolotl.utils.models import load_model
from axolotl.utils.dict import DictDefault

import torch
import torch.nn.functional as F

LOG = logging.getLogger("axolotl.trainers.distillation.distillation_trainer")

def calculate_distillation_loss(logits, teacher_logits, labels, loss_type="forward_kl"):
    if loss_type == "forward_kl":
        return forward_kl(logits, teacher_logits, labels)
    elif loss_type == "reverse_kl":
        return reverse_kl(logits, teacher_logits, labels)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Supported types are: 'forward_kl', 'reverse_kl'")

def forward_kl(logits, teacher_logits, labels):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (labels != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, labels):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (labels != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

# Adapted from https://github.com/jxiw/MambaInLlama/blob/main/trainer/kd_trainer.py
class DistillationTrainer(AxolotlTrainer):
    def __init__(self, teacher_model, **kwargs):
        super().__init__(**kwargs)

        self.teacher_model = teacher_model
        self.kl_weight = self.args.kl_weight
        self.distillation_loss_type = self.args.distillation_loss_type
        self.model_type = teacher_model.config.model_type

        # prepare the target model for deepspeed
        if self.teacher_model is not None:
            if self.is_deepspeed_enabled:
                # for qwen math 1.5B, we find a bug when using deepspeed for transfomer < 4.45.0
                self.teacher_model = self._prepare_deepspeed(self.teacher_model)
            else:
                self.teacher_model = self.accelerator.prepare_model(
                    self.teacher_model, evaluation_mode=True
                )

        self._train_eval = "train"

    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        student_model = model
        teacher_model = self.teacher_model

        student_outputs = student_model(**inputs)
        student_logits = student_outputs.logits
        student_loss = student_outputs.loss

        if self.kl_weight == 0.0:
            return student_outputs.loss

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_loss = teacher_outputs.loss.item()
                teacher_logits = teacher_outputs.logits

        # Calculate the distillation loss
        distillation_loss = calculate_distillation_loss(
            student_logits, teacher_logits, inputs["labels"], self.distillation_loss_type
        )

        loss = (1 - self.kl_weight) * student_loss + self.kl_weight * distillation_loss

        if self._train_eval:
            metrics_prefix = ""
        else:
            metrics_prefix = "eval_"

        metrics = {
            f"{metrics_prefix}teacher_lm_loss": teacher_loss,
            f"{metrics_prefix}student_lm_loss": student_loss.item(),
            f"{metrics_prefix}distill_loss": distillation_loss.item(),
        }

        self.store_metrics(metrics, train_eval=self._train_eval)

        return (loss, student_outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        self._train_eval = "train"
        return super().train(*args, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        self._train_eval = "eval"
        return super().evaluation_loop(*args, **kwargs)

    @staticmethod
    def update_trainer_kwargs(builder) -> Dict:
        trainer_kwargs = {}
        if builder.cfg.distillation:
            tokenizer = builder.tokenizer

            LOG.info("Loading distillation target model...")
            if builder.cfg.teacher_model is not None:
                teacher_model, _ = load_model(
                    DictDefault(
                        base_model=builder.cfg.teacher_model,
                        model_type=builder.cfg.model_type,
                        tokenizer_type=builder.cfg.tokenizer_type,
                        load_in_8bit=builder.cfg.load_in_8bit,
                        sequence_len=builder.cfg.sequence_len,
                        device_map=builder.cfg.device_map,
                        flash_attention=builder.cfg.flash_attention,
                    ), 
                    tokenizer
                )
                teacher_model.eval()
                # disable the gradient for target model
                for param in teacher_model.parameters():
                    param.requires_grad = False

                trainer_kwargs["teacher_model"] = teacher_model
            else:
                LOG.info("Distillation target model not found, Assuming to use offline logits for distillation.")
                trainer_kwargs["teacher_model"] = None

        return trainer_kwargs

    @staticmethod
    def update_training_args_kwargs(builder):
        cfg = builder.cfg
        if cfg.sample_packing:
            assert cfg.flash_attention, "Sample packing requires flash attention, please set `flash_attention=True`"
    
        trainer_kwargs = {}
        if cfg.distillation:
            trainer_kwargs["kl_weight"] = cfg.kl_weight
            trainer_kwargs["distillation_loss_type"] = cfg.distillation_loss_type
        return trainer_kwargs