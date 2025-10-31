
import math
from pathlib import Path
import sys
import importlib

import torch
from accelerate.logging import get_logger
import transformers
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.core.trainer_builder import AxolotlTrainingArguments
from axolotl.core.trainer_builder import HFCausalTrainerBuilder, ReLoRATrainer, AxolotlTrainer, AxolotlMambaTrainer


LOG = get_logger("axolotl")

from .distillation import DistillationTrainer, DistillationTrainingArguments

class DistillationTrainerBuilder(HFCausalTrainerBuilder):
    """
    To add new trainer, please add it in the `_get_trainer_cls`, `_get_training_arguments_cls`, and `hook_pre_create_trainer` method.
    """
    
    def _get_trainer_cls(self):
        if self.cfg.relora_steps:
            return ReLoRATrainer
        if self.cfg.model_config_type == "mamba":
            return AxolotlMambaTrainer
        if self.cfg.distillation:
            return DistillationTrainer
        return AxolotlTrainer
    
    def _get_training_arguments_cls(self):
        if self.cfg.distillation:
            return DistillationTrainingArguments
        return AxolotlTrainingArguments

    def hook_pre_create_trainer(self, trainer_kwargs, trainer_cls):
        if self.cfg.distillation:
            new_trainer_kwargs = DistillationTrainer.update_trainer_kwargs(self)
            trainer_kwargs.update(new_trainer_kwargs)
        return trainer_kwargs, trainer_cls
    
    def hook_pre_create_training_args(self, training_args_kwargs):
        if self.cfg.distillation:
            # DistillationTrainingArguments
            new_training_args_kwargs = DistillationTrainer.update_training_args_kwargs(self)
            training_args_kwargs.update(new_training_args_kwargs)

        # AxolotlTrainingArguments
        return training_args_kwargs

    def build(self, total_num_steps):
        warmup_steps = None
        if self.cfg.warmup_steps is not None:
            warmup_steps = self.cfg.warmup_steps
        elif self.cfg.warmup_ratio is not None:
            warmup_steps = max(int(self.cfg.warmup_ratio * total_num_steps), 0)
        else:
            warmup_steps = min(int(0.03 * total_num_steps), 100)
        if warmup_steps == 1:
            warmup_steps = 2

        logging_steps = (
            self.cfg.logging_steps
            if self.cfg.logging_steps is not None
            else max(min(int(0.005 * total_num_steps), 10), 1)
        )

        training_arguments_kwargs = {}
        if self.cfg.bf16 == "full":
            training_arguments_kwargs["bf16_full_eval"] = True
        else:
            training_arguments_kwargs["bf16"] = self.cfg.bf16
        training_arguments_kwargs["fp16"] = (
            self.cfg.fp16 and not self.cfg.bf16
        ) or False
        training_arguments_kwargs["tf32"] = self.cfg.tf32
        training_arguments_kwargs["warmup_steps"] = warmup_steps
        training_arguments_kwargs["logging_steps"] = logging_steps

        if self.cfg.seed:
            training_arguments_kwargs["seed"] = self.cfg.seed

        if self.cfg.gradient_checkpointing:
            training_arguments_kwargs[
                "gradient_checkpointing"
            ] = self.cfg.gradient_checkpointing
            if self.cfg.gradient_checkpointing_kwargs is not None:
                training_arguments_kwargs[
                    "gradient_checkpointing_kwargs"
                ] = self.cfg.gradient_checkpointing_kwargs
        if self.cfg.fsdp:
            training_arguments_kwargs["fsdp"] = self.cfg.fsdp
            if self.cfg.fsdp_config:
                training_arguments_kwargs["fsdp_config"] = {
                    k.lstrip("fsdp_"): v for k, v in dict(self.cfg.fsdp_config).items()
                }

        if self.cfg.adapter == "qlora":
            training_arguments_kwargs["qlora"] = True

        # deepspeed
        if self.cfg.deepspeed:
            training_arguments_kwargs["deepspeed"] = self.cfg.deepspeed

        if self.cfg.lr_quadratic_warmup is not None:
            training_arguments_kwargs[
                "lr_quadratic_warmup"
            ] = self.cfg.lr_quadratic_warmup

        if self.cfg.adam_beta1:
            training_arguments_kwargs["adam_beta1"] = self.cfg.adam_beta1
        if self.cfg.adam_beta2:
            training_arguments_kwargs["adam_beta2"] = self.cfg.adam_beta2
        if self.cfg.adam_epsilon:
            training_arguments_kwargs["adam_epsilon"] = self.cfg.adam_epsilon
        if self.cfg.max_grad_norm:
            training_arguments_kwargs["max_grad_norm"] = self.cfg.max_grad_norm

        if self.cfg.hub_model_id:
            training_arguments_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_arguments_kwargs["push_to_hub"] = True
            training_arguments_kwargs["hub_private_repo"] = True
            training_arguments_kwargs["hub_always_push"] = True

            if self.cfg.hub_strategy:
                training_arguments_kwargs["hub_strategy"] = self.cfg.hub_strategy

        if self.cfg.save_safetensors is not None:
            training_arguments_kwargs["save_safetensors"] = self.cfg.save_safetensors

        if self.cfg.dataloader_pin_memory is not None:
            training_arguments_kwargs[
                "dataloader_pin_memory"
            ] = self.cfg.dataloader_pin_memory
        if self.cfg.dataloader_num_workers is not None:
            training_arguments_kwargs[
                "dataloader_num_workers"
            ] = self.cfg.dataloader_num_workers
        if self.cfg.dataloader_prefetch_factor is not None:
            training_arguments_kwargs[
                "dataloader_prefetch_factor"
            ] = self.cfg.dataloader_prefetch_factor
        if self.cfg.dataloader_drop_last is not None:
            training_arguments_kwargs[
                "dataloader_drop_last"
            ] = self.cfg.dataloader_drop_last
        elif self.cfg.sample_packing and self.cfg.eval_sample_packing is False:
            training_arguments_kwargs["dataloader_drop_last"] = True

        if self.cfg.remove_unused_columns is not None:
            training_arguments_kwargs[
                "remove_unused_columns"
            ] = self.cfg.remove_unused_columns

        if not self.cfg.test_datasets and self.cfg.val_set_size == 0:
            # no eval set, so don't eval
            training_arguments_kwargs["evaluation_strategy"] = "no"
        elif self.cfg.eval_steps:
            training_arguments_kwargs["evaluation_strategy"] = "steps"
            training_arguments_kwargs["eval_steps"] = self.cfg.eval_steps
        elif self.cfg.evaluation_strategy:
            training_arguments_kwargs[
                "evaluation_strategy"
            ] = self.cfg.evaluation_strategy
        else:
            # we have an eval set, but no steps defined, default to use epoch
            training_arguments_kwargs["evaluation_strategy"] = "epoch"

        if self.cfg.save_steps:
            training_arguments_kwargs["save_strategy"] = "steps"
            training_arguments_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_arguments_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_arguments_kwargs["save_strategy"] = "epoch"

        training_arguments_kwargs["save_only_model"] = self.cfg.save_only_model

        if self.cfg.do_bench_eval:
            training_arguments_kwargs["do_bench_eval"] = self.cfg.do_bench_eval
            if self.cfg.bench_dataset:
                training_arguments_kwargs["bench_dataset"] = self.cfg.bench_dataset
        if self.cfg.do_causal_lm_eval:
            training_arguments_kwargs["do_causal_lm_eval"] = self.cfg.do_causal_lm_eval
        if self.cfg.metric_for_best_model:
            training_arguments_kwargs[
                "metric_for_best_model"
            ] = self.cfg.metric_for_best_model
        if self.cfg.greater_is_better:
            training_arguments_kwargs["greater_is_better"] = self.cfg.greater_is_better

        if self.cfg.torch_compile:
            if torch.__version__ < "2.1.0":  # pylint: disable=protected-access
                LOG.warning("torch>=2.1.0 required for torch_compile to work properly")
            elif torch._dynamo:  # pylint: disable=protected-access
                torch._dynamo.config.suppress_errors = (  # pylint: disable=protected-access
                    True
                )
                training_arguments_kwargs["torch_compile"] = self.cfg.torch_compile
                if self.cfg.torch_compile_backend:
                    training_arguments_kwargs[
                        "torch_compile_backend"
                    ] = self.cfg.torch_compile_backend
                if self.cfg.torch_compile_mode:
                    training_arguments_kwargs[
                        "torch_compile_mode"
                    ] = self.cfg.torch_compile_mode

        # DDP Config
        if self.cfg.ddp_timeout:
            training_arguments_kwargs["ddp_timeout"] = self.cfg.ddp_timeout
        # see https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        if self.cfg.ddp_bucket_cap_mb:
            training_arguments_kwargs["ddp_bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb
        if self.cfg.ddp_broadcast_buffers is not None:
            training_arguments_kwargs[
                "ddp_broadcast_buffers"
            ] = self.cfg.ddp_broadcast_buffers

        # these are all the "standard" kwargs that are def used
        training_arguments_kwargs["max_steps"] = (
            total_num_steps if self.cfg.max_steps else -1
        )
        training_arguments_kwargs["max_seq_length"] = self.cfg.sequence_len
        training_arguments_kwargs[
            "per_device_train_batch_size"
        ] = self.cfg.micro_batch_size
        if self.cfg.eval_batch_size:
            training_arguments_kwargs[
                "per_device_eval_batch_size"
            ] = self.cfg.eval_batch_size
        if self.cfg.auto_find_batch_size is not None:
            training_arguments_kwargs[
                "auto_find_batch_size"
            ] = self.cfg.auto_find_batch_size
        training_arguments_kwargs[
            "gradient_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs[
            "eval_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs["num_train_epochs"] = self.cfg.num_epochs
        training_arguments_kwargs["learning_rate"] = self.cfg.learning_rate
        training_arguments_kwargs["output_dir"] = self.cfg.output_dir
        training_arguments_kwargs["save_total_limit"] = (
            self.cfg.save_total_limit if self.cfg.save_total_limit else 4
        )
        training_arguments_kwargs["load_best_model_at_end"] = (
            (
                self.cfg.load_best_model_at_end is not False
                or self.cfg.early_stopping_patience
            )
            and (
                (not self.cfg.test_datasets and self.cfg.val_set_size > 0)
                or (self.cfg.test_datasets and self.cfg.val_set_size == 0)
            )
            and self.cfg.save_steps
            and self.cfg.eval_steps
            and self.cfg.save_steps % self.cfg.eval_steps == 0
        ) or False
        training_arguments_kwargs["ddp_find_unused_parameters"] = (
            False if self.cfg.ddp else None
        )
        training_arguments_kwargs["group_by_length"] = self.cfg.group_by_length
        training_arguments_kwargs["curriculum_sampling"] = self.cfg.curriculum_sampling
        report_to = []
        if self.cfg.use_wandb:
            report_to.append("wandb")
        if self.cfg.use_mlflow:
            report_to.append("mlflow")
        if self.cfg.use_tensorboard:
            report_to.append("tensorboard")

        training_arguments_kwargs["report_to"] = report_to
        training_arguments_kwargs["run_name"] = (
            self.cfg.wandb_name if self.cfg.use_wandb else None
        )
        training_arguments_kwargs["optim"] = (
            self.cfg.optimizer if self.cfg.optimizer else "adamw_hf"
        )
        if self.cfg.optim_args:
            if isinstance(self.cfg.optim_args, dict):
                optim_args = ",".join(
                    [f"{key}={value}" for key, value in self.cfg.optim_args.items()]
                )
            else:
                optim_args = self.cfg.optim_args
            training_arguments_kwargs["optim_args"] = optim_args
        if self.cfg.optim_target_modules:
            training_arguments_kwargs[
                "optim_target_modules"
            ] = self.cfg.optim_target_modules
        training_arguments_kwargs["loraplus_lr_ratio"] = self.cfg.loraplus_lr_ratio
        training_arguments_kwargs[
            "loraplus_lr_embedding"
        ] = self.cfg.loraplus_lr_embedding
        if self.cfg.lr_scheduler in ["one_cycle", "log_sweep"]:
            training_arguments_kwargs["lr_scheduler_type"] = "cosine"
            training_arguments_kwargs[
                "alternate_lr_scheduler_type"
            ] = self.cfg.lr_scheduler
        else:
            training_arguments_kwargs["lr_scheduler_type"] = (
                self.cfg.lr_scheduler if self.cfg.lr_scheduler else "cosine"
            )
        training_arguments_kwargs["lr_scheduler_kwargs"] = (
            self.cfg.lr_scheduler_kwargs if self.cfg.lr_scheduler_kwargs else {}
        )
        training_arguments_kwargs["cosine_min_lr_ratio"] = self.cfg.cosine_min_lr_ratio
        training_arguments_kwargs[
            "cosine_constant_lr_ratio"
        ] = self.cfg.cosine_constant_lr_ratio
        training_arguments_kwargs["weight_decay"] = (
            self.cfg.weight_decay if self.cfg.weight_decay is not None else 0.0
        )

        training_arguments_kwargs["sample_packing"] = bool(self.cfg.sample_packing)
        training_arguments_kwargs["multipack_real_batches"] = (
            not self.cfg.flash_attention or self.cfg.multipack_real_batches
        )
        training_arguments_kwargs["eval_sample_packing"] = bool(
            self.cfg.eval_sample_packing
        )
        if self.cfg.sample_packing_bin_size is not None:
            training_arguments_kwargs[
                "sample_packing_bin_size"
            ] = self.cfg.sample_packing_bin_size
        if self.cfg.sample_packing_group_size is not None:
            training_arguments_kwargs[
                "sample_packing_group_size"
            ] = self.cfg.sample_packing_group_size
        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs[
                "sample_packing_efficiency"
            ] = self.cfg.sample_packing_eff_est

        if self.cfg.relora_steps:
            training_arguments_kwargs["relora_steps"] = self.cfg.relora_steps
            training_arguments_kwargs[
                "relora_warmup_steps"
            ] = self.cfg.relora_warmup_steps
            if self.cfg.relora_anneal_steps:
                training_arguments_kwargs[
                    "relora_anneal_steps"
                ] = self.cfg.relora_anneal_steps
            if self.cfg.relora_prune_ratio:
                training_arguments_kwargs[
                    "relora_prune_ratio"
                ] = self.cfg.relora_prune_ratio

        if self.cfg.lisa_step_interval and self.cfg.lisa_n_layers:
            training_arguments_kwargs["lisa_n_layers"] = self.cfg.lisa_n_layers
            training_arguments_kwargs[
                "lisa_step_interval"
            ] = self.cfg.lisa_step_interval
            training_arguments_kwargs[
                "lisa_layers_attribute"
            ] = self.cfg.lisa_layers_attribute

        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_arguments_kwargs["model_type"] = self.cfg.model_config_type
        training_arguments_kwargs["pretraining"] = bool(self.cfg.pretraining_dataset)

        if self.cfg.rl == "orpo":
            training_arguments_kwargs["orpo_alpha"] = self.cfg.orpo_alpha

        if self.cfg.neftune_noise_alpha is not None:
            training_arguments_kwargs[
                "neftune_noise_alpha"
            ] = self.cfg.neftune_noise_alpha

        trainer_kwargs = {}

        if self.cfg.optimizer in [
            "optimi_adamw",
            "ao_adamw_4bit",
            "ao_adamw_8bit",
            "ao_adamw_fp8",
        ]:
            # Set default so transformers doesn't throw
            training_arguments_kwargs["optim"] = "adamw_hf"
            training_arguments_kwargs["alternate_optimizer"] = self.cfg.optimizer

        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

        if self.cfg.accelerator_config:
            training_arguments_kwargs[
                "accelerator_config"
            ] = self.cfg.accelerator_config

        training_args = (
            self._get_training_arguments_cls()(  # pylint: disable=unexpected-keyword-arg
                **training_arguments_kwargs,
            )
        )
        training_args = self.hook_post_create_training_args(training_args)

        data_collator_kwargs = {
            "padding": True,  # True/"longest" is the default
        }
        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = 64 * math.ceil(
                self.cfg.sequence_len / 64
            )
        else:
            # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
            data_collator_kwargs["pad_to_multiple_of"] = 64

        trainer_cls = self._get_trainer_cls()
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )

        print("self.model:", self.model)

        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self.build_collator(training_args, **data_collator_kwargs),
            eval_data_collator=self.build_collator(
                training_args, is_eval=True, **data_collator_kwargs
            ),
            bench_data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            ),
            callbacks=self.get_callbacks(),
            num_epochs=self.cfg.num_epochs,
            **trainer_kwargs,
        )
        trainer = self.hook_post_create_trainer(trainer)
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        if self.cfg.deepspeed and self.cfg.sample_packing:
            trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = self.cfg.micro_batch_size

        return trainer


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer, total_num_steps):
    trainer_builder = DistillationTrainerBuilder(cfg, model[0], tokenizer)

    trainer_builder.train_dataset = train_dataset
    trainer_builder.eval_dataset = eval_dataset

    return trainer_builder.build(total_num_steps)
