from typing import Optional, List
from dataclasses import field, dataclass
import logging
import pathlib
import torch

import transformers
from transformers import Trainer

from lmm_multi_token.training_data import (
    DataArguments,
    LMMDataset,
    DataCollatorForSupervisedLMMDataset,
)
from lmm_multi_token.model_utils import make_model_lora
from lmm_multi_token.modalities.base_modality import Modality


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.1"
    )


class LMMTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # TODO: Save non-LORA
        super(LMMTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # TODO: Save non-LORA
        super(LMMTrainer, self)._save(output_dir, state_dict)


def train_for_modalities(
    model_cls,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    modalities: List[Modality],
):
    for m in modalities:
        m.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.unk_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.unk_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.unk_token

    dataset = LMMDataset(data_args, tokenizer, modalities)
    collator = DataCollatorForSupervisedLMMDataset(tokenizer, modalities)

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.modalities = modalities
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        logging.info("Adding LoRA adapters...")
        model = make_model_lora(model, training_args)
    else:
        raise ValueError("LoRA must be enabled, full training is not supported (yet)")

    model.get_model().initialize_modules(modalities)
    print(model)

    trainer = LMMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=None,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
