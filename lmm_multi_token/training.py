from typing import Optional, List
from dataclasses import field, dataclass
import logging
import pathlib
import torch
import os

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer

from lmm_multi_token.training_data import (
    DataArguments,
    LMMDataset,
    DataCollatorForSupervisedLMMDataset,
)
from lmm_multi_token.model_utils import (
    make_model_lora,
    get_peft_state,
    get_peft_state_non_lora,
    get_adapter_state,
)
from lmm_multi_token.modalities.base_modality import Modality


README_TEMPLATE = """
---
license: apache-2.0
base_model: {base_model}
dataset: {dataset}
tags:
  - finetuned
inference: false
---

These are weights for a version of `{base_model}` for multimodal applications. 

### Modalities

{modalities}

### Usage

GitHub: https://github.com/sshh12/lmm_multi_token

"""


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
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.1")
    model_cls: str = field(default="MistralLMMForCausalLM")
    modality_builder: str = field(default="vision_clip")
    model_lora_path: Optional[str] = field(default=None)


class LMMTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMTrainer, self)._save(output_dir, state_dict)

    def _save_extras(self, output_dir: Optional[str] = None):
        keys_to_match = ["mm_projector", "vision_resampler"]
        with open("test.txt", "w") as f:
            f.write(repr(list(self.model.named_parameters())))
        weight_to_save = get_adapter_state(self.model.named_parameters(), keys_to_match)

        self.model.config.save_pretrained(output_dir)
        torch.save(weight_to_save, os.path.join(output_dir, f"projectors.bin"))


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

    if model_args.model_lora_path:
        raise ValueError("LoRA path not supported for training.")

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

    if list(pathlib.Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # model.config.use_cache = True

    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora(model.named_parameters())
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        torch.save(
            non_lora_state_dict,
            os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
        )

    with open(os.path.join(training_args.output_dir, "README.md"), "w") as f:
        modalities_text = [
            f"* {m.__class__.__name__} (use `{m.token}` in text and provide `{m.data_key}`)"
            for m in modalities
        ]
        f.write(
            README_TEMPLATE.format(
                base_model=model_args.model_name_or_path,
                dataset=data_args.dataset,
                modalities="\n".join(modalities_text),
            )
        )
