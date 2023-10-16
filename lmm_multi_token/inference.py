from typing import Type, List
import logging

from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel
import torch
import os

from lmm_multi_token.model_utils import fix_tokenizer
from lmm_multi_token.modalities.base_modality import Modality
from lmm_multi_token.language_models.mistral import MistralForCausalLM


def load_trained_lora_model(
    model_name_or_path: str,
    model_lora_path: str,
    model_cls: Type,
    modalities: List[Modality],
    device_map: str = "auto",
):
    load_kwargs = {"device_map": device_map}
    load_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    fix_tokenizer(tokenizer)

    lora_cfg = AutoConfig.from_pretrained(model_name_or_path)

    logging.info(f"Loading base model from {model_name_or_path}")
    model = model_cls.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, config=lora_cfg, **load_kwargs
    )
    model.modalities = modalities

    logging.info(f"Loading projector weights for {[m.name for m in modalities]}")
    non_lora_trainables = torch.load(
        os.path.join(model_lora_path, "non_lora_trainables.bin"), map_location="cpu"
    )
    model.get_model().initialize_pretrained_modules(modalities, non_lora_trainables)

    logging.info(f"Loading & merging LoRA weights from {model_lora_path}")
    model = PeftModel.from_pretrained(model, model_lora_path)
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer
