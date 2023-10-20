from typing import Type, List, Optional
import logging

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from peft import PeftModel
import torch
import os

from multi_token.model_utils import fix_tokenizer
from multi_token.modalities.base_modality import Modality
from multi_token.language_models.mistral import MistralForCausalLM
from multi_token.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from multi_token.modalities import MODALITY_BUILDERS


def load_trained_lora_model(
    model_name_or_path: str,
    model_lora_path: str,
    model_cls: Optional[Type] = None,
    modalities: Optional[List[Modality]] = None,
    load_bits: int = 16,
    device_map: str = "auto",
):
    load_kwargs = {"device_map": device_map}

    if load_bits == 8:
        load_kwargs["load_in_8bit"] = True
    elif load_bits == 4:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_bits == 16:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(f"Invalid load_bits: {load_bits}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    fix_tokenizer(tokenizer)

    cfg = AutoConfig.from_pretrained(model_lora_path)
    if model_cls is None:
        model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[cfg.model_cls]
    if modalities is None:
        modalities = MODALITY_BUILDERS[cfg.modality_builder]()

    logging.info(f"Loading base model from {model_name_or_path} as {load_bits} bits")
    model = model_cls.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, config=cfg, **load_kwargs
    )
    model.modalities = modalities

    logging.info(f"Loading projector weights for {[m.name for m in modalities]}")
    if os.path.exists(os.path.join(model_lora_path, "non_lora_trainables.bin")):
        non_lora_trainables = torch.load(
            os.path.join(model_lora_path, "non_lora_trainables.bin"), map_location="cpu"
        )
    else:
        local_fn = hf_hub_download(
            repo_id=model_lora_path,
            filename="non_lora_trainables.bin",
            repo_type="model",
        )
        non_lora_trainables = torch.load(local_fn, map_location="cpu")
    model.get_model().initialize_pretrained_modules(modalities, non_lora_trainables)

    logging.info(f"Loading and merging LoRA weights from {model_lora_path}")
    model = PeftModel.from_pretrained(model, model_lora_path)
    if load_bits == 16:
        # TODO: Figure out why this fails for other bit sizes
        model = model.merge_and_unload()
    model.eval()

    return model, tokenizer
