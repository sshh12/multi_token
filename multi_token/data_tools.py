from typing import Dict, List, Any
from collections import Counter
import contextlib
import tempfile
import shutil
import re
import io
import os

import torch
import requests
import transformers
from PIL import Image

from multi_token.constants import IGNORE_INDEX


def encode_chat(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List["Modality"],
) -> Dict:
    from multi_token.modalities.base_modality import Modality

    messages = list(item["messages"])
    chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

    token_to_modality = {m.token: m for m in modalities}
    modality_token_counts = Counter()
    instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
    pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

    chat_part = re.split(instruct_pattern, chat_as_string)
    input_ids = []
    labels = []
    for part in chat_part:
        if "[INST]" in part:
            is_instruction = True
        else:
            is_instruction = False
        for subpart in re.split(pattern, part):
            if not subpart:
                continue
            if subpart in token_to_modality:
                assert (
                    is_instruction
                ), "There should be no modality tokens outside of instructions"
                m = token_to_modality[subpart]
                modality_token_counts[m.name] += 1
                input_ids.extend([m.token_idx] * m.token_width)
                labels.extend([IGNORE_INDEX] * m.token_width)
            elif is_instruction:
                part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend([IGNORE_INDEX] * len(part_ids))
            else:
                part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend(part_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    data_dict = dict(
        input_ids=input_ids,
        labels=labels,
    )
    for m in modalities:
        data_dict[m.name] = m.preprocess_rows([item])[0]
        assert (
            data_dict[m.name].shape[0] == modality_token_counts[m.name]
        ), "The number of preprocessed items should match the number of tokens in the instruction"
    return data_dict


def load_image(value: Any) -> Image.Image:
    img = None
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            response = requests.get(value)
            img = Image.open(io.BytesIO(response.content))
        elif os.path.exists(value):
            img = Image.open(value)
    elif isinstance(value, Image.Image):
        img = value
    if img is None:
        raise ValueError(f"Could not load image from {value}")
    img = img.convert("RGB")
    return img


@contextlib.contextmanager
def with_local_files(fn_or_urls: List[Any]):
    local_fns = []
    fps = []
    for fn_or_url in fn_or_urls:
        if isinstance(fn_or_url, Image.Image):
            fp = tempfile.NamedTemporaryFile(suffix=suffix, mode="wb")
            fn_or_url.convert("RGB").save(fp)
            fps.append(fp)
            local_fns.append(fp.name)
        elif fn_or_url.startswith("http://") or fn_or_url.startswith("https://"):
            suffix = os.path.splitext(fn_or_url)[-1]
            with requests.get(fn_or_url, stream=True) as r:
                fp = tempfile.NamedTemporaryFile(suffix=suffix, mode="wb")
                shutil.copyfileobj(r.raw, fp)
                fps.append(fp)
                local_fns.append(fp.name)
        else:
            local_fns.append(fn_or_url)
    try:
        yield local_fns
    finally:
        for fp in fps:
            fp.close()
