from typing import Dict, List
from collections import Counter
import re
import io
import os

import torch
import requests
import transformers
from PIL import Image

from lmm_multi_token.modalities.base_modality import Modality
from lmm_multi_token.constants import IGNORE_INDEX


def encode_chat(
    item: Dict, tokenizer: transformers.PreTrainedTokenizer, modalities: List[Modality]
) -> Dict:
    chat_as_string = tokenizer.apply_chat_template(item["messages"], tokenize=False)

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
                m = token_to_modality[subpart]
                modality_token_counts[m.name] += 1
                input_ids.extend([m.token_idx] * m.token_width)
                labels.extend([IGNORE_INDEX] * m.token_width)
            elif is_instruction:
                part_ids = tokenizer(subpart).input_ids
                input_ids.extend(part_ids)
                labels.extend([IGNORE_INDEX] * len(part_ids))
            else:
                part_ids = tokenizer(subpart).input_ids
                input_ids.extend(part_ids)
                labels.extend(part_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    data_dict = dict(
        input_ids=input_ids,
        labels=labels,
    )
    for m in modalities:
        data_dict[m.name] = m.preprocess_row(item)
        assert data_dict[m.name].shape[0] == modality_token_counts[m.name]
    return data_dict


def load_image(path: str) -> Image:
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(path)
        img = Image.open(io.BytesIO(response.content))
    elif os.path.exists(path):
        img = Image.open(path)
    else:
        raise ValueError(path)
    img = img.convert("RGB")
    return img
