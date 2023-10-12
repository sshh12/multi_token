from typing import List, Dict, Sequence
from dataclasses import dataclass, field
from collections import Counter

from torch.utils.data import Dataset
from datasets import load_from_disk
import transformers
import torch
import re

from lmm_multi_token.modalities.base_modality import Modality

IGNORE_INDEX = -100


@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


class LMMDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
    ):
        super(LMMDataset, self).__init__()
        self.dataset = load_from_disk(data_args.dataset_path)
        self.tokenizer = tokenizer
        self.modalities = modalities

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict:
        item = self.dataset[i]

        chat_as_string = self.tokenizer.apply_chat_template(
            item["messages"], tokenize=False
        )

        token_to_modality = {m.token: m for m in self.modalities}
        modality_token_counts = Counter()
        instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
        pattern = "(" + "|".join(re.escape(m.token) for m in self.modalities) + ")"

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
                    part_ids = self.tokenizer(subpart).input_ids
                    input_ids.extend(part_ids)
                    labels.extend([IGNORE_INDEX] * len(part_ids))
                else:
                    part_ids = self.tokenizer(subpart).input_ids
                    input_ids.extend(part_ids)
                    labels.extend(part_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
        )
        for m in self.modalities:
            data_dict[m.name] = m.preprocess_row(item)
            assert data_dict[m.name].shape[0] == modality_token_counts[m.name]

        return data_dict


@dataclass
class DataCollatorForSupervisedLMMDataset:
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ["input_ids", "labels"]
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch
