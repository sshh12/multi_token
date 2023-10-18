from typing import List, Dict, Sequence
from dataclasses import dataclass, field
from collections import Counter

from torch.utils.data import Dataset
from datasets import load_from_disk
import transformers
import torch
import re

from multi_token.modalities.base_modality import Modality
from multi_token.constants import IGNORE_INDEX
from multi_token.data_tools import encode_chat


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

    def get_example(self) -> Dict:
        return self.dataset[0]

    def __getitem__(self, i) -> Dict:
        item = self.dataset[i]
        return encode_chat(item, self.tokenizer, self.modalities)


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
