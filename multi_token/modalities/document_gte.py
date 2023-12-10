from typing import Dict, List

import torch
import torch.nn as nn
import os
from functools import cache
from transformers import AutoTokenizer, AutoModel

from multi_token.modalities.base_modality import Modality
from multi_token.modalities.projectors import build_mlp_vector_projector

GTE_EMBEDDING_SIZE = 1024
GTE_CONTEXT_WINDOW = 512
GTE_DEFAULT_MODEL = "thenlper/gte-large"
DOCUMENT_GTE_FORCE_CPU = "DOCUMENT_GTE_FORCE_CPU"


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@cache
def _get_tokenizer(model_name_or_path: str = GTE_DEFAULT_MODEL):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def split_text_into_documents(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize

    tokenizer = _get_tokenizer(GTE_DEFAULT_MODEL)

    sentences = sent_tokenize(text)
    documents = [[]]

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(documents[-1]) + len(sentence_tokens) > GTE_CONTEXT_WINDOW:
            documents.append([])
        documents[-1].extend(sentence_tokens)

    return [tokenizer.decode(doc) for doc in documents]


class DocumentGTEModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.feature_layer = -2
        self.model_name_or_path = model_name_or_path

        self.model = AutoModel.from_pretrained("thenlper/gte-large")
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch_dict) -> torch.Tensor:
        outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings

    @property
    def embedding_size(self):
        return GTE_EMBEDDING_SIZE


class DocumentGTEModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = GTE_DEFAULT_MODEL,
        num_projector_layers: int = 2,
        num_tokens_output: int = 4,
    ):
        self.model_name_or_path = model_name_or_path
        self.module = DocumentGTEModule(model_name_or_path=self.model_name_or_path)
        self.tokenizer = _get_tokenizer(model_name_or_path)
        self.num_projector_layers = num_projector_layers
        self.num_tokens_output = num_tokens_output
        self.dtype = torch.float32
        self.device = "cpu"
        self.document_gte_device = "cpu"

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_mlp_vector_projector(
            input_hidden_size=self.module.embedding_size,
            lm_hidden_size=lm_hidden_size,
            num_layers=self.num_projector_layers,
            num_tokens=self.num_tokens_output,
        )

    @property
    def name(self) -> str:
        return "document_gte"

    @property
    def token(self) -> str:
        return "<document>"

    @property
    def data_key(self) -> str:
        return "documents"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "DocumentGTEModality":
        self.dtype = dtype
        self.device = device
        if DOCUMENT_GTE_FORCE_CPU not in os.environ:
            # running out of VRAM on 24GB GPU
            self.document_gte_device = device
        self.module.to(device=self.document_gte_device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Dict]:
        row_values = []
        for row in rows:
            documents = []
            for doc in row[self.data_key]:
                documents.append(doc)
            documents_tokenized = self.tokenizer(
                documents,
                max_length=GTE_CONTEXT_WINDOW,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            row_values.append(documents_tokenized)
        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[Dict]) -> List[torch.Tensor]:
        outputs = []
        for val in encoded_values:
            outputs.append(
                self.module.forward(val.to(device=self.document_gte_device))
                .to(device=self.device, dtype=self.dtype)
                .view(-1, 1, self.module.embedding_size)
            )
        # batch_size x num_items x 1 x embedding_size
        return outputs
