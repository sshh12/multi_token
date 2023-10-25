from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from functools import cached_property

import torch.nn as nn
import torch


class Modality(ABC):
    @abstractmethod
    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def token(self) -> str:
        pass

    @property
    @abstractmethod
    def data_key(self) -> str:
        pass

    @property
    @abstractmethod
    def token_width(self) -> int:
        pass

    @cached_property
    def token_idx(self) -> int:
        hash_ = sum(ord(c) ** i for i, c in enumerate(self.token))
        return -abs(hash_ % 10_000)

    @abstractmethod
    def preprocess_rows(self, rows: List[Dict]) -> List[Optional[Any]]:
        pass

    @abstractmethod
    def forward(self, encoded_values: List[Any]) -> List[torch.Tensor]:
        pass

    def to(self, dtype: torch.dtype, device: torch.device) -> "Modality":
        return self
