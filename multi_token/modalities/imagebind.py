from typing import Dict, List
import os

import torch
import torch.nn as nn

from multi_token.modalities.base_modality import Modality
from multi_token.modalities.projectors import build_mlp_vector_projector
from multi_token.data_tools import with_local_files

IMAGE_BIND_FORCE_CPU = "IMAGE_BIND_FORCE_CPU"
IMAGE_BIND_EMBEDDING_SIZE = 1024


class ImageBindModule(nn.Module):
    def __init__(self):
        super().__init__()
        from imagebind.models import imagebind_model
        from imagebind import data

        data.BPE_PATH = os.path.join(
            os.path.dirname(data.__file__), "..", "bpe", "bpe_simple_vocab_16e6.txt.gz"
        )
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, items: Dict) -> torch.Tensor:
        forward_outs = self.model(items)
        return forward_outs

    @property
    def embedding_size(self):
        return IMAGE_BIND_EMBEDDING_SIZE


class ImageBindModality(Modality):
    def __init__(
        self,
        num_projector_layers: int = 2,
        num_tokens: int = 4,
        preprocess_device: str = "cpu",
    ):
        self.module = ImageBindModule()
        self.dtype = torch.float32
        self.device = "cpu"  # used for outputs
        self.imagebind_device = "cpu"  # used for imagebind model itself
        self.preprocess_device = preprocess_device  # used for preprocessing
        self.num_projector_layers = num_projector_layers
        self.num_tokens = num_tokens

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_mlp_vector_projector(
            self.module.embedding_size,
            lm_hidden_size,
            num_layers=self.num_projector_layers,
            num_tokens=self.num_tokens,
        )

    @property
    def name(self) -> str:
        return "imagebind"

    @property
    def token(self) -> str:
        return "<imagebind>"

    @property
    def data_key(self) -> str:
        return "imagebinds"

    @property
    def token_width(self) -> int:
        return self.num_tokens

    def to(self, dtype: torch.dtype, device: torch.device) -> "ImageBindModality":
        # we ignore dtype and sometimes device as well
        self.device = device
        self.dtype = dtype
        if IMAGE_BIND_FORCE_CPU not in os.environ:
            # running out of VRAM on 24GB GPU
            self.module.to(device=device)
            self.imagebind_device = device
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[List[Dict]]:
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data

        row_values = []
        for row in rows:
            items = []
            with with_local_files(row[self.data_key]) as item_paths:
                for item_path in item_paths:
                    ib_modality = filename_to_imagebind_modality(item_path)
                    if ib_modality == ModalityType.TEXT:
                        items.append(
                            {
                                ModalityType.TEXT: data.load_and_transform_text(
                                    [item_path], self.preprocess_device
                                )
                            }
                        )
                    elif ib_modality == ModalityType.VISION:
                        items.append(
                            {
                                ModalityType.VISION: data.load_and_transform_vision_data(
                                    [item_path], self.preprocess_device
                                )
                            }
                        )
                    elif ib_modality == ModalityType.AUDIO:
                        items.append(
                            {
                                ModalityType.AUDIO: data.load_and_transform_audio_data(
                                    [item_path], self.preprocess_device
                                )
                            }
                        )
                    else:
                        raise ValueError(f"Unknown modality type: {ib_modality}")
            row_values.append(items)
        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[List[Dict]]) -> List[torch.Tensor]:
        item_features = []
        for item_batch in encoded_values:
            item_batch_emb = []
            for item in item_batch:
                item = {
                    k: v.to(device=self.imagebind_device, dtype=torch.float32)
                    for k, v in item.items()
                }
                item_batch_emb.extend(list(self.module.forward(item).values()))
            item_features.append(
                torch.stack(item_batch_emb).to(device=self.device, dtype=self.dtype)
            )
        # batch_size x num_items x 1 x embedding_size
        return item_features


def filename_to_imagebind_modality(fn: str) -> str:
    from imagebind.models.imagebind_model import ModalityType

    _, ext = os.path.splitext(fn)
    if ext in {".wav"}:
        return ModalityType.AUDIO
    elif ext in {".jpg", ".png", ".jpeg"}:
        return ModalityType.VISION
    else:
        return ModalityType.TEXT
