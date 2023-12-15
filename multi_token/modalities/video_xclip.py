from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel

from multi_token.data_tools import load_video
from multi_token.modalities.base_modality import Modality
from multi_token.modalities.projectors import (
    build_mlp_vector_projector,
)


OUTPUT_EMB_SIZE = 512


class XCLIPVideoModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None

        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, video_inputs) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**(video_inputs.to(device=self.device)))

        emb = outputs.video_embeds.to(device=self.device, dtype=self.dtype).view(
            -1, 1, OUTPUT_EMB_SIZE
        )
        return emb

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device


class XCLIPVideoModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "microsoft/xclip-base-patch32",
        num_projector_layers: int = 2,
        num_tokens_output: int = 10,
    ):
        self.model_name_or_path = model_name_or_path
        self.module = XCLIPVideoModule(model_name_or_path=self.model_name_or_path)
        self.num_projector_layers = num_projector_layers
        self.num_tokens_output = num_tokens_output

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_mlp_vector_projector(
            input_hidden_size=OUTPUT_EMB_SIZE,
            lm_hidden_size=lm_hidden_size,
            num_layers=self.num_projector_layers,
            num_tokens=self.num_tokens_output,
        )

    @property
    def name(self) -> str:
        return "video_xclip"

    @property
    def token(self) -> str:
        return "<video>"

    @property
    def data_key(self) -> str:
        return "videos"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "XCLIPVideoModality":
        self.module.to(dtype=dtype, device=device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Optional[Dict]]:
        row_values = []
        for row in rows:
            video_arrays = [
                load_video(
                    video_info,
                )
                for video_info in row[self.data_key]
            ]
            videos_enc = self.module.processor(
                videos=[list(video) for video in video_arrays],
                text=["IGNORE"],
                return_tensors="pt",
                padding=True,
            )
            row_values.append(videos_enc)
        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        video_features = []
        for video_batch in encoded_values:
            video_features.append(self.module.forward(video_batch))
        return video_features
