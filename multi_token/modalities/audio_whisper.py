from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WhisperModel

from multi_token.data_tools import load_audio
from multi_token.modalities.base_modality import Modality
from multi_token.modalities.projectors import (
    build_mlp_vector_projector,
)


OUTPUT_EMB_SIZE = 768


class WhisperAudioModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.feature_extractor = None

        self.load_model()

    def load_model(self):
        self.model = WhisperModel.from_pretrained(self.model_name_or_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name_or_path
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, audios) -> torch.Tensor:
        hidden_states = []
        for i in range(audios.shape[0]):
            decoder_input_ids = (
                torch.tensor([[1]]) * self.model.config.decoder_start_token_id
            )
            last_hidden_state = self.model(
                audios[i].to(device=self.device, dtype=self.dtype),
                decoder_input_ids=decoder_input_ids.to(device=self.device),
            ).last_hidden_state
            hidden_states.append(last_hidden_state)
        last_hidden_state = torch.stack(hidden_states)
        return last_hidden_state.view(-1, 1, OUTPUT_EMB_SIZE)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device


class WhisperAudioModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-small",
        num_projector_layers: int = 2,
        num_tokens_output: int = 10,
    ):
        self.model_name_or_path = model_name_or_path
        self.module = WhisperAudioModule(model_name_or_path=self.model_name_or_path)
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
        return "audio_whisper"

    @property
    def token(self) -> str:
        return "<speech>"

    @property
    def data_key(self) -> str:
        return "speech_audios"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "WhisperAudioModality":
        self.module.to(dtype=dtype, device=device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Optional[torch.Tensor]]:
        row_values = []
        for row in rows:
            audios = []
            for audio_dict in row[self.data_key]:
                audio_dict = load_audio(
                    audio_dict,
                    target_sampling_rate=self.module.feature_extractor.sampling_rate,
                )
                audio_processed = self.module.feature_extractor(
                    audio_dict["array"],
                    return_tensors="pt",
                    sampling_rate=audio_dict["sampling_rate"],
                ).input_features
                audios.append(audio_processed)
            row_values.append(torch.stack(audios) if len(audios) > 0 else None)
        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        audio_features = []
        for audio_batch in encoded_values:
            audio_features.append(self.module.forward(audio_batch))
        return audio_features
