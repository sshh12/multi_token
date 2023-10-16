from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image

from lmm_multi_token.modalities.base_modality import Modality
from lmm_multi_token.data_tools import load_image


class CLIPVisionModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.feature_layer = -2
        self.model_name_or_path = model_name_or_path
        self.image_processor = None
        self.image_model = None

        self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model_name_or_path
        )
        self.image_model = CLIPVisionModel.from_pretrained(self.model_name_or_path)
        self.image_model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images) -> torch.Tensor:
        image_forward_outs = self.image_model(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        image_features = image_forward_outs.hidden_states[self.feature_layer]
        image_features = image_features[:, 1:].to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_model.dtype

    @property
    def device(self):
        return self.image_model.device

    @property
    def config(self):
        return self.image_model.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def _expand2square(pil_img: Image, background_color: Tuple) -> Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class CLIPVisionModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        pad_non_square_images: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.module = CLIPVisionModule(model_name_or_path=self.model_name_or_path)
        self.pad_non_square_images = pad_non_square_images

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        modules = [nn.Linear(self.module.hidden_size, lm_hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(lm_hidden_size, lm_hidden_size))
        return nn.Sequential(*modules)

    @property
    def name(self) -> str:
        return "vision_clip"

    @property
    def token(self) -> str:
        return "<image>"

    @property
    def data_key(self) -> str:
        return "images"

    @property
    def token_width(self) -> int:
        return self.module.num_patches

    def to(self, dtype: torch.dtype, device: torch.device) -> "CLIPVisionModality":
        self.module.to(dtype=dtype, device=device)
        return self

    def preprocess_row(self, row: Dict) -> Optional[torch.Tensor]:
        images = []
        for image_fn in row[self.data_key]:
            image_obj = load_image(image_fn)
            if self.pad_non_square_images:
                image_obj = _expand2square(
                    image_obj,
                    tuple(int(x * 255) for x in self.module.image_processor.image_mean),
                )
            image = self.module.image_processor.preprocess(
                image_obj, return_tensors="pt"
            )["pixel_values"][0]
            images.append(image)
        return torch.stack(images) if len(images) > 0 else None

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        image_features = []
        for image_batch in encoded_values:
            image_features.append(self.module.forward(image_batch))
        return image_features
