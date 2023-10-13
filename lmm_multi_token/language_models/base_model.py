from typing import List
from abc import ABC, abstractmethod

from torch.nn.functional import conv1d
import torch
import torch.nn as nn

from lmm_multi_token.modalities.base_modality import Modality


class LMMMetaModel:
    def __init__(self, config):
        super(LMMMetaModel, self).__init__(config)
        # if hasattr(config, "mm_vision_tower"):
        #     self.vision_tower = build_vision_tower(config, delay_load=True)
        #     self.mm_projector = build_vision_projector(config)

    def initialize_modules(self, modalities: List[Modality]):
        names = [m.name for m in modalities]

        self.config.modalities = names

        for m in modalities:
            module = m.build_module()
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name, module)
            setattr(self, m.name + "_projector", projector)

        # TODO: support pretrained modalities


class LMMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, **kwargs
    ):
        model = self.get_model()

        try:
            inputs_embeds = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.config.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )

            projected_tensors = []
            for m in self.modalities:
                m_vals = m.forward(kwargs.get(m.name))
                mp_vals = []
                proj = getattr(model, m.name + "_projector")
                for m_val in m_vals:
                    mp_vals.append(proj(m_val))
                projected_tensors.append(mp_vals)

            for i, input_ids_sample in enumerate(input_ids):
                is_text_mask = input_ids_sample >= 0
                inputs_embeds[i, is_text_mask] = model.embed_tokens(
                    input_ids_sample[is_text_mask]
                )

                for mi, m in enumerate(self.modalities):
                    m_mask = (input_ids_sample == m.token_idx).float()
                    m_kernel = torch.tensor(
                        [-1] * m.token_width, dtype=m_mask.dtype, device=m_mask.device
                    )
                    m_conv = conv1d(
                        m_mask.unsqueeze(0).unsqueeze(0),
                        m_kernel.unsqueeze(0).unsqueeze(0),
                    )
                    indices = (m_conv[0, 0] == -m.token_width).nonzero(as_tuple=True)[
                        0
                    ][::2]
                    for k, index in enumerate(indices):
                        batch_modality_tensor = projected_tensors[mi][i][k]
                        inputs_embeds[
                            i, index : index + m.token_width
                        ] = batch_modality_tensor
        except Exception as e:
            print(e)

            import IPython

            IPython.embed()

        return None, attention_mask, past_key_values, inputs_embeds, labels
