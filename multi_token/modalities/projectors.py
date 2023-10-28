import torch.nn as nn
import torch


def build_patch_mlp_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int
) -> nn.Module:
    modules = [nn.Linear(input_hidden_size, lm_hidden_size)]
    for _ in range(1, num_layers):
        modules.append(nn.GELU())
        modules.append(nn.Linear(lm_hidden_size, lm_hidden_size))
    return nn.Sequential(*modules)


class _MLPVectorProjector(nn.Module):
    def __init__(
        self, input_hidden_size: int, lm_hidden_size: int, num_layers: int, width: int
    ):
        super(_MLPVectorProjector, self).__init__()
        self.mlps = nn.ModuleList()
        for _ in range(width):
            mlp = [nn.Linear(input_hidden_size, lm_hidden_size)]
            for _ in range(1, num_layers):
                mlp.append(nn.GELU())
                mlp.append(nn.Linear(lm_hidden_size, lm_hidden_size))
            self.mlps.append(nn.Sequential(*mlp))

    def forward(self, x):
        return torch.cat([mlp(x) for mlp in self.mlps], dim=-2)


def build_mlp_vector_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int, num_tokens: int
):
    return _MLPVectorProjector(
        input_hidden_size, lm_hidden_size, num_layers, num_tokens
    )
