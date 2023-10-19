import torch.nn as nn


def build_patch_mlp_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int
) -> nn.Module:
    modules = [nn.Linear(input_hidden_size, lm_hidden_size)]
    for _ in range(1, num_layers):
        modules.append(nn.GELU())
        modules.append(nn.Linear(lm_hidden_size, lm_hidden_size))
    return nn.Sequential(*modules)
