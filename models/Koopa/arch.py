import math

import torch
import torch.nn as nn


class Koopa(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x_in = history_data[
            ..., 0
        ]  # from [batch_size, seq_len, num_channels, 1] -> [batch_size, seq_len, num_channels]

        mean_enc = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_in = x_in / std_enc
