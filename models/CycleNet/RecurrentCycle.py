import torch
import torch.nn as nn


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()

        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size), requires_grad=True
        )

    def forward(self, index, length):
        gather_index = (
            index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len

        return self.data[gather_index.long()]
