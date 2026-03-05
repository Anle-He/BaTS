from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from .modules import RecurrentCycle


@dataclass
class CycleNetArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    cycle_len: int
    cycle_pattern: str
    d_model: int
    add_norm: bool
    model_type: str


class CycleNet(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.args = CycleNetArgs(**model_args)
        self._build()

    def _build(self) -> None:

        self.cycleQueue = RecurrentCycle(
            cycle_len=self.args.cycle_len, channel_size=self.args.num_nodes
        )

        assert self.args.model_type in ['linear', 'mlp']
        if self.args.model_type == 'linear':
            self.model = nn.Linear(self.args.seq_len_in, self.args.seq_len_out)
        elif self.args.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.args.seq_len_in, self.args.d_model),
                nn.ReLU(),
                nn.Linear(self.args.d_model, self.args.seq_len_out),
            )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x = history_data[..., 0]

        if self.cycle_pattern == 'daily':
            cycle_index = history_data[..., 1] * self.args.cycle_len  # [B]
            cycle_index = cycle_index[
                :, -1, 0
            ]  # from CycleNet data_loader.py: "cycle_index = torch.tensor(self.cycle_index[s_end])"
        elif self.cycle_pattern == 'daily&weekly':
            cycle_index = (
                history_data[..., 1] * self.args.cycle_len * 7
                + history_data[..., 2] * 7
            )
            cycle_index = cycle_index[:, -1, 0]
        else:
            raise Exception('please specify cycle pattern, daily OR weekly OR others')

        if self.args.add_norm:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.history_seq_len)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue(
            (cycle_index + self.args.seq_len_in) % self.args.cycle_len,
            self.args.seq_len_out,
        )

        # instance denorm
        if self.args.add_norm:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y.unsqueeze(-1)
