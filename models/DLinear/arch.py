from typing import Any
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from .modules import SeriesDecomp


@dataclass
class DLinearConfig:
    history_seq_len: int
    future_seq_len: int
    num_nodes: int
    individual: bool
    kernel_size: int


class DLinear(nn.Module):
    """
    Paper:
        - Are Transformers Effective for Time Series Forecasting?
        - AAAI 2023
    Official Code:
        - https://github.com/cure-lab/LTSF-Linear
    Other Implementations can be found at:
        - BasicTS
    """

    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.config = DLinearConfig(**model_args)
        for field in fields(self.config):
            setattr(self, field.name, getattr(self.config, field.name))

        self.build()

    def build(self) -> None:
        self.decomposition = SeriesDecomp(self.config.kernel_size)

        if self.config.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for _ in range(self.config.num_nodes):
                self.linear_seasonal.append(
                    nn.Linear(self.config.history_seq_len, self.config.future_seq_len)
                )
                self.linear_trend.append(
                    nn.Linear(self.config.history_seq_len, self.config.future_seq_len)
                )
        else:
            self.linear_seasonal = nn.Linear(self.config.history_seq_len, self.config.future_seq_len)
            self.linear_trend = nn.Linear(self.config.history_seq_len, self.config.future_seq_len)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x = history_data[
            ..., 0
        ]  # from [batch_size, seq_len, num_nodes, 1] -> [batch_size, seq_len, num_nodes]

        seasonal_init, trend_init = self.decomposition(x)
        # [batch_size, seq_len, num_nodes -> batch_size, num_nodes, seq_len]
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )

        if self.config.individual:
            seasonal_output = self._create_output_tensor(seasonal_init)
            trend_output = self._create_output_tensor(trend_init)

            for i in range(self.config.num_nodes):
                seasonal_output[:, i, :] = self.linear_seasonal[i](  # type: ignore
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])  # type: ignore
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        prediction = seasonal_output + trend_output
        prediction = prediction.permute(0, 2, 1)
        prediction = prediction.unsqueeze(-1)

        return prediction

    def _create_output_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            [input_tensor.size(0), input_tensor.size(1), self.config.future_seq_len],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    def __repr__(self) -> str:
        return (
            f'DLinear(history_seq_len={self.config.history_seq_len}, '
            f'future_seq_len={self.config.future_seq_len}, '
            f'num_nodes={self.config.num_nodes}, '
            f'individual={self.config.individual}, '
            f'kernel_size={self.config.kernel_size})'
        )
