from typing import Any

import torch
import torch.nn as nn

from .series_decomp import SeriesDecomp


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

        # Verify the necessary args
        required_keys = [
            'history_seq_len',
            'future_seq_len',
            'num_channels',
            'individual',
            'kernel_size',
        ]
        for key in required_keys:
            if key not in model_args:
                raise ValueError(f'Missing required parameter: {key}')

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.individual = model_args['individual']
        self.kernel_size = model_args['kernel_size']

        self.build()

    def build(self) -> None:
        self.decomposition = SeriesDecomp(self.kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for _ in range(self.num_channels):
                self.linear_seasonal.append(
                    nn.Linear(self.history_seq_len, self.future_seq_len)
                )
                self.linear_trend.append(
                    nn.Linear(self.history_seq_len, self.future_seq_len)
                )
        else:
            self.linear_seasonal = nn.Linear(self.history_seq_len, self.future_seq_len)
            self.linear_trend = nn.Linear(self.history_seq_len, self.future_seq_len)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        assert history_data.shape[-1] == 1, (
            f'Expected last dimension to be 1, got {history_data.shape[-1]}'
        )
        x = history_data[
            ..., 0
        ]  # from [batch_size, seq_len, num_channels, 1] -> [batch_size, seq_len, num_channels]

        seasonal_init, trend_init = self.decomposition(x)
        # [batch_size, seq_len, num_channels -> batch_size, num_channels, seq_len]
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )

        if self.individual:
            seasonal_output = self._create_output_tensor(seasonal_init)
            trend_output = self._create_output_tensor(trend_init)

            for i in range(self.num_channels):
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
            [input_tensor.size(0), input_tensor.size(1), self.future_seq_len],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    def __repr__(self) -> str:
        return (
            f'DLinear(history_seq_len={self.history_seq_len}, '
            f'future_seq_len={self.future_seq_len}, '
            f'num_channels={self.num_channels}, '
            f'individual={self.individual}, '
            f'kernel_size={self.kernel_size})'
        )
