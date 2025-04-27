import sys

import torch
import torch.nn as nn

from .SeriesDec import series_decomp


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
    def __init__(self, **model_args):
        super(DLinear, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.individual = model_args['individual']

        self.kernel_size = model_args['kernel_size']

        self.build()


    def build(self):

        self.decomposition = series_decomp(self.kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for _ in range(self.num_channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.history_seq_len, self.future_seq_len))
                self.Linear_Trend.append(
                    nn.Linear(self.history_seq_len, self.future_seq_len))
                
        else:
            self.Linear_Seasonal = nn.Linear(self.history_seq_len, self.future_seq_len)
            self.Linear_Trend = nn.Linear(self.history_seq_len, self.future_seq_len)


    def forward(self, 
                history_data: torch.Tensor) -> torch.Tensor:
        
        assert history_data.shape[-1] == 1
        x = history_data[..., 0] 

        seasonal_init, trend_init = self.decompsition(x)
        # [batch_size, seq_len, num_channels -> batch_size, num_channels, seq_len]
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(
                1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(
                1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        prediction = seasonal_output + trend_output
        prediction = prediction.permute(0, 2, 1)
        prediction = prediction.unsqueeze(-1)

        return prediction