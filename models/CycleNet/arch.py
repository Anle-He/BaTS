import torch
import torch.nn as nn

from .RecurrentCycle import RecurrentCycle


class CycleNet(nn.Module):
    def __init__(self, **model_args):
        super(CycleNet, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']

        self.num_channels = model_args['num_channels']
        self.cycle_len = model_args['cycle_len']
        self.d_model = model_args['d_model']
        self.use_revin = model_args['use_revin']

        self.model_type = model_args['model_type']
        self.cycle_pattern = model_args['cycle_pattern']

        self.cycleQueue = RecurrentCycle(
            cycle_len=self.cycle_len, channel_size=self.num_channels
        )

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.history_seq_len, self.future_seq_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.history_seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.future_seq_len),
            )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x = history_data[..., 0]  # Only use the target feature.

        if self.cycle_pattern == 'daily':
            cycle_index = history_data[..., 1] * self.cycle_len  # [B]
            cycle_index = cycle_index[
                :, -1, 0
            ]  # from CycleNet data_loader.py: "cycle_index = torch.tensor(self.cycle_index[s_end])""
        elif self.cycle_pattern == 'daily&weekly':
            cycle_index = (
                history_data[..., 1] * self.cycle_len * 7 + history_data[..., 2] * 7
            )
            cycle_index = cycle_index[:, -1, 0]
        else:
            raise Exception('please specify cycle pattern, daily OR weekly OR others')

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.history_seq_len)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue(
            (cycle_index + self.history_seq_len) % self.cycle_len, self.future_seq_len
        )

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
        return y.unsqueeze(-1)
