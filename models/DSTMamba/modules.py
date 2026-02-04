import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _get_statistics(self, x):

        dim2reduce = tuple(range(1, x.ndim - 1))

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):

        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):

        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x


class DataEmbedding(nn.Module):
    def __init__(self, history_seq_len, d_model, dropout):
        super().__init__()

        self.ValueEmb = nn.Linear(history_seq_len, d_model)
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x_in):
        # x_in: (batch_size, history_seq_len <-> num_channels)
        x_in = x_in.permute(0, 2, 1)

        x_emb = self.ValueEmb(x_in)  # x_emb: (batch_size, num_channels, d_model)
        x_emb = self.Dropout(x_emb)

        return x_emb


class Temporal_Decomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):

        moving_mean = self.moving_avg(x)
        res = x - moving_mean

        return res, moving_mean


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):

        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class MultiScaleTrendMixing(nn.Module):
    def __init__(
        self, history_seq_len, future_seq_len, num_channels, ds_layers, ds_window
    ):
        super().__init__()

        self.history_seq_len = history_seq_len
        self.future_seq_len = future_seq_len
        self.num_channels = num_channels
        self.ds_layers = ds_layers
        self.ds_window = ds_window

        # Length Alignment
        self.up_sampling = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    self.history_seq_len // (self.ds_window ** (l + 1)),
                    self.history_seq_len // (self.ds_window ** (l)),
                ),
                nn.GELU(),
                nn.Linear(
                    self.history_seq_len // (self.ds_window ** (l)),
                    self.history_seq_len // (self.ds_window ** (l)),
                ),
            )
            for _ in reversed(range(self.ds_layers))
        ])

    def forward(self, ms_trend_list):

        length_list = []
        trend_list = []
        for x in ms_trend_list:
            _, t, _ = x.size()
            length_list.append(t)
            trend_list.append(x.permute(0, 2, 1))  # [B, N, t]

        # Trend mixing
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]

        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()

        out_list = []
        for out_trend, length in zip(out_trend_list, length_list):
            out_list.append(
                out_trend[:, :length, :]
            )  # list of each element in [B, t, C]

        return out_list


class Encoder(nn.Module):
    def __init__(self, ssm_layers, norm_layer):
        super(Encoder, self).__init__()

        self.ssm_layers = nn.ModuleList(ssm_layers)
        self.norm_layer = norm_layer

    def forward(self, x_emb):
        # x_emb: [batch_size, num_channels, d_model]
        x_enc = x_emb

        for ssm_layer in self.ssm_layers:
            x_enc = ssm_layer(x_enc)

        # TODO: Test the effectiveness of _RMSNorm_
        if self.norm_layer is not None:
            enc_out = self.norm_layer(x_enc)

        return enc_out


class EncoderLayer(nn.Module):
    def __init__(self, ssm, ssm_r, d_model, d_ff, dropout, activation):
        super().__init__()

        self.ssm = ssm
        self.ssm_r = ssm_r

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x_enc):
        if self.ssm_r is not None:
            ssm_out = self.ssm(x_enc) + self.ssm_r(x_enc.flip(dims=[1])).flip(dims=[1])
        else:
            ssm_out = self.ssm(x_enc)

        out = x_enc = self.norm1(ssm_out)
        out = self.dropout(self.activation(self.conv1(out.transpose(-1, 1))))
        out = self.dropout(self.conv2(out).transpose(-1, 1))

        return self.norm2(out + x_enc)
