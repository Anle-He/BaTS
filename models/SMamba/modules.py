import torch.nn as nn
import torch.nn.functional as F


class SeriesEmbedding(nn.Module):
    def __init__(self, history_seq_len, d_model, dropout):
        super().__init__()

        self.FeatureEmb = nn.Linear(history_seq_len, d_model)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x_in):
        # x_in: [batch_size, history_seq_len <-> num_channels]
        x_in = x_in.permute(0, 2, 1)

        # [batch_size, num_channels, d_model]
        return self.Dropout(self.FeatureEmb(x_in))


class Encoder(nn.Module):
    def __init__(self, ssm_layers, norm=None):
        super().__init__()

        self.ssm_layers = nn.ModuleList(ssm_layers)
        self.norm = norm

    def forward(self, x_emb):
        # x_emb: [batch_size, num_nodes, d_model]
        x_enc = x_emb

        for ssm_layer in self.ssm_layers:
            x_enc = ssm_layer(x_enc)

        if self.norm is not None:
            return self.norm(x_enc)
        else:
            return x_enc


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
