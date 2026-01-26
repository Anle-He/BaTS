import torch.nn as nn


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
