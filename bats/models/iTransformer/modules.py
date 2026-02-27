from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesEmbedding(nn.Module):
    def __init__(self, seq_len_in, d_model, dropout):
        super().__init__()

        self.proj = nn.Linear(seq_len_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, L, N) -> (B, N, L) -> proj(x) -> (B, N, D)
        x = x.transpose(1, 2)
        return self.dropout(self.proj(x))


class FullAttention(nn.Module):
    def __init__(self, attention_dropout):
        super().__init__()

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / sqrt(E)

        scores = torch.einsum('blhe,bshe->bhls', queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, num_heads):
        super().__init__()

        d_keys = d_model // num_heads
        d_values = d_model // num_heads
        self.num_heads = num_heads
        self.inner_attention = attention

        self.query_projection = nn.Linear(d_model, d_keys * num_heads)
        self.key_projection = nn.Linear(d_model, d_keys * num_heads)
        self.value_projection = nn.Linear(d_model, d_values * num_heads)
        self.out_projection = nn.Linear(d_values * num_heads, d_model)

    def forward(self, queries, keys, values):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation):
        super().__init__()

        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):

        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer):
        super().__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):

        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        x = self.norm(x)

        return x
