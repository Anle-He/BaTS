import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()

        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):

        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout):
        super().__init__()

        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        return x


class PositionalEmbedding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), n_vars


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
