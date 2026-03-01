from typing import Any
from dataclasses import dataclass

import torch
from torch import nn


from .modules import (
    Transpose,
    FlattenHead,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
    PatchEmbedding,
)


@dataclass
class PatchTSTArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    d_ff: int
    num_heads: int
    e_layers: int
    dropout: float
    activation: str
    patch_len: int
    stride: int
    add_norm: bool


class PatchTST(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.args = PatchTSTArgs(**model_args)
        self._build()

    def _build(self) -> None:

        self.patch_embedding = PatchEmbedding(
            d_model=self.args.d_model,
            patch_len=self.args.patch_len,
            stride=self.args.stride,
            padding=self.args.stride,
            dropout=self.args.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=self.args.dropout),
                        self.args.d_model,
                        self.args.num_heads,
                    ),
                    self.args.d_model,
                    self.args.d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.activation,
                )
                for _ in range(self.args.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(self.args.d_model), Transpose(1, 2)
            ),
        )

        patch_num = int(
            (self.args.seq_len_in - self.args.patch_len) / self.args.stride + 2
        )
        self.head_nf = self.args.d_model * patch_num
        self.head = FlattenHead(
            self.args.num_nodes,
            self.head_nf,
            self.args.seq_len_out,
            head_dropout=self.args.dropout,
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[..., 0]

        if self.args.add_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in /= stdev

        # do patching and embedding
        x_in = x_in.permute(0, 2, 1)  # [B, N, L]

        # enc_out: [B*N, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_in)

        # Encoder
        enc_out = self.encoder(enc_out)

        # Reshape for Head
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # [B, N, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction Head
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.args.add_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len_out, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len_out, 1)
            )

        y = dec_out.unsqueeze(-1)

        return y
