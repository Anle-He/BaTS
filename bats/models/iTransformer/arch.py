from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from .modules import (
    SeriesEmbedding,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)


@dataclass
class iTransformerArgs:
    seq_len_in: int
    seq_len_out: int
    d_model: int
    d_ff: int
    dropout: float
    num_heads: int
    activation: str
    e_layers: int
    use_norm: bool


class iTransformer(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.args = iTransformerArgs(**model_args)
        self._build()

    def _build(self) -> None:

        self.embedding = SeriesEmbedding(
            self.args.seq_len_in, self.args.d_model, self.args.dropout
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
            norm_layer=nn.LayerNorm(self.args.d_model),
        )

        self.projector = nn.Linear(self.args.d_model, self.args.seq_len_out, bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[..., 0]

        if self.use_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in /= stdev

        x_enc = self.embedding(x_in)

        enc_out = self.encoder(x_enc)

        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len_out, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len_out, 1)
            )

        y = dec_out.unsqueeze(-1)

        return y
