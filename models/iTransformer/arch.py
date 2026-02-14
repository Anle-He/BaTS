from typing import Any
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from .modules import (
    DataEmbedding_inverted,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)

@dataclass
class iTransformerConfig:
    seq_len_in: int
    seq_len_out: int
    d_model: int
    d_ff: int
    dropout: float
    n_heads: int
    activation: str
    e_layers: int
    use_norm: bool


class iTransformer(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.config = iTransformerConfig(**model_args)
        for field in fields(self.config):
            setattr(self, field.name, getattr(self.config, field.name))

        self._build()

    def _build(self) -> None:

        self.embedding = DataEmbedding_inverted(
            self.config.seq_len_in, self.config.d_model, self.config.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout),
                        self.config.d_model,
                        self.config.n_heads,
                    ),
                    self.config.d_model,
                    self.config.d_ff,
                    dropout=self.config.dropout,
                    activation=self.config.activation,
                )
                for _ in range(self.config.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.config.d_model),
        )

        self.projector = nn.Linear(self.config.d_model, self.config.seq_len_out, bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[..., 0]

        if self.config.use_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in /= stdev

        x_enc = self.embedding(x_in)

        enc_out = self.encoder(x_enc)

        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.config.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.config.seq_len_out, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.config.seq_len_out, 1)
            )

        return dec_out.unsqueeze(-1)
