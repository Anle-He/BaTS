from typing import Any
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from mamba_ssm import Mamba  # type: ignore

from .modules import SeriesEmbedding, Encoder, EncoderLayer


@dataclass
class SMambaArgs:
    seq_len_in: int
    seq_len_out: int
    use_norm: bool
    d_model: int
    d_state: int
    d_conv: int
    expand: int
    d_ff: int
    dropout: float
    ffn_activation: str
    e_layers: int


class SMamba(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.args = SMambaArgs(**model_args)
        self._build()

    def _build(self) -> None:

        self.embedding = SeriesEmbedding(
            self.args.seq_len_in,
            self.args.d_model,
            self.args.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=self.args.d_model,
                        d_state=self.args.d_state,
                        d_conv=self.args.d_conv,
                        expand=self.args.expand,
                    ),
                    Mamba(
                        d_model=self.args.d_model,
                        d_state=self.args.d_state,
                        d_conv=self.args.d_conv,
                        expand=self.args.expand,
                    ),
                    self.args.d_model,
                    self.args.d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.ffn_activation,
                )
                for layer in range(self.args.e_layers)
            ],
            norm=nn.LayerNorm(self.args.d_model),
        )

        self.projector = nn.Linear(
            self.args.d_model, self.args.seq_len_out, bias=True
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[
            ..., 0
        ]  # from [batch_size, seq_len, num_channels, 1] -> [batch_size, seq_len, num_channels]

        if self.use_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in /= stdev

        # Embedding: [B, T, N] -> [B, N, E]
        emb_out = self.embedding(x_in)

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(emb_out)

        # Projector: [B, N, E] -> [B, N, T] -> [B, T, N]
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
