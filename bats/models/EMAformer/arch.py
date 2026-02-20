from typing import Any
from dataclasses import dataclass, fields

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
class EMAformerConfig:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    d_ff: int
    n_heads: int
    cycle: int
    cycle_len: int
    cycle_pattern: str
    dropout: float
    activation: str
    e_layers: int
    use_norm: bool


class EMAformer(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.config = EMAformerConfig(**model_args)
        for field in fields(self.config):
            setattr(self, field.name, getattr(self.config, field.name))

        self._build()

    def _build(self) -> None:

        self.embedding = SeriesEmbedding(self.config.seq_len_in, self.config.d_model, self.config.dropout)

        self.channel_embedding = nn.Parameter(torch.zeros(self.config.num_nodes, self.config.d_model))
        self.phase_embedding = nn.Embedding(self.config.cycle, self.config.d_model)
        self.joint_embedding = nn.Embedding(self.config.cycle_len, self.config.num_nodes * self.config.d_model)
        nn.init.xavier_normal_(self.channel_embedding)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.joint_embedding.weight)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.config.dropout),
                        self.config.d_model,
                        self.config.n_heads,
                    ),
                    self.config.d_model,
                    self.config.d_ff,
                    dropout=self.config.dropout,
                    activation=self.config.activation
                ) for layer in range(self.config.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.config.d_model),
        )

        self.projector = nn.Linear(self.config.d_model, self.config.seq_len_out, bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[..., 0]
        B, L, N = x_in.shape

        if self.config.cycle_pattern == 'daily':
            cycle_index = history_data[..., 1] * self.config.cycle_len
            cycle_index = cycle_index[:, -1, 0]  # from CycleNet data_loader.py: "cycle_index = torch.tensor(self.cycle_index[s_end])"
        elif self.config.cycle_pattern == 'daily&weekly':
            cycle_index = (
                history_data[..., 1] * self.config.cycle * 7 + history_data[..., 2] * 7
            )
            cycle_index = cycle_index[:, -1, 0]
        else:
            raise Exception('please specify cycle pattern, daily OR weekly OR others')

        if self.config.use_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in /= stdev

        x_enc = self.embedding(x_in)

        cycle_index = cycle_index.long()
        channel_emb = self.channel_embedding.expand(x_enc.shape[0], N, -1)  # Channel embedding
        phase_emb = self.phase_embedding(cycle_index.view(-1, 1).expand(B, N))  # Phase embedding
        joint_emb = self.joint_embedding(cycle_index).reshape(B, self.config.num_nodes, self.config.d_model)  # Joint Channel-Phase embedding
        x_enc = x_enc + channel_emb + phase_emb + joint_emb

        enc_out = self.encoder(x_enc)
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.config.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.config.seq_len_out, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.config.seq_len_out, 1))

        y = dec_out.unsqueeze(-1)

        return y
