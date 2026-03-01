from typing import Any
from dataclasses import dataclass, fields

import torch
from torch import nn

from .modules import MultiLayerPerceptron


@dataclass
class STIDConfig:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    embed_dim: int
    hidden_dim: int
    add_tod: bool
    add_dow: bool
    add_spa: bool
    node_dim: int
    tod_feat_size: int
    dow_feat_size: int
    tod_emb_dim: int
    dow_emb_dim: int
    spa_emb_dim: int
    e_layers: int


class STID(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.config = STIDConfig(**model_args)
        for field in fields(self.config):
            setattr(self, field.name, getattr(self.config, field.name))

        self._build()

    def _build(self):

        if self.config.add_tod:
            self.tod_embedding = nn.Parameter(
                torch.empty(self.config.tod_feat_size, self.config.tod_emb_dim)
            )
            nn.init.xavier_uniform_(self.tod_embedding)

        if self.config.add_dow:
            self.dow_embedding = nn.Parameter(
                torch.empty(self.config.dow_feat_size, self.config.dow_emb_dim)
            )
            nn.init.xavier_uniform_(self.dow_embedding)

        if self.config.add_spa:
            self.spa_embedding = nn.Parameter(
                torch.empty(self.config.num_nodes, self.config.spa_emb_dim)
            )
            nn.init.xavier_uniform_(self.spa_embedding)

        self.embedding = nn.Conv2d(
            in_channels=self.config.seq_len_in,
            out_channels=self.config.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        hidden_dim = (
            self.config.embed_dim
            + self.config.tod_emb_dim
            + self.config.dow_emb_dim
            + self.config.spa_emb_dim
        )

        self.encoder = nn.Sequential(*[
            MultiLayerPerceptron(hidden_dim, hidden_dim)
            for _ in range(self.config.e_layers)
        ])

        self.regression_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=self.config.seq_len_out,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[..., 0]
        # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
        # If you use other datasets, you may need to change this line.

        tod_feat = history_data[..., 1]
        dow_feat = history_data[..., 2]

        # Debug: print min/max to check if normalized to [0, 1]
        print(
            f'[DEBUG] tod_feat: min={tod_feat.min().item():.4f}, max={tod_feat.max().item():.4f}'
        )
        print(
            f'[DEBUG] dow_feat: min={dow_feat.min().item():.4f}, max={dow_feat.max().item():.4f}'
        )

        tod_emb = self.tod_embedding[
            (tod_feat[:, -1, :] * self.config.tod_feat_size).type(torch.long)
        ]
        dow_emb = self.dow_embedding[(dow_feat[:, -1, :]).type(torch.long)]

        B, _, N = x_in.shape
        x_in = x_in.unsqueeze(-1)
        x_emb = self.embedding(x_in)

        spa_emb = []
        spa_emb.append(
            self.spa_embedding
            .unsqueeze(0)
            .expand(B, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )
        tem_emb = []
        tem_emb.append(tod_emb.transpose(1, 2).unsqueeze(-1))
        tem_emb.append(dow_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([x_emb] + spa_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        y = self.regression_layer(hidden)

        return y
