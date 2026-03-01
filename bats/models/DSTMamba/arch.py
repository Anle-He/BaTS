from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm import Mamba  # type: ignore

from .modules import (
    RevIN,
    Temporal_Decomposition,
    MultiScaleTrendMixing,
    SeriesEmbedding,
    Encoder,
    EncoderLayer,
)


@dataclass
class DSTMambaArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    d_ff: int
    d_state: int
    d_conv: int
    expand: int
    dropout: float
    use_revin: bool
    activation: str
    e_layers: int
    std_kernel: int
    ds_type: str
    ds_layers: int
    ds_window: int


class DSTMamba(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.args = DSTMambaArgs(**model_args)
        self._build()

    def _build(self) -> None:

        self.revin = RevIN(self.args.num_nodes)

        self.decom = Temporal_Decomposition(self.args.std_kernel)

        self.embedding = SeriesEmbedding(
            self.args.seq_len_in, self.args.d_model, self.args.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    ssm=Mamba(
                        self.args.d_model,
                        self.args.d_state,
                        self.args.d_conv,
                        self.args.expand,
                    ),
                    ssm_r=Mamba(
                        self.args.d_model,
                        self.args.d_state,
                        self.args.d_conv,
                        self.args.expand,
                    ),
                    d_model=self.args.d_model,
                    d_ff=self.args.d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.activation,
                )
                for _ in range(self.args.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.args.d_model),
        )

        self.projector = nn.Linear(self.args.d_model, self.args.seq_len_out, bias=True)

        if self.args.ds_type == 'max':
            self.down_pool = nn.MaxPool1d(self.args.ds_window, return_indices=False)
        elif self.args.ds_type == 'avg':
            self.down_pool = nn.AvgPool1d(self.args.ds_window)
        elif self.args.ds_type == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(
                in_channels=self.args.num_nodes,
                out_channels=self.args.num_nodes,
                kernel_size=3,
                padding=padding,
                stride=self.args.ds_window,
                padding_mode='circular',
                bias=False,
            )

        self.ms_mixing = MultiScaleTrendMixing(
            self.args.seq_len_in,
            self.args.seq_len_out,
            self.args.num_nodes,
            self.args.ds_layers,
            self.args.ds_window,
        )

        self.linear_mappings = nn.ModuleList([
            nn.Linear(
                self.args.seq_len_in // (self.args.ds_window ** (layer)),
                self.args.seq_len_out,
            )
            for layer in range(self.args.ds_layers + 1)
        ])

        self.tre_w = nn.Parameter(
            torch.FloatTensor([1.0] * self.args.num_nodes),
            requires_grad=True,
        )

        self.node_embedding = nn.Parameter(
            torch.zeros(self.args.num_nodes, self.args.d_model)
        )
        nn.init.xavier_normal_(self.node_embedding)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data[..., 0]  # [Batch_size, Seq_len, Num_channels]

        if self.args.use_revin:
            x_in = self.revin(x_in, mode='norm')

        x_sea, _ = self.decom(x_in)

        # Embedding: [B, T, N] -> [B, N, E]
        x_emb = self.embedding(x_sea)

        # Add noe3 embedding
        B, _, _ = x_emb.shape()
        node_emb = self.node_embedding.unsqueeze(0).expand(B, -1, -1)
        x_emb = x_emb + node_emb

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(x_emb)

        sea_out = self.projector(enc_out).permute(
            0, 2, 1
        )  # (B, N, d_model) -> (B, N, T) -> (B, T, N)

        # Trend part: multi-scale processing
        ms_list = []
        ms_list.append(x_in)  # [B, T, N]

        x_ms = x_in.permute(0, 2, 1)
        for _ in range(self.args.ds_layers):
            x_sampling = self.down_pool(x_ms)  # [B, N, t_1/t_2/t_3 ... ]

            ms_list.append(x_sampling.permute(0, 2, 1))
            x_ms = x_sampling

        ms_trend_list = []
        for x in ms_list:
            _, x_tre = self.decom(x)
            ms_trend_list.append(x_tre)

        ms_trend_list = self.ms_mixing(ms_trend_list)

        # multi-scale mappings
        out_trend_list = []
        for i, trend in zip(range(len(ms_trend_list)), ms_trend_list, strict=True):
            trend_out = self.linear_mappings[i](trend.permute(0, 2, 1)).permute(0, 2, 1)
            out_trend_list.append(trend_out)

        tre_out = torch.stack(out_trend_list, dim=-1).sum(-1)

        # Weighted Sum
        prediction = sea_out + self.tre_w * tre_out
        prediction = self.revin(prediction, mode='denorm')

        y = prediction.unsqueeze(-1)

        return y
