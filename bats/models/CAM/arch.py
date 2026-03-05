import math
from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from mamba_ssm import Mamba  # type: ignore
from .modules import GTR, SeriesEmbedding, MHCBlock, CrossDomainAlignment, RevIN


@dataclass
class CAMArgs:
    add_revin: bool
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    cycle_len: int
    cycle_pattern: str
    d_model: int
    d_ff: int
    d_state: int
    d_conv: int
    expand: int
    dropout: float
    e_layers: int
    num_streams: int
    period_len: int
    lpf: int


class CAM(nn.Module):
    def __init__(self, **model_args: Any) -> None:
        super().__init__()

        self.args = CAMArgs(**model_args)
        self._build()

    def _build(self) -> None:

        self.cycle_query = nn.Parameter(
            torch.zeros(self.args.cycle_len, self.args.num_nodes), requires_grad=True
        )
        self.gtr = GTR(seq_len=self.args.seq_len_in)
        self.embedding = SeriesEmbedding(
            seq_len_in=self.args.seq_len_in,
            d_model=self.args.d_model,
            dropout=self.args.dropout,
        )

        self.layers = nn.ModuleList([
            MHCBlock(
                ssm=Mamba(
                    d_model=self.args.d_model,
                    d_state=self.args.d_state,
                    d_conv=self.args.d_conv,
                    expand=self.args.expand,
                ),
                ssm_r=Mamba(
                    d_model=self.args.d_model,
                    d_state=self.args.d_state,
                    d_conv=self.args.d_conv,
                    expand=self.args.expand,
                ),
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                num_streams=self.args.num_streams,
            )
            for _ in range(self.args.e_layers)
        ])

        # 1. 低秩投影 (保持高效)
        self.bottleneck_dim = max(self.args.seq_len_out // 4, 32)
        self.freq_proj = nn.Sequential(
            nn.Linear(self.args.seq_len_out, self.bottleneck_dim),
            nn.GELU(),
            nn.Linear(self.bottleneck_dim, self.args.d_model),
        )

        # 2. 核心对齐模块
        self.alignment = CrossDomainAlignment(self.args.d_model)

        # 3. 最终预测层
        self.decoder = nn.Linear(self.args.d_model, self.args.seq_len_out)

        self.final_agg = nn.Parameter(
            torch.ones(self.args.num_streams) / self.args.num_streams
        )

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.args.period_len + 1,
            stride=1,
            padding=self.args.period_len // 2,
            padding_mode='zeros',
            bias=False,
        )
        self.seg_num_y = math.ceil(self.args.seq_len_out / self.args.period_len)
        self.freq_linear1 = nn.Linear(self.args.lpf, 2, bias=False).to(torch.cfloat)
        self.freq_linear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

        self.revin = RevIN(self.args.d_model)

        self.node_embedding = nn.Parameter(
            torch.zeros(self.args.num_nodes, self.args.d_model)
        )
        self.phase_embedding = nn.Embedding(self.args.cycle_len, self.args.d_model)
        nn.init.xavier_normal_(self.node_embedding)
        nn.init.xavier_normal_(self.phase_embedding.weight)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x_in = history_data[..., 0]
        B, L, N = x_in.shape

        # RevIN
        if self.args.add_revin:
            x_in = self.revin(x_in, mode='norm')

        # Frequency-domain Branch
        xf_in = x_in.permute(0, 2, 1)
        if self.args.cycle_pattern == 'daily':
            cycle_index = history_data[..., 1] * self.args.cycle_len
            cycle_index = cycle_index[:, -1, 0]
        elif self.args.cycle_pattern == 'daily&weekly':
            cycle_index = (
                history_data[..., 1] * self.args.cycle_len * 7
                + history_data[..., 2] * 7
            )
            cycle_index = cycle_index[:, -1, 0]
        else:
            raise ValueError(f'Unsupported cycle pattern: {self.args.cycle_pattern}')

        gather_idx = (
            cycle_index.view(-1, 1)
            + torch.arange(self.args.cycle_len, device=cycle_index.device).view(1, -1)
        ) % self.args.cycle_len
        query_input = self.cycle_query[gather_idx].permute(0, 2, 1)
        global_information = self.gtr(xf_in, query_input)
        xf_in = xf_in + global_information

        xf = (
            self.conv1d(xf_in.reshape(-1, 1, self.args.seq_len_in)).reshape(
                -1, self.args.num_nodes, self.args.seq_len_in
            )
            + xf_in
        )
        xf = xf.reshape(B, N, -1, self.args.period_len).permute(0, 1, 3, 2)

        x_fft = torch.fft.fft(xf, dim=3)[:, :, :, : self.args.lpf]
        x_fft = self.freq_linear1(x_fft)
        x_fft = self.freq_linear2(x_fft).reshape(B, N, self.args.period_len, -1)
        x_rfft = torch.fft.ifft(x_fft, dim=3).float()
        f_out = (
            x_rfft.permute(0, 1, 3, 2).reshape(B, N, -1).permute(0, 2, 1)
        )  # [B, L_out, N]

        # Time-domain Branch
        x_emb = self.embedding(x_in)

        node_emb = self.node_embedding.unsqueeze(0).expand(B, -1, -1)
        phase_emb = self.phase_embedding(cycle_index).unsqueeze(1).expand(-1, N, -1)

        x_emb = x_emb + node_emb + phase_emb + node_emb * phase_emb

        h_stream = x_emb.unsqueeze(1).repeat(1, self.args.num_streams, 1, 1, 1)
        for layer in self.layers:
            h_stream = layer(h_stream)
        w_final = torch.softmax(self.final_agg, dim=0)
        t_out = torch.einsum('s,bsne->bne', w_final, h_stream)  # [B, N, D]

        # ==================== 执行改进流程 ====================
        # 1. 低秩投影
        f_latent = self.freq_proj(f_out.permute(0, 2, 1))

        # 2. 稳定对齐融合
        # 初始状态：gate 全为 0 (因为 zeros + sigmoid)
        # 输出：aligned_feat = t_out + 0 * f_orth = t_out
        aligned_feat = self.alignment(t_out, f_latent)

        dec_out = self.decoder(aligned_feat).permute(0, 2, 1)

        # RevIN Denorm
        if self.args.add_revin:
            dec_out = self.revin(dec_out, mode='denorm')

        y = dec_out.unsqueeze(-1)

        return y
