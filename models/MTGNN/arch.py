import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, fields

from .modules import (
    GraphConstructor,
    MixProp,
    DilatedInception,
    LayerNorm,
)


@dataclass
class MTGNNConfig:
    receptive_field: int
    history_seq_len: int
    gcn_true: bool
    gcn_depth: int
    buildA_true: bool
    num_nodes: int
    subgraph_size: int
    node_dim: int
    tanhalpha: float
    propalpha: float
    dropout: float
    layers: int
    in_dim: int
    out_dim: int
    residual_channels: int
    dilation_exponential: int
    conv_channels: int
    skip_channels: int
    end_channels: int


class MTGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

        self.config = MTGNNConfig(**model_args)
        for field in fields(self.config):
            setattr(self, field.name, getattr(self.config, field.name))

        self._build()

    def _build(self):

        device = (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.idx = torch.arange(self.config.num_nodes, device=device)

        self.gc = GraphConstructor(
            num_nodes=self.config.num_nodes,
            subgraph_size=self.config.subgraph_size,
            node_dim=self.config.node_dim,
            alpha=self.config.tanhalpha,
        )

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=self.config.in_dim,
            out_channels=self.config.residual_channels,
            kernel_size=(1, 1),
        )

        kernel_size = 7
        if self.config.dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (self.config.dilation_exponential**self.config.layers - 1)
                / (self.config.dilation_exponential - 1)
            )
        else:
            self.receptive_field = self.config.layers * (kernel_size - 1) + 1

        for i in range(1):
            if self.config.dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (self.config.dilation_exponential**self.config.layers - 1)
                    / (self.config.dilation_exponential - 1)
                )
            else:
                rf_size_i = i * self.config.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.config.layers + 1):
                if self.config.dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (self.config.dilation_exponential**j - 1)
                        / (self.config.dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    DilatedInception(
                        self.config.residual_channels,
                        self.config.conv_channels,
                        dilation_factor=new_dilation,
                    )
                )
                self.gate_convs.append(
                    DilatedInception(
                        self.config.residual_channels,
                        self.config.conv_channels,
                        dilation_factor=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=self.config.conv_channels,
                        out_channels=self.config.residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                if self.config.history_seq_len > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=self.config.conv_channels,
                            out_channels=self.config.skip_channels,
                            kernel_size=(
                                1,
                                self.config.history_seq_len - rf_size_j + 1,
                            ),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=self.config.conv_channels,
                            out_channels=self.config.skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(
                        MixProp(
                            self.config.conv_channels,
                            self.config.residual_channels,
                            self.config.gcn_depth,
                            self.config.dropout,
                            self.config.propalpha,
                        )
                    )
                    self.gconv2.append(
                        MixProp(
                            self.config.conv_channels,
                            self.config.residual_channels,
                            self.config.gcn_depth,
                            self.config.dropout,
                            self.config.propalpha,
                        )
                    )

                if self.config.history_seq_len > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (
                                self.config.residual_channels,
                                self.config.num_nodes,
                                self.seq_length - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (
                                self.config.residual_channels,
                                self.config.num_nodes,
                                self.receptive_field - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= self.config.dilation_exponential

        self.layers = self.config.layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=self.config.skip_channels,
            out_channels=self.config.end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.config.end_channels,
            out_channels=self.config.out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        if self.config.history_seq_len > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=self.config.in_dim,
                out_channels=self.config.skip_channels,
                kernel_size=(1, self.config.history_seq_len),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=self.config.residual_channels,
                out_channels=self.config.skip_channels,
                kernel_size=(1, self.config.history_seq_len - self.receptive_field + 1),
                bias=True,
            )

        else:
            self.skip0 = nn.Conv2d(
                in_channels=self.config.in_dim,
                out_channels=self.config.skip_channels,
                kernel_size=(1, self.config.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=self.config.residual_channels,
                out_channels=self.config.skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self.idx = torch.arange(self.config.num_nodes).to(device)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x_in = history_data.permute(0, 3, 2, 1)
        seq_len = x_in.size(3)
        assert seq_len == self.config.history_seq_len, (
            'input sequence length not equal to preset sequence length'
        )
        if seq_len < self.config.receptive_field:
            x_in = nn.functional.pad(
                x_in, (self.config.receptive_field - seq_len, 0, 0, 0)
            )

        if self.config.gcn_true:
            adp = self.gc(self.idx)

        x_enc = self.start_conv(x_in)

        skip = self.skip0(F.dropout(x_in, self.config.dropout, training=self.training))
        for i in range(self.config.layers):
            residual = x_enc
            filter = self.filter_convs[i](x_enc)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x_enc)
            gate = torch.sigmoid(gate)
            x_enc = filter * gate
            x_enc = F.dropout(x_enc, self.config.dropout, training=self.training)
            s = x_enc
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.config.gcn_true:
                x_enc = self.gconv1[i](x_enc, adp) + self.gconv2[i](
                    x_enc, adp.transpose(1, 0)
                )
            else:
                x_enc = self.residual_convs[i](x_enc)

            x_enc = x_enc + residual[:, :, :, -x_enc.size(3) :]
            x_enc = self.norm[i](x_enc, self.idx)

        skip = self.skipE(x_enc) + skip
        x_enc = F.relu(skip)
        x_enc = F.relu(self.end_conv_1(x_enc))
        out = self.end_conv_2(x_enc)

        return out.unsqueeze(-1)
