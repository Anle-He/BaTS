import torch
import torch.nn as nn

from .embed import DataEmbedding_inverted
from .transformer_blocks import Encoder, EncoderLayer
from .attention import FullAttention, AttentionLayer


class iTransformer(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.d_model = model_args['d_model']
        self.d_ff = model_args['d_ff']

        self.dropout = model_args['dropout']
        self.n_heads = model_args['n_heads']
        self.activation = model_args['activation']
        self.e_layers = model_args['e_layers']

        self.use_norm = model_args['use_norm']

        self.embedding = DataEmbedding_inverted(
            self.history_seq_len, self.d_model, self.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        self.projector = nn.Linear(self.d_model, self.future_seq_len, bias=True)

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
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.future_seq_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.future_seq_len, 1)
            )

        return dec_out.unsqueeze(-1)
