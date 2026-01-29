import torch
import torch.nn as nn
from mamba_ssm import Mamba  # type: ignore

from .embed import SeriesEmbedding
from .mamba_enc import Encoder, EncoderLayer


class SMamba(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']

        self.use_norm = model_args['use_norm']

        self.embedding = SeriesEmbedding(
            model_args['history_seq_len'],
            model_args['d_model'],
            model_args['emb_dropout'],
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=model_args['d_model'],
                        d_state=model_args['d_state'],
                        d_conv=model_args['d_conv'],
                        expand=model_args['expand'],
                    ),
                    Mamba(
                        d_model=model_args['d_model'],
                        d_state=model_args['d_state'],
                        d_conv=model_args['d_conv'],
                        expand=model_args['expand'],
                    ),
                    model_args['d_model'],
                    model_args['d_ff'],
                    dropout=model_args['ffn_dropout'],
                    activation=model_args['ffn_activation'],
                )
                for layer in range(model_args['e_layers'])
            ],
            norm=nn.LayerNorm(model_args['d_model']),
        )

        self.projector = nn.Linear(
            model_args['d_model'], model_args['future_seq_len'], bias=True
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
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.future_seq_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.future_seq_len, 1)
            )

        prediction = dec_out.unsqueeze(-1)

        return prediction
