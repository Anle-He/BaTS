import torch
import torch.nn as nn

from mamba_ssm import Mamba

from .RevIN import RevIN
from .Embed import DataEmbedding_inverted
from .MambaEnc import Encoder, EncoderLayer


class SDMamba(nn.Module):
    """
    Paper:
        - 
        - 
    Official Code:
        -
    Other Implementations can be found at:
        - 
    """
    def __init__(SDMamba, **model_args):
        super(SDMamba, self).__init__()

    self.history_seq_len = model_args['history_seq_len']
    self.future_seq_len = model_args['future_seq_len']
    self.num_channels = model_args['num_channels']
    self.d_model = model_args['d_model']

    self.use_revin = model_args['use_revin']
    self.emb_dropout = model_args['emb_dropout']

    self.mamba_mode = model_args['mamba_mode']
    assert self.mamba_mode in ['bi-directional', 'uni-directional']

    self.e_layers = model_args['e_layers']
    self.d_state = model_args['d_state']
    self.d_conv = model_args['d_conv']
    self.expand = model_args['expand']

    self.d_ff = model_args['d_ff']
    self.ffn_dropout = model_args['ffn_dropout']
    self.ffn_activation = model_args['ffn_activation']

    self._build()


    def _build(self):
        self.revin = RevIN(self.num_channels) if self.use_revin else None
        self.NodeEmbed = DataEmbedding_inverted(self.history_seq_len, self.d_model, self.emb_dropout)

        if self.mamba_mode == 'bi-directional':
            self.Encoder = Encoder(
                [
                    EncoderLayer(
                        ssm = Mamba(
                            d_model = self.d_model,
                            d_state = self.d_state,
                            d_conv = self.d_conv,
                            expand = self.expand
                        ),
                        ssm_r = Mamba(
                            d_model = self.d_model,
                            d_state = self.d_state,
                            d_conv = self.d_conv,
                            expand = self.expand
                        ),
                        d_model = self.d_model,
                        d_ff = self.d_ff,
                        dropout = self.ffn_dropout,
                        activation = self.ffn_activation
                    ) for _ in range(self.e_layers)
                ],
                norm = nn.LayerNorm(self.d_model)
            )
        else:
            self.Encoder = Encoder(
                [
                    EncoderLayer(
                        ssm = Mamba(
                            d_model = self.d_model,
                            d_state = self.d_state,
                            d_conv = self.d_conv,
                            expand = self.expand
                        ),
                        ssm_r = None,
                        d_model = self.d_model,
                        d_ff = self.d_ff,
                        dropout = self.ffn_dropout,
                        activation = self.ffn_activation
                    ) for _ in range(self.e_layers)
                ],
                norm = nn.LayerNorm(self.d_model)
            )

        self.Projector = nn.Linear(self.d_model, self.future_seq_len, bias=True)

    
    def forward(self, 
                history_data: torch.Tensor) -> torch.Tensor:
        
        x_in = history_data[:, :, :, 0] # Only the target feature is used.

        if self.use_revin:
            x_in = self.revin(x_in, mode='norm')

        x_emb = self.NodeEmbed(x_in) # (B, T, N) -> (B, N, d_model) 

        enc_out = self.Encoder(x_emb) # (B, N, d_model) -> (B, N, d_model)

        dec_out = self.Projector(enc_out).permute(0, 2, 1) # (B, N, d_model) -> (B, T, N)

        if self.use_revin:
            dec_out = self.revin(dec_out, mode='denorm')

        prediction = dec_out.unsqueeze(-1)

        return prediction