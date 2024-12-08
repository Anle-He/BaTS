import torch
import torch.nn as nn


class Mamba(nn.Module):
    def __init__(self, **model_args)
        super(Mamba, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.e_layers = model_args['e_layers']

        self._build()

    
    def _build(self):

