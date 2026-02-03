import torch.nn as nn


class MTGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__

        self.num_nodes = model_args['num_nodes']
