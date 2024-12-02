import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, 
                 history_seq_len, 
                 d_model, 
                 dropout):
        super(DataEmbedding_inverted, self).__init__()

        self.ValueEmb = nn.Linear(history_seq_len, d_model)
        self.EmbDropout = nn.Dropout(p=dropout)


    def forward(self, x_in):
        # x_in: (batch_size, history_seq_len <-> num_nodes)
        x_in = x_in.permute(0, 2, 1)

        x_emb = self.ValueEmb(x_in) # x_emb: (batch_size, num_nodes, d_model)
        x_emb = self.EmbDropout(x_emb)

        return x_emb