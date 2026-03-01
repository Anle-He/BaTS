import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True
        )

        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.act = nn.ReLU()

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:

        hidden = self.fc2(self.drop(self.act(self.fc1(x_emb))))  # MLP
        out = hidden + x_emb  # Residual

        return out
