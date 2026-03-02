import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes: int, subgraph_size: int, node_dim: int, alpha: float):
        super().__init__()

        self.subgraph_size = subgraph_size
        self.alpha = alpha

        self.emb1 = nn.Embedding(num_nodes, node_dim)
        self.emb2 = nn.Embedding(num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb1.weight)
        nn.init.xavier_uniform_(self.emb2.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros_like(adj)

        topk_values, topk_indices = adj.topk(self.subgraph_size, dim=1)
        mask.scatter_(1, topk_indices, topk_values.fill_(1))

        return adj * mask


class GraphConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ncwl,vw->ncvl', x, A)


class MixProp(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super().__init__()

        self.graph_conv = GraphConv()
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=1)
        self.gdep = gdep
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Add self-loop
        adj = adj + torch.eye(adj.size(0), device=x.device)

        # Degree normalization
        d = adj.sum(dim=1).clamp(min=1e-6)
        a = adj / d.unsqueeze(1)

        h = x
        out = [h]

        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.graph_conv(h, a)
            out.append(h)

        ho = torch.cat(out, dim=1)
        ho = self.dropout(ho)
        ho = self.mlp(ho)

        return ho


class DilatedInception(nn.Module):
    def __init__(self, cin: int, cout: int, dilation_factor: int = 2):
        super().__init__()

        self.kernel_set = [2, 3, 6, 7]

        # Ensure cout is divisible by number of kernels
        assert cout % len(self.kernel_set) == 0, (
            f'cout ({cout}) must be divisible by number of kernels ({len(self.kernel_set)})'
        )

        cout_per_kernel = cout // len(self.kernel_set)

        self.tconv = nn.ModuleList([
            nn.Conv2d(
                cin,
                cout_per_kernel,
                kernel_size=(1, kern),
                dilation=(1, dilation_factor),
                padding=(0, (kern - 1) // 2 * dilation_factor),
            )
            for kern in self.kernel_set
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        outputs = [conv(x) for conv in self.tconv]

        # Align temporal dimensions
        min_len = min(out.size(-1) for out in outputs)
        outputs = [out[..., -min_len:] for out in outputs]

        return torch.cat(outputs, dim=1)


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(
                input,
                tuple(input.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        else:
            return F.layer_norm(
                input, tuple(input.shape[1:]), self.weight, self.bias, self.eps
            )

    def extra_repr(self):
        return (
            '{normalized_shape}, eps={eps}, '
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
        )
