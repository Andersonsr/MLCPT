from torch import nn
import torch


class Mapper(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        self.k = k
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * k * 2),
            nn.GELU(),
            nn.Linear(output_dim * k * 2, output_dim * k),
        )

    def forward(self, x):
        _x = self.mlp(x)
        return _x.view(x.shape[0], self.k, self.output_dim)


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)



