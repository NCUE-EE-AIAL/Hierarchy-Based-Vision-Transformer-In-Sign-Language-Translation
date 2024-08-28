import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-12, device=None):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim).to(device))
        self.beta = nn.Parameter(torch.zeros(dim).to(device))
        self.eps = eps
        self.device = device  # Store device

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

