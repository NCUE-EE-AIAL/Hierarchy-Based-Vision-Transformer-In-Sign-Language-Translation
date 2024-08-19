import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        constructor of positional encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model, device=device))
        self.to(device)

    def forward(self, x):
        # x is expected to have shape [batch_size, seq_len, d_model]
        print("x shape before batch_size, n, d_model = x.size()", x.shape)
        batch_size, n, d_model = x.size()
        print("pos_embedding shape before n = self.pos_embedding[:, :n, :].shape", self.pos_embedding[:, :n, :].shape)
        # Ensure pos_embedding has the same device and dtype as x
        return self.pos_embedding[:, :n, :]