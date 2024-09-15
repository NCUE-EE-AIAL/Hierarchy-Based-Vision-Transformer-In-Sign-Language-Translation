from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.mlp import Mlp
from models.layers.window_partition import window_partition, window_reverse


class EncoderLayer(nn.Module):

    def __init__(self, dim, ffn_hidden, n_head, drop_prob, device):
        super().__init__()
        self.window_attention = MultiHeadAttention(dim=dim, num_heads=n_head, drop_prob=drop_prob, device=device)
        self.norm1 = LayerNorm(dim=dim, device=device)

        self.mlp = Mlp(in_features=dim, hidden_features=ffn_hidden, drop=drop_prob, device=device)
        self.norm2 = LayerNorm(dim=dim, device=device)

    def forward(self, x, window_size):
        # window partition
        # _, N, _ = x.shape
        # x = window_partition(x, window_size)

        # compute self attention
        _x = x
        x = self.window_attention(q=x, k=x, v=x)

        # add and norm
        x = self.norm1(x + _x)
        
        # positionwise feed forward network
        _x = x
        x = self.mlp(x)
      
        # add and norm
        x = self.norm2(x + _x)

        # reverse window partition
        # x = window_reverse(x, window_size, N)
        return x
