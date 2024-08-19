from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.window_attention import WindowAttention, window_partition, window_reverse
from models.layers.Mlp import Mlp


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.window_attention = WindowAttention(dim=d_model, num_heads=n_head, attn_drop=drop_prob, proj_drop=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)

        self.ffn = Mlp(in_features=d_model, hidden_features=ffn_hidden, drop=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)

    def forward(self, x, window_size):
        # window partition
        _, N, _ = x.shape
        x = window_partition(x, window_size)

        # compute self attention
        _x = x
        x = self.window_attention(q=x, k=x, v=x)

        # add and norm
        x = self.norm1(x + _x)
        
        # positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # add and norm
        x = self.norm2(x + _x)

        # reverse window partition
        x = window_reverse(x, window_size, N)
        return x
