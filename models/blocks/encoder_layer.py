from torch import nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.mlp import Mlp
from models.layers.window_partition import window_partition, window_reverse
from models.layers.mask_operation import mask_partition


class EncoderLayer(nn.Module):

    def __init__(self, dim, ffn_hidden, n_head, drop_prob, device):
        super().__init__()
        self.window_attention = MultiHeadAttention(dim=dim, num_heads=n_head, drop_prob=drop_prob, device=device)
        self.norm1 = nn.LayerNorm(normalized_shape=dim, device=device)

        self.mlp = Mlp(in_features=dim, hidden_features=ffn_hidden, drop=drop_prob, device=device)
        self.norm2 = nn.LayerNorm(normalized_shape=dim, device=device)

    def forward(self, x, window_size, src_mask):
        # window partition
        _, N, _ = x.shape
        x = window_partition(x, window_size)
        mask = mask_partition(src_mask, window_size)

        # compute self attention
        _x = x
        x = self.norm1(x)
        x = self.window_attention(q=x, k=x, v=x, mask=mask)
        x = x + _x  # Residual connection
        
        # positionwise feed forward network
        _x = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + _x  # Residual connection

        # reverse window partition
        x = window_reverse(x, window_size, N)
        return x
