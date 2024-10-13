from torch import nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.mlp import Mlp
from models.layers.windows_operation import windows_partition, windows_partition_reverse, windows_shift, windows_shift_reverse


class EncoderLayer(nn.Module):

    def __init__(self, dim, ffn_hidden, n_head, drop_prob, device):
        super().__init__()
        self.window_attention1 = MultiHeadAttention(dim=dim, num_heads=n_head, drop_prob=drop_prob, device=device)
        self.norm1 = nn.LayerNorm(normalized_shape=dim, device=device)

        self.mlp1 = Mlp(in_features=dim, hidden_features=ffn_hidden, drop=drop_prob, device=device)
        self.norm2 = nn.LayerNorm(normalized_shape=dim, device=device)

        self.window_attention2 = MultiHeadAttention(dim=dim, num_heads=n_head, drop_prob=drop_prob, device=device)
        self.norm3 = nn.LayerNorm(normalized_shape=dim, device=device)

        self.mlp2 = Mlp(in_features=dim, hidden_features=ffn_hidden, drop=drop_prob, device=device)
        self.norm4 = nn.LayerNorm(normalized_shape=dim, device=device)

    def forward(self, x, window_size, src_mask):
        # window partition
        _, N, _ = x.shape
        x, mask = windows_partition(x, window_size, src_mask)

        # compute self attention
        _x = x
        x = self.norm1(x)
        x = self.window_attention1(q=x, k=x, v=x, mask=mask)
        x = x + _x  # Residual connection
        
        # positionwise feed forward network
        _x = x
        x = self.norm2(x)
        x = self.mlp1(x)
        x = x + _x  # Residual connection

        # reverse window partition
        x, _ = windows_partition_reverse(x, window_size, N)

        # shift
        if not window_size >= N:
            x, mask = windows_shift(x, window_size//2, src_mask)
            x, mask = windows_partition(x, window_size, src_mask)

        # compute self attention
        _x = x
        x = self.norm3(x)
        x = self.window_attention2(q=x, k=x, v=x, mask=mask)
        x = x + _x  # Residual connection

        # positionwise feed forward network
        _x = x
        x = self.norm4(x)
        x = self.mlp2(x)
        x = x + _x  # Residual connection

        # reverse window partition
        if not window_size >= N:
            x, _ = windows_partition_reverse(x, window_size, N)
            x, _ = windows_shift_reverse(x, window_size//2)

        return x
