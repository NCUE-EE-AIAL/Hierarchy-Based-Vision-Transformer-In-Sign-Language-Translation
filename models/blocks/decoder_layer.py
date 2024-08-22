from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.window_attention import WindowAttention, window_partition, window_reverse
from models.layers.Mlp import Mlp


class DecoderLayer(nn.Module):

    def __init__(self, dim, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.self_attention = WindowAttention(dim=dim, num_heads=n_head, attn_drop=drop_prob, proj_drop=drop_prob)
        self.norm1 = LayerNorm(dim=dim)

        self.enc_dec_attention = WindowAttention(dim=dim, num_heads=n_head, attn_drop=drop_prob, proj_drop=drop_prob)
        self.norm2 = LayerNorm(dim=dim)

        self.ffn = Mlp(in_features=d_model, hidden_features=ffn_hidden, drop=drop_prob)
        self.norm3 = LayerNorm(dim=dim)

    def forward(self, dec, enc, window_size):
        # window partition
        _, N, _ = dec.shape
        dec = window_partition(dec, window_size)
        enc = window_partition(enc, window_size)

        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec)
        
        # 2. add and norm
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc)
            
            # 4. add and norm
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.norm3(x + _x)

        # reverse window partition
        x = window_reverse(x, window_size, N)
        return x
