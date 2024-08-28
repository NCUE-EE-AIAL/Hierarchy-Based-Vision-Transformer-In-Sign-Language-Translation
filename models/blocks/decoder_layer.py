from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.mlp import Mlp


class DecoderLayer(nn.Module):

    def __init__(self, dim, ffn_hidden, n_head, drop_prob, device):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim=dim, num_heads=n_head, drop_prob=drop_prob, device=device)
        self.norm1 = LayerNorm(dim=dim, device=device)

        self.enc_dec_attention = MultiHeadAttention(dim=dim, num_heads=n_head, drop_prob=drop_prob, device=device)
        self.norm2 = LayerNorm(dim=dim, device=device)

        self.mlp = Mlp(in_features=dim, hidden_features=ffn_hidden, drop=drop_prob, device=device)
        self.norm3 = LayerNorm(dim=dim, device=device)

    def forward(self, dec, enc, trg_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=None)
            
            # 4. add and norm
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.mlp(x)
        
        # 6. add and norm
        x = self.norm3(x + _x)
        return x
