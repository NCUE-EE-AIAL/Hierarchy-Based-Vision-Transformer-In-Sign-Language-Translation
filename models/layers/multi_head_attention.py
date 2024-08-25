from torch import nn


class MultiHeadAttention(nn.Module):
    r""" Window based multi-head self attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, drop_prob, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_prob)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k ,v, mask=None):
        """
        Args:
            q, k, v: input features with shape of (num_windows*B, N, C)
        """
        B_, N_q, C = q.shape
        _, N_k, _ = k.shape

        q = self.w_q(q).reshape(B_, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.w_k(k).reshape(B_, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.w_v(v).reshape(B_, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            print("q shape: ", q.shape)
            print("mask after unsqueeze: ", mask.shape)
            attn = attn.masked_fill(mask == 0, -10000)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops