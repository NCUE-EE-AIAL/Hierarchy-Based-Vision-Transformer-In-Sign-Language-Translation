from torch import nn

class MultiHeadAttention(nn.Module):
    r""" Window based multi-head self attention

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout ratio of attention weight. Default: 0.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        device (torch.device, optional): Device to which the tensors should be moved. Default: None
    """

    def __init__(self, dim, num_heads, drop_prob, qkv_bias=True, qk_scale=None, device=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.device = device  # Store device
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define layers and move them to the specified device
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias).to(device)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias).to(device)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias).to(device)

        self.attn_drop = nn.Dropout(drop_prob).to(device)
        self.proj = nn.Linear(dim, dim).to(device)
        self.proj_drop = nn.Dropout(drop_prob).to(device)

        self.softmax = nn.Softmax(dim=-1).to(device)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: input features with shape of (num_windows*B, N, C)
            mask: optional attention mask of shape (num_windows*B, num_heads, N_q, N_k)
        """
        B_, N_q, C = q.shape
        _, N_k, _ = k.shape

        # Ensure the inputs are on the right device
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        q = self.w_q(q).reshape(B_, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.w_k(k).reshape(B_, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.w_v(v).reshape(B_, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -10000)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x