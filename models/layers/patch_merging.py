from torch import nn
import torch


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for frame-wise input.

    Args:
        dim (int): Number of input channels (features) per frame.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)  # Reduce frame dimension and double feature dim
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        Forward function.
        x: (batch_size, frames, dim)
        Output: (batch_size, frames // 2, dim * 2)
        """
        B, F, C = x.shape
        assert F % 2 == 0, f"Number of frames ({F}) must be even."

        # Split frames into two groups: even and odd frames
        x_even = x[:, 0::2, :]  # (batch_size, frames // 2, dim)
        x_odd = x[:, 1::2, :]   # (batch_size, frames // 2, dim)

        # Concatenate along the feature dimension
        x = torch.cat([x_even, x_odd], dim=-1)  # (batch_size, frames // 2, dim * 2)

        # Apply normalization and reduction
        x = self.norm(x)  # (batch_size, frames // 2, dim * 2)
        x = self.reduction(x)  # Reduce the dimension if necessary

        return x