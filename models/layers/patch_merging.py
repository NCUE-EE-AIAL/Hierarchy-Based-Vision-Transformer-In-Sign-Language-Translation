from torch import nn
import torch

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


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