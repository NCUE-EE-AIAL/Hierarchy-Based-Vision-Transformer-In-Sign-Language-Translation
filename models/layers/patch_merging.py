from torch import nn
import torch

import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for frame-wise input.

    Args:
        dim (int): Number of input channels (features) per frame.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        device (torch.device or str, optional): The device on which to place the module and its operations (default: None)
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, device=None):
        super().__init__()
        self.dim = dim
        self.device = device if device is not None else torch.device('cpu')

        # Move the layers to the device
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False).to(
            self.device)  # Reduce frame dimension and double feature dim
        self.norm = norm_layer(2 * dim).to(self.device)

    def forward(self, x):
        """
        Forward function.
        x: (batch_size, frames, dim)
        Output: (batch_size, frames // 2, dim * 2)
        """
        # Ensure x is moved to the correct device
        x = x.to(self.device)

        B, F, C = x.shape
        assert F % 2 == 0, f"Number of frames ({F}) must be even."

        # Split frames into two groups: even and odd frames
        x_even = x[:, 0::2, :]  # (batch_size, frames // 2, dim)
        x_odd = x[:, 1::2, :]  # (batch_size, frames // 2, dim)

        # Concatenate along the feature dimension
        x = torch.cat([x_even, x_odd], dim=-1)  # (batch_size, frames // 2, dim * 2)

        # Apply normalization and reduction
        x = self.norm(x)  # (batch_size, frames // 2, dim * 2)
        x = self.reduction(x)  # Reduce the dimension if necessary

        return x

if __name__ == "__main__":
    # Test the function with input shape (32, 512, 64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_merging_layer = PatchMerging(dim=64, device=device)

    # Create a dummy input tensor with shape (32, 512, 64)
    x = torch.randn(32, 512, 64).to(device)

    # Forward pass
    output = patch_merging_layer(x)
    print(output.shape)