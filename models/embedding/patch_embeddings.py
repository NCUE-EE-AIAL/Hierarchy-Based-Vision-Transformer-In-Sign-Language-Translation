from torch import nn
import torch

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, image_patch_size, max_frames, frame_patch_size, dim, channels=1, drop_prob=0.1, device=None):
        """
        Args:
            image_size (tuple[int]): dimensions of the image shaped as (height, width)
            image_patch_size (tuple[int]): dimensions of the image patch shaped as (ph, pw)
            max_frames (int): number of frames
            frame_patch_size (int): number of patches in a frame
            dim (int): embedding dimension
            channels (int): number of channels in the image

        Returns:
            nn.Embedding: embedding layer shaped as (batch, num_patches, dim)
        """
        super().__init__()
        self.device = device
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert max_frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        # num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, dim, device=device))

        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = x.to(self.device)
        patch_emb = self.to_patch_embedding(x)

        batch_size, n, d_model = patch_emb.size()
        pos_emb = self.pos_embedding[:, :n, :]
        return self.drop_out(patch_emb + pos_emb)