from torch import nn
import torch

from models.embedding.transformer_embedding import TransformerEmbedding
from models.layers.window_attention import WindowAttention, window_partition

from models.model.transformer import Transformer
class PartitionTransformer(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, dim, max_len, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., device):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_patch_embedding = TransformerEmbedding(image_size, image_patch_size, frames, frame_patch_size, dim, max_len, dropout, device)
        self.dropout = nn.Dropout(emb_dropout)

        self.window_attention = WindowAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)

        # not finished yet
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        # patch embedding
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        # pos drop (vit use, while swin transformer not)
        # x = self.dropout(x)

        # attention
        # not finished yet

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)