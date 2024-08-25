from torch import nn
import torch

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.patch_embeddings import PatchEmbedding


# not to use this class
# split the embedding and encoder layer
class Encoder(nn.Module):

    def __init__(self,image_size, image_patch_size, max_frames, frame_patch_size, dim, ffn_hidden, n_head, drop_prob, device):
        super().__init__()
        self.emb = PatchEmbedding(image_size=image_size,
                                  image_patch_size=image_patch_size,
                                  max_frames=max_frames,
                                  frame_patch_size=frame_patch_size,
                                  dim=dim,
                                  drop_prob=drop_prob,
                                  device=device)

        self.layer = EncoderLayer(dim=dim,
                                  ffn_hidden=ffn_hidden,
                                  n_head=n_head,
                                  drop_prob=drop_prob)


    def forward(self, x):
        x = self.emb(x)

        for i in range(1):
            x = self.layer(x, window_size=64)
        for i in range(1):
            x = self.layer(x, window_size=128)
        for i in range(1):
            x = self.layer(x, window_size=256)
        for i in range(1):
            x = self.layer(x, window_size=512)

        return x


