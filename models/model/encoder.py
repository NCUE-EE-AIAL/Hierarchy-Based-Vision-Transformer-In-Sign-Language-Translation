from torch import nn
import torch

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.patch_embeddings import PatchEmbedding
from models.layers.patch_merging import PatchMerging

# not to use this class
# split the embedding and encoder layer
class Encoder(nn.Module):

    def __init__(self,image_size, image_patch_size, max_frames, frame_patch_size, dim: tuple, ffn_hidden, n_head: tuple, drop_prob, enc_layers: tuple, device):
        super().__init__()
        n_layer1, n_layer2, n_layer3, n_layer4 = enc_layers
        dim1, dim2, dim3, dim4 = dim
        n_head1, n_head2, n_head3, n_head4 = n_head

        self.emb = PatchEmbedding(image_size=image_size,
                                  image_patch_size=image_patch_size,
                                  max_frames=max_frames,
                                  frame_patch_size=frame_patch_size,
                                  dim=dim1,
                                  drop_prob=drop_prob,
                                  device=device)

        self.layers1 = nn.ModuleList([EncoderLayer(dim=dim1,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head1,
                                                  drop_prob=drop_prob,
                                                  device=device)
                                     for _ in range(n_layer1)])
        self.patch_merge1 = PatchMerging(dim=dim1, device=device)

        self.layers2 = nn.ModuleList([EncoderLayer(dim=dim2,
                                                   ffn_hidden=ffn_hidden,
                                                   n_head=n_head2,
                                                   drop_prob=drop_prob,
                                                   device=device)
                                      for _ in range(n_layer2)])
        self.patch_merge2 = PatchMerging(dim=dim2, device=device)

        self.layers3 = nn.ModuleList([EncoderLayer(dim=dim3,
                                                   ffn_hidden=ffn_hidden,
                                                   n_head=n_head3,
                                                   drop_prob=drop_prob,
                                                   device=device)
                                      for _ in range(n_layer3)])
        self.patch_merge3 = PatchMerging(dim=dim3, device=device)

        self.layers4 = nn.ModuleList([EncoderLayer(dim=dim4,
                                                   ffn_hidden=ffn_hidden,
                                                   n_head=n_head4,
                                                   drop_prob=drop_prob,
                                                   device=device)
                                      for _ in range(n_layer4)])
    def forward(self, x):
        x = self.emb(x)

        for layer in self.layers1:
            x = layer(x, window_size=128)
        x = self.patch_merge1(x)

        for layer in self.layers2:
            x = layer(x, window_size=128)
        x = self.patch_merge2(x)

        for layer in self.layers3:
            x = layer(x, window_size=128)
        x = self.patch_merge3(x)

        for layer in self.layers4:
            x = layer(x, window_size=128)



        return x


