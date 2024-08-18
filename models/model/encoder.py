from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


# not to use this class
# split the embedding and encoder layer
class Encoder(nn.Module):

    def __init__(self,image_size, image_patch_size, frames, frame_patch_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(image_size=image_size,
                                        image_patch_size=image_patch_size,
                                        frames=frames,
                                        frame_patch_size=frame_patch_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x