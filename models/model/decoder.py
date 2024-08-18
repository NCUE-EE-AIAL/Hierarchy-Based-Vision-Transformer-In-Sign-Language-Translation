import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

# not to use this class
# split the embedding and decoder layer
class Decoder(nn.Module):
    def __init__(self, image_size, image_patch_size, frames, frame_patch_size, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(image_size=image_size,
                                        image_patch_size=image_patch_size,
                                        frames=frames,
                                        frame_patch_size=frame_patch_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output