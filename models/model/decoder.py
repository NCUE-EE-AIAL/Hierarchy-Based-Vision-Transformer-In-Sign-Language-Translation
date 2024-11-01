import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.token_embeddings import TokenEmbedding

# not to use this class
# split the embedding and decoder layer
class Decoder(nn.Module):
    def __init__(self, max_len, dec_voc_size, dim, ffn_hidden_ratio, n_head, drop_prob, dec_layers, device=None):
        super().__init__()
        self.emb = TokenEmbedding(vocab_size=dec_voc_size,
                                  dim=dim,
                                  n=max_len,
                                  device=device)

        self.layers = nn.ModuleList([DecoderLayer(dim=dim,
                                                  ffn_hidden_ratio=ffn_hidden_ratio,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  device=device)
                                     for _ in range(dec_layers)])

        self.linear = nn.Linear(dim, dec_voc_size).to(device)

    def forward(self, trg, enc_src, trg_mask, cross_attn_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, cross_attn_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

