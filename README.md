# Hierarchy-Based-Vision-Transformer-In-Sign-Language-Translation

## Dataset
### How2Sign
Link: https://how2sign.github.io/
### Youtube-ASL
Consturcting

## Model Structure
![model structure](https://github.com/NCUE-EE-AIAL/Hierarchy-Based-Vision-Transformer-In-Sign-Language-Translation/blob/main/doc/heirarchy-based-transformer-workflow.png)

### Encoder
```python
class Encoder(nn.Module):

    def __init__(self,image_size, image_patch_size, max_frames, frame_patch_size, dim: tuple, ffn_hidden, n_head: tuple, drop_prob, enc_layers: tuple, device):
        super().__init__()
        n_layer1, n_layer2, n_layer3= enc_layers
        dim1, dim2, dim3= dim
        n_head1, n_head2, n_head3= n_head

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

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers1:
            x = layer(x, 128, src_mask)
        x, merge_mask = self.patch_merge1(x, src_mask)

        for layer in self.layers2:
            x = layer(x, 128, merge_mask)
        x, merge_mask = self.patch_merge2(x, merge_mask)

        for layer in self.layers3:
            x = layer(x, 128, merge_mask)

        return x, merge_mask
```
### Decoder
```python
class Decoder(nn.Module):
    def __init__(self, max_len, dec_voc_size, dim, ffn_hidden, n_head, drop_prob, dec_layers, device=None):
        super().__init__()
        self.emb = TokenEmbedding(vocab_size=dec_voc_size,
                                  dim=dim,
                                  n=max_len,
                                  device=device)

        self.layers = nn.ModuleList([DecoderLayer(dim=dim,
                                                  ffn_hidden=ffn_hidden,
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
```
