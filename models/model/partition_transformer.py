from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class PartitionTransformer(nn.Module):
    def __init__(self, *, image_size, image_patch_size, max_frames, frame_patch_size, dim, max_len, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.encoder = Encoder(image_size=image_size,
                               image_patch_size=image_patch_size,
                               max_frames=max_frames,
                               frame_patch_size=frame_patch_size,
                               dim=dim,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               drop_prob=drop_prob,
                               device=device)

        self.decoder = Decoder(max_len=max_len,
                               dec_voc_size=dec_voc_size,
                               dim=dim,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               drop_prob=drop_prob)
    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src, trg_mask)
        return output