from torch import nn
import torch

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class HierarchicalTransformer(nn.Module):
    def __init__(self, pad_idx, image_size, image_patch_size, max_frames, frame_patch_size, dim, ffn_hidden, n_head, drop_prob, max_len, dec_voc_size, enc_layers, dec_layers, device):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.encoder = Encoder(image_size=image_size,
                               image_patch_size=image_patch_size,
                               max_frames=max_frames,
                               frame_patch_size=frame_patch_size,
                               dim=dim,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               drop_prob=drop_prob,
                               enc_layers=enc_layers,
                               device=device)

        self.decoder = Decoder(max_len=max_len,
                               dec_voc_size=dec_voc_size,
                               dim=dim[-1],
                               ffn_hidden=ffn_hidden,
                               n_head=n_head[-1],
                               drop_prob=drop_prob,
                               dec_layers=dec_layers,
                               device=device)
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src, merge_mask = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, merge_mask)
        return output

    def make_src_mask(self, src):
        batch_size, _, frames, _, width = src.shape

        non_pad_elements = (src != 0).sum(dim=-1)  # Shape: (batch_size, 1, 512, 1)
        src_mask = (non_pad_elements > 0).squeeze(-1).unsqueeze(1)  # Shape: (batch_size, 1, 1, 512)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3).to(self.device)  # Shape: (batch_size, 1, 512, 1)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask