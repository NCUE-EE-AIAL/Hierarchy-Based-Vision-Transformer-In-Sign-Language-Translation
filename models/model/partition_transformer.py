from torch import nn
import torch

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class PartitionTransformer(nn.Module):
    def __init__(self, trg_pad_idx, image_size, image_patch_size, max_frames, frame_patch_size, dim, ffn_hidden, n_head, drop_prob, max_len, dec_voc_size, enc_layers, dec_layers, device):
        super().__init__()
        self.device = device
        self.trg_pad_idx = trg_pad_idx
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
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src, trg_mask)
        return output

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3).to(self.device)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

if __name__ == '__main__':
    from models.model.encoder import Encoder
    from transformers import T5Tokenizer

    # Initialize random input tensor with shape (batch_size, channels, frames, height, width)
    input_tensor = torch.rand(3, 1, 512, 1, 183)

    # Define model parameters
    image_size = (1, 183)
    image_patch_size = (1, 183)
    max_frames = 512
    frame_patch_size = 1
    dim = 128
    ffn_hidden = 2048
    n_head = 8
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    max_len = 256
    dec_voc_size = 32128


    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    sentences = ["sentence one we are happy", "sentence two", "sentence three"]
    encoding = tokenizer(
        sentences,
        padding="longest",  # Pad to the longest sentence in the batch
        max_length=256,  # Maximum length to truncate or pad to
        truncation=True,  # Truncate if the sentence exceeds max_length
        return_tensors="pt"  # Return PyTorch tensors
    )
    print(encoding.input_ids.shape)

    model = PartitionTransformer(trg_pad_idx=tokenizer.pad_token_id,
                                 image_size=image_size,
                                 image_patch_size=image_patch_size,
                                 max_frames=max_frames,
                                 frame_patch_size=frame_patch_size,
                                 dim=dim,
                                 ffn_hidden=ffn_hidden,
                                 n_head=n_head,
                                 drop_prob=drop_prob,
                                 dec_voc_size=dec_voc_size,
                                 max_len=max_len,
                                 device=device)

    output_tensor = model(input_tensor, encoding.input_ids)
    print(output_tensor.shape)