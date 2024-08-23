import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.token_embeddings import TokenEmbedding

# not to use this class
# split the embedding and decoder layer
class Decoder(nn.Module):
    def __init__(self, max_len, dec_voc_size, dim, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.emb = TokenEmbedding(vocab_size=dec_voc_size,
                                  dim=dim,
                                  n=max_len)

        self.layers = nn.ModuleList([DecoderLayer(dim=dim,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(6)])

        self.linear = nn.Linear(dim, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask)

        # pass to LM head
        output = self.linear(x)
        return output

if __name__ == '__main__':
    # Initialize random input tensor with shape (batch_size, channels, frames, height, width)
    input_tensor = torch.rand(32, 512, 183)

    # Define model parameters
    image_size = (1, 183)
    image_patch_size = (1, 183)
    max_frames = 512
    frame_patch_size = 1
    dec_voc_size = 128
    d_model = 128
    ffn_hidden = 2048
    n_head = 8
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Initialize and test the Encoder model
    model = Decoder(image_size, image_patch_size, max_frames, frame_patch_size, dec_voc_size, d_model, ffn_hidden, n_head, drop_prob, device)
    output_tensor = model(input_tensor)

    print(output_tensor.shape)