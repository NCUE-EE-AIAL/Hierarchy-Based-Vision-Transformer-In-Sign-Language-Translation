from torch import nn
import torch

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


# not to use this class
# split the embedding and encoder layer
class Encoder(nn.Module):

    def __init__(self,image_size, image_patch_size, max_frames, frame_patch_size, d_model, ffn_hidden, n_head, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(image_size=image_size,
                                        image_patch_size=image_patch_size,
                                        frames=max_frames,
                                        frame_patch_size=frame_patch_size,
                                        d_model=d_model,
                                        drop_prob=drop_prob,
                                        device=device)

        # self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
        #                                           ffn_hidden=ffn_hidden,
        #                                           n_head=n_head,
        #                                           drop_prob=drop_prob)
        #                              for _ in range(n_layers)])

        self.layer = EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob)


    def forward(self, x):
        x = self.emb(x)

        for i in range(2):
            x = self.layer(x, window_size=64)
        for i in range(6):
            x = self.layer(x, window_size=128)
        for i in range(2):
            x = self.layer(x, window_size=256)
        for i in range(2):
            x = self.layer(x, window_size=512)

        return x

if __name__ == '__main__':
    # Initialize random input tensor with shape (batch_size, channels, frames, height, width)
    input_tensor = torch.rand(32, 1, 512, 1, 183)

    # Define model parameters
    image_size = (1, 183)
    image_patch_size = (1, 183)
    max_frames = 512
    frame_patch_size = 1
    d_model = 128
    ffn_hidden = 2048
    n_head = 8
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and test the Encoder model
    model = Encoder(image_size, image_patch_size, max_frames, frame_patch_size, d_model, ffn_hidden, n_head, drop_prob, device)
    output_tensor = model(input_tensor)

    print(output_tensor.shape)