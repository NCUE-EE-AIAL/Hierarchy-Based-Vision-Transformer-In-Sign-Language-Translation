import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    Provides a dense representation of words using a weighted matrix.
    """

    def __init__(self, vocab_size, dim, n=512, device=None):
        """
        Class for token embedding that includes positional information.

        :param vocab_size: Size of the vocabulary.
        :param dim: Dimensions of the model.
        :param n: Maximum length of input sequences (default is 512).
        :param device: Device on which to create the embeddings (e.g., 'cpu' or 'cuda').
        """
        super(TokenEmbedding, self).__init__(vocab_size, dim, padding_idx=1, device=device)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n, dim).to(device))
        self.device = device  # Store the device

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Get the token embeddings
        tok_emb = super(TokenEmbedding, self).forward(x)

        # Add the positional embeddings to the token embeddings
        tok_emb = tok_emb + self.pos_embedding[:, :x.size(1), :]

        return tok_emb



