import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, dim, n=512):
        """
        Class for token embedding that includes positional information.

        :param vocab_size: Size of the vocabulary.
        :param d_model: Dimensions of the model.
        :param max_len: Maximum length of input sequences (default is 512).
        """
        super(TokenEmbedding, self).__init__(vocab_size, dim, padding_idx=1)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n, dim))

    def forward(self, x):
        # Get the token embeddings
        tok_emb = super(TokenEmbedding, self).forward(x)

        # Add the positional embeddings to the token embeddings
        tok_emb = tok_emb + self.pos_embedding[:, :x.size(1), :]

        return tok_emb


