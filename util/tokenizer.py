from transformers import T5Tokenizer


class Tokenizer:
    def __init__(self):
        # Initialize the T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

    def tokenize_en(self, sentences, padding="max_length", max_length=256, truncation=True, return_tensors="pt"):
        """
        Tokenizes English text using the T5 tokenizer.

        Args:
            sentences (list of str): List of sentences to tokenize.
            padding (str): Padding strategy ("longest" or "max_length").
            max_length (int): Maximum length for padding/truncation.
            truncation (bool): Whether to truncate sentences to max_length.
            return_tensors (str): Return type for tokenized output (e.g., "pt" for PyTorch tensors).

        Returns:
            Tokenized output with padding, truncation, and specified return type.
        """
        # Tokenize the sentences using the T5 tokenizer
        tokens = self.tokenizer(
            sentences,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors
        )

        return tokens.input_ids, tokens.attention_mask



