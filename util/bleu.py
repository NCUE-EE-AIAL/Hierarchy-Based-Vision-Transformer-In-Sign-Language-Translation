import math
from collections import Counter

import numpy as np


def get_bleu(bp, precisions):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 based on n-gram precisions and brevity penalty.

    Args:
        bp (float): The brevity penalty (BP) value.
        precisions (list): A list containing n-gram precisions (P₁, P₂, P₃, P₄).

    Returns:
        dict: A dictionary containing the BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.
    """
    # Calculate BLEU-1
    bleu1 = bp * precisions[0]  # BLEU-1 is just BP * P₁

    # Calculate BLEU-2 using geometric mean of P₁ and P₂
    bleu2 = bp * math.exp((1 / 2) * (math.log(precisions[0]) + math.log(precisions[1])))

    # Calculate BLEU-3 using geometric mean of P₁, P₂, and P₃
    bleu3 = bp * math.exp((1 / 3) * (math.log(precisions[0]) + math.log(precisions[1]) + math.log(precisions[2])))

    # Calculate BLEU-4 using geometric mean of P₁, P₂, P₃, and P₄
    bleu4 = bp * math.exp((1 / 4) * (
                math.log(precisions[0]) + math.log(precisions[1]) + math.log(precisions[2]) + math.log(precisions[3])))

    return bleu1, bleu2, bleu3, bleu4


def idx_to_word(x, vocab):
    batch_words = []  # To store the converted sentences for the entire batch

    # Iterate through each sentence in the batch
    for sentence in x:
        words = []
        for i in sentence:
            word = vocab.lookup_token(i)  # Retrieve the word using the index
            if '<' not in word:  # Exclude words with '<', typically special tokens like <pad>, <sos>, <eos>
                words.append(word)  # Append valid words to the list

        words = ''.join(words)  # Join the words to form a sentence
        batch_words.append(words)  # Append valid words to the list

    return batch_words


if __name__ == '__main__':
    from transformers import T5Tokenizer
    import torch

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    sentence = "This is a test."
    tokens = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Example token IDs, including special tokens
    token_ids = torch.randint(0, 2, (1, 32, 32))
    print(token_ids[0])
    token_ids = token_ids[0].max(dim=1)[1]
    print(token_ids)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(token_ids[0].shape)

    # Decode with special tokens
    sentence_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(sentence_with_special)  # Might print: '<pad> This is a test. <eos>'

    # Decode without special tokens
    sentence_without_special = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(sentence_without_special)  # Might print: 'This is a test.'
