import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def idx_to_word(x, tokenizer):
    """
    Converts a sequence of token IDs into a string using the T5 tokenizer.

    Args:
    x (list of int): A list of token indices.
    tokenizer (T5Tokenizer): The T5 tokenizer.

    Returns:
    str: The decoded sentence as a string.
    """
    # Convert the list of token IDs into a string
    sentence = tokenizer.decode(x, skip_special_tokens=True)

    return sentence

if __name__ == '__main__':
    from transformers import T5Tokenizer
    import torch

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    sentence = ["So what I usually do is I don't necessarily stand in the traditional sense to start as far as a boxer's stance"]
    tokens = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Example token IDs, including special tokens
    token_ids = tokens.input_ids  # Assuming 0 is <pad>, 329 is a word, 10 is another word, 2 is <eos>
    token_ids = torch.tensor(token_ids)
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
    print(token_ids[0].shape)

    # Decode with special tokens
    sentence_with_special = tokenizer.decode(token_ids[0], skip_special_tokens=False)
    print(sentence_with_special)  # Might print: '<pad> This is a test. <eos>'

    # Decode without special tokens
    sentence_without_special = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    print(sentence_without_special)  # Might print: 'This is a test.'
