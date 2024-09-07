import os
import torch
from collections import Counter
from torchtext.vocab import vocab

def get_vocabulary(tokenizer, data, vocab_file='vocab.pth'):
	if not os.path.exists(vocab_file):
		counter = Counter()
		for line in data:
			counter.update(tokenizer.tokenize_en(line))

		# Build the vocabulary
		vocabulary = vocab(counter, min_freq=1, specials=['<pad>', '<sos>', '<eos>'])
		torch.save(vocabulary, 'vocab.pth')
		print("Vocabulary size:", len(vocabulary))
	else:
		vocabulary = torch.load('vocab.pth')
		print("Vocabulary size:", len(vocabulary))

	return vocabulary