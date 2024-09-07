import torch
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from util.tokenizer import Tokenizer
from transformers import T5Tokenizer

from conf import *
# from util.data_loader import DataLoader
from util.How2signDataset import How2signDataset
from torch.utils.data import DataLoader

sample = "This is a test."
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
tokens = t5_tokenizer.tokenize(sample)
print(['1'] + tokens + ['2'])

# tokenizer = Tokenizer()
#
# train_dataset = How2signDataset(files_dir=h2s_train_dir, tokenizer=tokenizer, seq_len=seq_len, time_len=max_frames)
# data = train_dataset.load_y().values()
# print("data len:", len(data))
#
# counter = Counter()
# for line in data:
#     line = line.lower()
#     counter.update(t5_tokenizer.tokenize(line))
# vocabulary = vocab(counter, min_freq=1, specials=['<pad>', '<sos>', '<eos>'])
# print("Vocabulary size:", len(vocabulary))
# torch.save(vocabulary, 'vocab.pth')

vocabulary = torch.load('vocab.pth')
print(vocabulary.lookup_token(0))