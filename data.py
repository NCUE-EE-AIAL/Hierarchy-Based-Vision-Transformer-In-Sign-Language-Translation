from collections import Counter
from torchtext.vocab import vocab

from conf import *
# from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from util.How2signDataset import How2signDataset
from util.vocab_libri import get_vocabulary

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import evaluate

# get tokenizer
tokenizer = Tokenizer()

# get vocabulary
train_dataset = How2signDataset(files_dir=h2s_train_dir, tokenizer=tokenizer, seq_len=seq_len, time_len=max_frames)
data = train_dataset.load_y().values()
vocabulary = get_vocabulary(tokenizer, data, vocab_file='vocab.pth')

pad_token_id = vocabulary['<pad>']  # must not be 0 or 1
sos_token_id = vocabulary['<sos>']
eos_token_id = vocabulary['<eos>']
dec_voc_size = len(vocabulary)

# initialize dataset
train_dataset = How2signDataset(files_dir=h2s_train_dir, tokenizer=tokenizer, vocabulary=vocabulary, seq_len=seq_len, time_len=max_frames)
val_dataset = How2signDataset(files_dir=h2s_val_dir, tokenizer=tokenizer, vocabulary=vocabulary, seq_len=seq_len, time_len=max_frames)
test_dataset = How2signDataset(files_dir=h2s_test_dir, tokenizer=tokenizer, vocabulary=vocabulary, seq_len=seq_len, time_len=max_frames)


def collate_fn(batch):
	# Assuming each item in batch is a tuple (input, target)
	inputs, targets = zip(*batch)  # Unpack inputs and targets

	# Process inputs and targets as tensors
	inputs = torch.stack(inputs)  # Assuming inputs are tensors
	padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_token_id)

	return inputs, padded_targets

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
valid_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)



sacrebleu = evaluate.load("sacrebleu")