from collections import Counter
from torchtext.vocab import vocab

from conf import *

# from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from util.How2signDataset import How2signDataset
from util.YoutubeASLDataset import YoutubeASLDataset
from util.vocab_libri import get_vocabulary

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import evaluate

# get tokenizer
tokenizer = Tokenizer()

# get vocabulary
train_dataset = How2signDataset(
    files_dir=h2s_train_dir,
    tokenizer=tokenizer,
    seq_len=seq_len,
    time_len=max_frames,
    max_output=max_output,
)
data = train_dataset.load_y().values()
vocabulary = get_vocabulary(tokenizer, data, vocab_file="vocab.pth")

# initialize dataset
# How2Sign dataset
train_dataset = How2signDataset(
    files_dir=h2s_train_dir,
    tokenizer=tokenizer,
    vocabulary=vocabulary,
    seq_len=seq_len,
    time_len=max_frames,
    max_output=max_output,
)
val_dataset = How2signDataset(
    files_dir=h2s_val_dir,
    tokenizer=tokenizer,
    vocabulary=vocabulary,
    seq_len=seq_len,
    time_len=max_frames,
    max_output=max_output,
)
test_dataset = How2signDataset(
    files_dir=h2s_test_dir,
    tokenizer=tokenizer,
    vocabulary=vocabulary,
    seq_len=seq_len,
    time_len=max_frames,
    max_output=max_output,
)

# YoutubeASL dataset
# train_dataset = YoutubeASLDataset(files_dir=yt_asl_dir, tokenizer=tokenizer, vocabulary=vocabulary, seq_len=seq_len, time_len=max_frames)

pad_token_id = vocabulary["<pad>"]
dec_voc_size = len(vocabulary)

train_iter = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
valid_iter = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
test_iter = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

sacrebleu = evaluate.load("sacrebleu")

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    for i, batch in enumerate(train_iter):
        print(batch)
        pass
