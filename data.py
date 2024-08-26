from conf import *
# from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from util.How2signDataset import How2signDataset
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

mp.set_start_method('spawn')
tokenizer = Tokenizer()

train_dataset = How2signDataset(files_dir=h2s_train_dir, tokenizer=tokenizer)
val_dataset = How2signDataset(files_dir=h2s_val_dir, tokenizer=tokenizer, batch_size=batch_size)
test_dataset = How2signDataset(files_dir=h2s_test_dir, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

pad_token_id = 0
eos_token_id = 1
