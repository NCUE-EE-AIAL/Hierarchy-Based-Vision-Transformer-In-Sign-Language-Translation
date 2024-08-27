from conf import *
# from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from util.How2signDataset import How2signDataset
from torch.utils.data import DataLoader

tokenizer = Tokenizer()
generator = torch.Generator(device='cpu')

train_dataset = How2signDataset(files_dir=h2s_train_dir, tokenizer=tokenizer, batch_size=batch_size, seq_len=seq_len, time_len=max_frames)
val_dataset = How2signDataset(files_dir=h2s_val_dir, tokenizer=tokenizer, batch_size=batch_size, seq_len=seq_len, time_len=max_frames)
test_dataset = How2signDataset(files_dir=h2s_test_dir, tokenizer=tokenizer, batch_size=batch_size, seq_len=seq_len, time_len=max_frames)

# train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
train_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
valid_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
# test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

pad_token_id = 0
eos_token_id = 1

if __name__ == '__main__':
	# mp.set_start_method('spawn')
	for i, batch in enumerate(train_iter):
		print(batch)
		pass

