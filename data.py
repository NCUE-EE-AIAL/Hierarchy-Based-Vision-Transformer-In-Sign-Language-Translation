from conf import *
# from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from src.dataset import How2signDataset
from torch.utils.data import Dataset, DataLoader

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train_dataset = How2signDataset(root_dir=h2s_train_dir, json_file=h2s_train_csv)
val_dataset = How2signDataset(root_dir=h2s_val_dir, json_file=h2s_train_csv)
test_dataset = How2signDataset(root_dir=h2s_test_dir, json_file=h2s_train_csv)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
