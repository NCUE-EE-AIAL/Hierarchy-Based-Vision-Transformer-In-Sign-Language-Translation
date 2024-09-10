import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device :', device)

# model parameter setting
batch_size = 32
seq_len = 183
image_size = (1, seq_len)
image_patch_size = (1, seq_len)
max_frames = 512
frame_patch_size = 4
dim = 512
n_layers = 6
n_heads = 4
ffn_hidden = 1024
drop_prob = 0.2
max_len = 256

# optimizer parameter setting
init_lr = 0.001
factor = 0.9
patience = 10
warmup = 10
epoch = 50
clip = 5.0
weight_decay = 1e-2
inf = float('inf')
# input shape -> (batch_size, channels, frames, height, width)

# file path setting
h2s_train_dir = 'dataset/how2sign/train_2D_Keypoints'
h2s_val_dir = 'dataset/how2sign/val_2D_Keypoints'
h2s_test_dir = 'dataset/how2sign/test_2D_Keypoints'

# h2s_train_dir = 'dataset/how2sign/for_test'
# h2s_test_dir = 'dataset/how2sign/for_test'
# h2s_val_dir = 'dataset/how2sign/for_test'