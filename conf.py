"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Initialize random input tensor with shape (batch_size, channels, frames, height, width)
input_tensor = torch.rand(3, 1, 512, 1, 183)

# model parameter setting
batch_size = 128
image_size = (1, 183)
image_patch_size = (1, 183)
max_frames = 512
frame_patch_size = 1
dim = 128
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
max_len = 256
dec_voc_size = 32128

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

# input shape -> (batch_size, channels, frames, height, width)

# file path setting
h2s_train_dir = 'dataset/how2sign/train_2D_Keypoints'
h2s_val_dir = 'dataset/how2sign/val_2D_Keypoints'
h2s_test_dir = 'dataset/how2sign/test_2D_Keypoints'
