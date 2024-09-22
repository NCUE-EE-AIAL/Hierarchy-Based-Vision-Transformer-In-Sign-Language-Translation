import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device :', device)

# model parameter setting
batch_size = 32
seq_len = 255
image_size = (1, seq_len)
image_patch_size = (1, seq_len)
max_frames = 512
frame_patch_size = 1
dim = (64, 128, 256) # must be double of the previous layer
enc_layers=(2, 2, 4)
dec_layers = 4
n_heads = (4, 4, 4) # not must be double
ffn_hidden = 1024
drop_prob = 0.1
max_len = 256

# loss
label_smoothing = 0.2

# lr_scheduler
t_0 = 18
end_lr = 1e-7

# optimizer parameter setting
init_lr = 1e-3
warmup = 4
epoch = 110
clip = 1.0
weight_decay = 1e-4
inf = float('inf')
# input shape -> (batch_size, channels, frames, height, width)

# file path setting
h2s_train_dir = 'dataset/how2sign/train_2D_Keypoints_85'
h2s_val_dir = 'dataset/how2sign/val_2D_Keypoints_85'
h2s_test_dir = 'dataset/how2sign/test_2D_Keypoints_85'

# h2s_train_dir = 'dataset/how2sign/for_test'
# h2s_test_dir = 'dataset/how2sign/for_test'
# h2s_val_dir = 'dataset/how2sign/for_test'

# Prepare the information as a formatted string
info = f"""
# Model Parameter Settings
batch_size = {batch_size}
seq_len = {seq_len}
image_size = {image_size}
image_patch_size = {image_patch_size}
max_frames = {max_frames}
frame_patch_size = {frame_patch_size}
dim = {dim}
enc_layers = {enc_layers}
dec_layers = {dec_layers}
n_heads = {n_heads}
ffn_hidden = {ffn_hidden}
drop_prob = {drop_prob}
max_len = {max_len}

# Optimizer Parameter Settings
init_lr = {init_lr}
warmup = {warmup}
epoch = {epoch}
clip = {clip}
weight_decay = {weight_decay}
inf = {inf}
"""

f = open('result/parameters.txt', 'w')
f.write(str(info))
f.close()
