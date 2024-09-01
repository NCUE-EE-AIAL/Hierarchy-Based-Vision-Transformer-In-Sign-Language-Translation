import torch

output = torch.arange(1, 51).view(1, 5, 10)
print("Original Tensor (output):", output[0])

output_words = output[0].max(dim=1)[1]
print("output_words : ", output_words)
