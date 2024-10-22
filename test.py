import torch

# Test parameters
batch_size = 1
seq_len = 12  # Sequence length must be divisible by window_size
window_size = 4

# Create a dummy mask tensor of shape (batch_size, 1, 1, seq_len)
mask = torch.randn(batch_size, 1, 1, seq_len)

# Run the transformation logic
def test_mask_function(mask, window_size):
    batch_size, _, _, seq_len = mask.shape
    assert seq_len % window_size == 0, "Sequence length must be divisible by the window size"

    mask = mask.view(batch_size, 1, 1, seq_len // window_size, window_size)
    mask = mask.permute(0, 3, 1, 2, 4).contiguous()  # (batch_size, num_windows, 1, 1, window_size)
    mask = mask.view(-1, 1, 1, window_size)  # (batch_size * num_windows, 1, 1, window_size)

    return mask

def reverse_mask_function(mask, window_size, seq_len):
    batch_size = mask.size(0) // (seq_len // window_size)  # Compute batch size
    num_windows = seq_len // window_size  # Number of windows

    mask = mask.view(batch_size, num_windows, 1, 1, window_size)
    mask = mask.permute(0, 2, 3, 1, 4).contiguous()
    mask = mask.view(batch_size, 1, 1, seq_len)

    return mask

# Test the function
# result_mask = test_mask_function(mask, window_size)
# reverse_mask = reverse_mask_function(result_mask, window_size, seq_len)
#
# print(result_mask.shape)
# print(mask)
# print(result_mask)
#
# print(reverse_mask.shape)
# print(reverse_mask)

import torch


def shift_windows(input_tensor, shift_size, mask=None):
    """
    Shift windows in the input tensor by shift_size.

    Parameters:
    - input_tensor: (batch_size, seq_len, dimension)
    - shift_size: size of the shift, typically 64.

    Returns:
    - shifted_tensor: Tensor with windows shifted by shift_size.
    """
    batch_size, seq_len, dimension = input_tensor.shape
    assert shift_size < seq_len, "Shift size must be smaller than sequence length"

    first_part = input_tensor[:, shift_size:, :]  # (b, seq_len-shift_size, dimension)
    second_part = input_tensor[:, :shift_size, :]  # (b, shift_size, dimension)
    shifted_tensor = torch.cat([first_part, second_part], dim=1)  # (b, seq_len, dimension)

    if mask is not None:
        first_part = mask[:, :, :, shift_size:]  # (b, 1, 1, seq_len-shift_size)
        second_part = torch.ones_like(mask[:, :, :, :shift_size])  # (b, 1, 1, shift_size)
        mask = torch.cat([first_part, second_part], dim=-1)

    return shifted_tensor, mask


def reverse_shift_windows(shifted_tensor, shift_size, mask=None):
    """
    Reverse the window shift applied by the shift_windows function.

    Parameters:
    - shifted_tensor: (batch_size, seq_len, dimension) Tensor with shifted windows.
    - shift_size: size of the shift, typically 64.

    Returns:
    - reversed_tensor: Tensor with the original window order restored.
    """
    batch_size, seq_len, dimension = shifted_tensor.shape
    assert shift_size < seq_len, "Shift size must be smaller than sequence length"

    first_part = shifted_tensor[:, :-shift_size, :]  # (b, seq_len-shift_size, dimension)
    second_part = shifted_tensor[:, -shift_size:, :]  # (b, shift_size, dimension)
    reversed_tensor = torch.cat([second_part, first_part], dim=1)  # (b, seq_len, dimension)

    if mask is not None:
        first_part = mask[:, :, :, :-shift_size]  # (b, 1, 1, seq_len-shift_size)
        second_part = mask[:, :, :, -shift_size:]  # (b, 1, 1, shift_size)
        mask = torch.cat([second_part, first_part], dim=-1)

    return reversed_tensor, mask

# Example input
input_tensor = torch.randn(1, 16, 1)  # (batch_size=4, seq_len=512, dimension=128)
input_mask = torch.randn(1, 1, 1, 16)

# Apply window shift with shift_size = 64
shifted_tensor, shifted_mask = shift_windows(input_tensor, 4, input_mask)
original_tensor, original_mask = reverse_shift_windows(shifted_tensor, 4, shifted_mask)

print("Original Shape:", input_mask)
print("Shifted Shape:", shifted_mask)
# print("Original Shape:", original_mask)
