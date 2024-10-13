import torch

def windows_partition(x, window_size, mask=None):
    """
    Args:
        x: (B, N, C)
        window_size (int): window frame size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)

    if mask is not None:
        batch_size, _, _, seq_len = mask.shape
        assert seq_len % window_size == 0, "Sequence length must be divisible by the window size"

        mask = mask.view(batch_size, 1, 1, seq_len // window_size, window_size)
        mask = mask.permute(0, 3, 1, 2, 4).contiguous()  # (batch_size, num_windows, 1, 1, window_size)
        mask = mask.view(-1, 1, 1, window_size)  # (batch_size * num_windows, 1, 1, window_size)

    return windows, mask


def windows_partition_reverse(windows, window_size, N, mask=None):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)  # Tensor of windows flattened across the batch
        window_size (int): Window size  # Size of the windows
        N (int): length of the original sequence

    Returns:
        x: (B, N, C)  # Reconstructed tensor with shape (Batch size, N, Channels)
    """
    B = int(windows.shape[0] / (N / window_size))
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.contiguous().view(B, N, -1)

    if mask is not None:
        batch_size = int(mask.shape[0] / (N / window_size))

        mask = mask.view(batch_size, N // window_size, 1, 1, window_size)
        mask = mask.permute(0, 2, 3, 1, 4).contiguous()
        mask = mask.view(batch_size, 1, 1, N)

    return x, mask


def windows_shift(input_tensor, shift_size, mask=None):
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


def windows_shift_reverse(shifted_tensor, shift_size, mask=None):
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
