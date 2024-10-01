
def window_partition(x, window_size):
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

    return windows


def window_reverse(windows, window_size, N):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)  # Tensor of windows flattened across the batch
        window_size (int): Window size  # Size of the windows
        N (int): length of the original image

    Returns:
        x: (B, N, C)  # Reconstructed tensor with shape (Batch size, Height, Width, Channels)
    """
    B = int(windows.shape[0] / (N / window_size))
    x = windows.view(B, N // window_size,window_size, -1)
    x = x.contiguous().view(B, N, -1)
    return x