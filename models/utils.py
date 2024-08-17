# !!! not yet finish !!!
def window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, C)
    return windows