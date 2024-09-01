
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

if __name__ == '__main__':
    import torch

    # Test Case
    B, N, C = 32, 512, 512  # Batch size, sequence length, number of channels
    window_size = 64

    # Create a test tensor
    x = torch.arange(B * N * C).view(B, N, C)
    print("Original Tensor (x):")
    print(x)

    # Partition the tensor into windows
    windows = window_partition(x, window_size)
    print("\nWindows after partitioning:")
    print(windows)

    # Reverse the windows back to the original tensor
    x_reconstructed = window_reverse(windows, window_size, N)
    print("\nReconstructed Tensor:")
    print(x_reconstructed)

    # Verify the reconstruction
    print("\nReconstruction successful:", torch.equal(x, x_reconstructed))