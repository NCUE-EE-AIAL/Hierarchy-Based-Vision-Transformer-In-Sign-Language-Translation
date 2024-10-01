def mask_partition(mask, window_size):
	batch_size, _, _, seq_len = mask.shape
	assert seq_len % window_size == 0, "Sequence length must be divisible by the window size"

	num_windows = seq_len // window_size
	windows_mask = mask.view(batch_size, 1, 1, num_windows, window_size)
	windows_mask = windows_mask.permute(0, 3, 1, 2, 4).contiguous()  # (batch_size, num_windows, 1, 1, window_size)
	windows_mask = windows_mask.view(batch_size * num_windows, 1, 1, window_size)  # (batch_size * num_windows, 1, 1, window_size)

	return windows_mask


def patch_merge_mask(src_mask):
	batch_size, _, _, seq_len = src_mask.shape

	assert seq_len % 2 == 0, "Sequence length must be divisible by 2 for patch merging"

	# Reshape the mask to group every two patches together
	src_mask = src_mask.view(batch_size, 1, 1, seq_len // 2, 2)  # (batch_size, 1, 1, 256, 2)

	merged_mask = src_mask.max(dim=-1)[0]

	return merged_mask