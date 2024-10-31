import os.path
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class YoutubeASLDataset(Dataset):
    def __init__(self, files_dir, tokenizer, seq_len, time_len, vocabulary=None):
        self.seq_len = seq_len
        self.time_len = time_len
        self.npy_files = self.find_files(files_dir, pattern="**/*.npy")
        self.csv_file = self.find_files(files_dir, pattern="**/*.csv")[
            0
        ]  # should be only one csv file
        self.tokenizer = tokenizer
        self.vocab = vocabulary

        self.sentence_dict = self.load_y()

    def find_files(self, directory, pattern):
        """Recursively finds all files matching the pattern and returns every other file."""
        return glob(os.path.join(directory, pattern), recursive=True)

    def get_x(self, x_path):
        # Load the numpy data
        data_array = np.load(x_path)  # Shape: (num_frames, seq_len)

        # Handle time_len by slicing or padding
        num_frames = data_array.shape[0]
        if num_frames > self.time_len:
            # Truncate to time_len frames
            data_array = data_array[: self.time_len, :]
        elif num_frames < self.time_len:
            # Pad with zeros
            padding = np.zeros((self.time_len - num_frames, self.seq_len))
            data_array = np.vstack((data_array, padding))

        # Reshape to (1, time_len, 1, seq_len)
        x = data_array.reshape(1, self.time_len, 1, self.seq_len)
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def load_y(self):
        data = pd.read_csv(self.csv_file, delimiter="\t", on_bad_lines="skip")
        df = data[["SENTENCE_NAME", "SENTENCE"]]
        sentence_dict = pd.Series(df.SENTENCE.values, index=df.SENTENCE_NAME).to_dict()

        return sentence_dict

    def get_y(self, x_base_path):
        sentence = self.sentence_dict.get(x_base_path, "<pad>")
        sentence = self.tokenizer.tokenize_en(sentence)
        y = [
            self.vocab[token] if token in self.vocab else self.vocab["<pad>"]
            for token in sentence
        ]
        y = torch.tensor(y, dtype=torch.long)

        return y

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_files = self.npy_files[idx]
        x = self.get_x(npy_files)

        json_base = os.path.splitext(os.path.basename(npy_files))[0]
        y = self.get_y(json_base)

        return x, y
