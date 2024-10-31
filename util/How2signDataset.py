import os.path
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class How2signDataset(Dataset):
    def __init__(self, files_dir, tokenizer, seq_len, time_len, vocabulary=None):
        self.seq_len = seq_len
        self.time_len = time_len
        self.npy_files = self.find_files(files_dir, pattern="**/*.npy")
        self.csv_file = self.find_files(files_dir, pattern="**/*.csv")[0]
        self.tokenizer = tokenizer
        self.vocab = vocabulary

        self.sentence_dict = self.load_y()

        # Preprocess labels and cache them
        if vocabulary is not None:
            self.labels_cache = {}
            for npy_file in self.npy_files:
                base_name = os.path.splitext(os.path.basename(npy_file))[0]
                y = self.process_label(base_name)
                self.labels_cache[base_name] = y

    def process_label(self, base_name):
        sentence = self.sentence_dict.get(base_name, "<pad>")
        tokens = self.tokenizer.tokenize_en(sentence)
        y = [self.vocab.get(token, self.vocab["<pad>"]) for token in tokens]
        return torch.tensor(y, dtype=torch.long)

    def find_files(self, directory, pattern):
        """Recursively finds all files matching the pattern and returns every other file."""
        return glob(os.path.join(directory, pattern), recursive=True)

    def process_label(self, base_name):
        sentence = self.sentence_dict.get(base_name, "<unk>")
        sentence = self.tokenizer.tokenize_en(sentence)
        y = [
            self.vocab[token] if token in self.vocab else self.vocab["<unk>"]
            for token in sentence
        ]

        return torch.tensor(y, dtype=torch.long)

    def get_x(self, x_path):
        # Load the numpy data
        data_array = np.load(x_path, mmap_mode="r")  # Shape: (num_frames, seq_len)

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

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_file = self.npy_files[idx]
        x = self.get_x(npy_file)

        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        y = self.labels_cache[base_name]

        return x, y
