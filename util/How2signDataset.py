import json
import os.path
from glob import glob
import numpy as np #importing necessary libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class How2signDataset(Dataset):
    def __init__(self, files_dir, tokenizer, batch_size, seq_len=183, time_len=512):
        self.seq_len = seq_len
        self.time_len = time_len
        self.json_files = self.find_files(files_dir, pattern='**/*.json')
        self.csv_file = self.find_files(files_dir, pattern='**/*.csv')[0] # should be only one csv file
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.sentence_dict = self.load_y()

    def find_files(self, directory, pattern='**/*.json'):
        """Recursively finds all files matching the pattern and returns every other file."""
        return glob(os.path.join(directory, pattern), recursive=True)

    def get_x(self, x_path):
        # Load the JSON data
        with open(x_path, 'r') as f:
            data = json.load(f)

        # Extract the hand_pose_face keypoints
        hand_pose_faces = [person['hand_pose_face'] for person in data['people']]

        hand_pose_faces = np.array(hand_pose_faces)
        hand_pose_faces = hand_pose_faces.reshape(1, 1, -1, 1, self.seq_len)

        # Create an array to store the padded data
        x = np.zeros((1, 1, self.time_len, 1, self.seq_len))

        # Fill the padded array with the actual data
        time_length = len(hand_pose_faces[0])
        x[:, :, time_length, :, :] = hand_pose_faces[:, :, :time_length, :, :]

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)

        return x

    def load_y(self):
        data = pd.read_csv(self.csv_file, delimiter='\t', on_bad_lines='skip')
        df = data[['SENTENCE_NAME', 'SENTENCE']]
        sentence_dict = pd.Series(df.SENTENCE.values, index=df.SENTENCE_NAME).to_dict()

        return sentence_dict

    def get_y(self, x_base_path):
        sentence = self.sentence_dict.get(x_base_path, "0")
        y, _ = self.tokenizer.tokenize_en(sentence)

        return y

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        x = self.get_x(json_file)
        y = self.get_y(json_file.split(".")[0])

        return x, y

