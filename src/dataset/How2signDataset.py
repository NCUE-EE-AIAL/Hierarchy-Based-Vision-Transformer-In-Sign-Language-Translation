import json
import os.path

import numpy as np #importing necessary libraries
import pandas as pd

class How2signDataset:
    def __init__(self, json_files, csv_file, seq_len=183, max_frame=512):
        self.seq_len = seq_len
        self.max_frame = max_frame
        self.json_files = json_files
        self.csv_file = csv_file

        self.sentence_dict = self.load_y()

    def get_x(self, x_path):
        """
        Load and preprocess the input data from a JSON file.

        Args:
            x_path (str): Path to the JSON file.

        Returns:
            numpy.ndarray: Preprocessed input data of shape (1, max_frame, seq_len).
        """
        # Load the JSON data
        with open(x_path, 'r') as f:
            data = json.load(f)

        # Extract the hand_pose_face keypoints
        hand_pose_faces = [person['hand_pose_face'] for person in data['people']]

        hand_pose_faces = np.array(hand_pose_faces)
        hand_pose_faces = hand_pose_faces.reshape(1, -1, self.seq_len)

        # Create an array to store the padded data
        x = np.zeros((1, self.max_frame, self.seq_len))

        # Fill the padded array with the actual data
        frame = len(hand_pose_faces[0])
        x[:, :frame, :] = hand_pose_faces[:, :frame, :]

        return x

    def load_y(self):
        """
        Load the target data (sentences) from the CSV file.

        Returns:
            dict: A dictionary mapping SENTENCE_NAME to SENTENCE.
        """
        data = pd.read_csv(self.csv_file, delimiter='\t', on_bad_lines='skip')
        df = data[['SENTENCE_NAME', 'SENTENCE']]
        sentence_dict = pd.Series(df.SENTENCE.values, index=df.SENTENCE_NAME).to_dict()

        return sentence_dict

    def get_y(self, x_base_path):
        y = self.sentence_dict.get(x_base_path, "0")

        return y

    def how2sign_keypoints_sentence(self):
        """
        Load the data from the JSON files and the CSV file.

        Returns:
            x (numpy.ndarray): The input data of shape (num_samples, time_len, seq_len).
            y (numpy.ndarray): The output data of shape (num_samples,).
        """
        # Load the data from multiple files
        x = [self.get_x(json_file) for json_file in self.json_files]
        x = np.concatenate(x, axis=0)

        json_files_base = [json_file.split(".")[0] for json_file in self.json_files]
        print(json_files_base)
        y = [self.get_y(json_file_base) for json_file_base in json_files_base]
        y = np.array(y)

        # Concatenate the data from the files
        return x, y