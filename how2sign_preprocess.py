import os
import json
from multiprocessing import Pool
from glob import glob
import numpy as np
import cv2

# Define indices for the required keypoints
# all the hand keypoints
hand_indices = list(range(21))

# neck, shoulders, elbows: 1, 2, 3, 4, 5, 6, 7, 8
pose_indices = [1, 2, 3, 4, 5, 6, 7, 8]

# face shape: 0, 2, 4, 6, 8, 10, 12,14, 16, 17, 19, 21, 22, 24, 26
# nose: 27, 30, 31, 33, 35
# eyes: 36, 38, 39, 41, 68, 42, 43, 45, 46, 69
# mouth: 48, 50, 52, 54, 56, 58
face_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26, 27, 31, 33, 35, 36, 38, 39, 41, 68, 42, 43, 45, 46, 69, 48, 50, 52, 54, 56, 58]


def find_files(directory, pattern='**/*.json', interval=1):
    """Recursively finds all files matching the pattern and returns every other file."""
    files = glob(os.path.join(directory, pattern), recursive=True)
    return files[::interval]


def reduce_keypoints(person):
    reduced_person = {
        "person_id": person.get("person_id", [-1]),
        "hand_pose_face": []
    }
    hand_left_keypoints_2d = [person["hand_left_keypoints_2d"][i * 3:(i * 3) + 3] for i in hand_indices]
    hand_right_keypoints_2d = [person["hand_right_keypoints_2d"][i * 3:(i * 3) + 3] for i in hand_indices]
    pose_keypoints_2d = [person['pose_keypoints_2d'][i * 3:(i * 3) + 3] for i in pose_indices]
    face_keypoints_2d = [person['face_keypoints_2d'][i * 3:(i * 3) + 3] for i in face_indices]

    hand_pose_face = hand_left_keypoints_2d + hand_right_keypoints_2d + pose_keypoints_2d + face_keypoints_2d

    # normalize the hand_pose_face keypoints
    hand_pose_face = np.array(hand_pose_face)
    hand_pose_face = hand_pose_face.reshape(-1, 3)
    divisors = np.array([1280, 800, 1]) # max of width, height, and visibility
    hand_pose_face = hand_pose_face / divisors
    hand_pose_face = hand_pose_face.flatten().tolist()

    reduced_person["hand_pose_face"].extend(hand_pose_face)

    return reduced_person



# Function to read and process JSON files in a directory
def combine_frame(input_dir):
    all_people = []
    files_path = find_files(input_dir, pattern='**/*.json', interval=3) # 30 -> 10 fps
    for file_path in files_path:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for person in data["people"]:
                reduced_person = reduce_keypoints(person)
                all_people.append(reduced_person)

    return all_people


def extract_json(output_dir, input_dir):
    filename = os.path.basename(input_dir)
    output_file = os.path.join(output_dir, f'{filename}.json')

    # Process the files in the subfolder and merge the results
    merged_data = {"version": 1.3, "people": combine_frame(input_dir)}

    # Save the merged data to a JSON file in the output directory
    with open(output_file, 'w') as out_file:
        json.dump(merged_data, out_file, indent=4)

    print(f'Merged and reduced JSON saved to {output_file}')


def prep(main_dir, output_dir, num_workers=4):
    # Loop through each subfolder in the main directory
    subfolders_path = [os.path.join(main_dir, subfolder) for subfolder in os.listdir(main_dir)]
    with Pool(num_workers) as pool:
        pool.starmap(extract_json, [(output_dir, subfolder_path) for subfolder_path in subfolders_path])


# Video preprocessing function
def extract_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            cv2.imwrite(f"{output_folder}/frame{count}.jpg", image)
        success, image = vidcap.read()
        count += 1
    print(f"Extracted frames from {video_path}")

def video_prep(vid_dir, output_dir, num_workers=4):
    files = find_files(vid_dir, pattern='**/*.*')
    output_dirs = [os.path.join(output_dir, os.path.basename(file).split('.')[0]) for file in files]

    length = len(files)
    with Pool(num_workers) as pool:
        # 30 -> 10 fps
        pool.starmap(extract_frames, [(files[i], output_dirs[i], 3) for i in range(length)])


if __name__ == '__main__':
    # video preprocessing
    # video_directory = input("Enter the path to the main folder containing subfolders with video files: ")
    # output_directory = input("Enter the path where the files will be saved: ")
    # video_prep(video_directory, output_directory)

    # keypoints preprocessing
    keypoints_directory = input("Enter the path to the main folder containing subfolders with JSON files: ")
    output_directory = input("Enter the path where the files will be saved: ")
    prep(keypoints_directory, output_directory)