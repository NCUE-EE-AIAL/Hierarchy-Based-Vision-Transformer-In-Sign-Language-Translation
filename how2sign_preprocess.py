import os
import orjson
from multiprocessing import Pool
from glob import glob
import numpy as np

# Define indices for the required keypoints
# all the hand keypoints
hand_indices = list(range(21))

# neck, shoulders, elbows: 1, 2, 3, 4, 5, 6, 7, 8
pose_indices = [1, 2, 3, 4, 5, 6, 7, 8]

# face shape: 0, 2, 4, 6, 8, 10, 12,14, 16, 17, 19, 21, 22, 24, 26
# nose: 27, 30, 31, 33, 35
# eyes: 36, 38, 39, 41, 68, 42, 43, 45, 46, 69
# mouth: 48, 50, 52, 54, 56, 58
face_indices = [
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    17,
    19,
    21,
    22,
    24,
    26,
    27,
    31,
    33,
    35,
    36,
    38,
    39,
    41,
    68,
    42,
    43,
    45,
    46,
    69,
    48,
    50,
    52,
    54,
    56,
    58,
]


def find_files(directory, pattern="**/*.json", interval=1):
    """Recursively finds all files matching the pattern and returns every other file."""
    files = glob(os.path.join(directory, pattern), recursive=True)
    return files[::interval]


def reduce_keypoints(person):
    hand_left_keypoints_2d = [
        person["hand_left_keypoints_2d"][i * 3 : (i * 3) + 3] for i in hand_indices
    ]
    hand_right_keypoints_2d = [
        person["hand_right_keypoints_2d"][i * 3 : (i * 3) + 3] for i in hand_indices
    ]
    pose_keypoints_2d = [
        person["pose_keypoints_2d"][i * 3 : (i * 3) + 3] for i in pose_indices
    ]
    face_keypoints_2d = [
        person["face_keypoints_2d"][i * 3 : (i * 3) + 3] for i in face_indices
    ]

    hand_pose_face = (
        hand_left_keypoints_2d
        + hand_right_keypoints_2d
        + pose_keypoints_2d
        + face_keypoints_2d
    )

    # normalize the hand_pose_face keypoints
    hand_pose_face = np.array(hand_pose_face)
    hand_pose_face = hand_pose_face.reshape(-1, 3)
    divisors = np.array([1280, 800, 1])  # max of width, height, and visibility
    hand_pose_face = hand_pose_face / divisors
    hand_pose_face = hand_pose_face.flatten().tolist()

    return {"hand_pose_face": hand_pose_face}


# Function to read and process JSON files in a directory
def combine_frame(input_dir, seq_len):
    hand_pose_faces = []
    files_path = find_files(
        input_dir, pattern="**/*.json", interval=3
    )  # Adjust interval as needed

    for file_path in files_path:
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())
            if data["people"]:
                person = data["people"][0]  # Use the first detected person
                reduced_person = reduce_keypoints(person)
                hand_pose_face = reduced_person["hand_pose_face"]
                hand_pose_faces.append(hand_pose_face)
            else:
                # No person detected; append zero vector
                hand_pose_faces.append(np.zeros(seq_len))

    # Convert list to NumPy array (num_frames, seq_len)
    data_array = np.array(hand_pose_faces)
    return data_array


def extract_npy(output_dir, input_dir, seq_len):
    filename = os.path.basename(input_dir)
    output_file = os.path.join(output_dir, f"{filename}.npy")

    # Get data array
    data_array = combine_frame(input_dir, seq_len)

    # Save as .npy file
    np.save(output_file, data_array)

    print(f"Numpy array saved to {output_file}")


def prep(main_dir, output_dir, seq_len, num_workers=4):
    subfolders_path = [
        os.path.join(main_dir, subfolder) for subfolder in os.listdir(main_dir)
    ]
    with Pool(num_workers) as pool:
        pool.starmap(
            extract_npy,
            [
                (output_dir, subfolder_path, seq_len)
                for subfolder_path in subfolders_path
            ],
        )


if __name__ == "__main__":
    # keypoints preprocessing
    keypoints_directory = input(
        "Enter the path to the main folder containing subfolders with JSON files: "
    )
    output_directory = input("Enter the path where the files will be saved: ")
    prep(keypoints_directory, output_directory, 255)
