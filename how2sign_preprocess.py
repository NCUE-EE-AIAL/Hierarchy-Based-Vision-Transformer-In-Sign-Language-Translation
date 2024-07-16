import os
import json
from multiprocessing import Pool
from glob import glob

# Define indices for the required keypoints
hand_indices = list(range(21))
pose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]  # nose, neck, shoulders, elbows, wrists
face_indices = [48, 49, 50, 51, 52, 38, 41, 44, 47]
# mouth: 48, 49, 50, 51, 52
# left eye: 38, 41,
# right eye: 44, 47


def find_files(directory, pattern='**/*.json'):
	"""Recursively finds all files matching the pattern."""
	return glob(os.path.join(directory, pattern), recursive=True)


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

	reduced_person["hand_pose_face"].extend(hand_pose_face)

	return reduced_person



# Function to read and process JSON files in a directory
def combine_frame(input_dir):
	all_people = []
	files_path = find_files(input_dir, pattern='**/*.json')
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


if __name__ == '__main__':
	# Prompt the user to enter the path of the folder containing subfolders of JSON files
	main_directory = input("Enter the path to the main folder containing subfolders with JSON files: ")

	# Prompt the user to enter the path where the reduced and merged JSON files will be saved
	output_directory = input("Enter the path where the reduced and merged JSON files will be saved: ")

	prep(main_directory, output_directory)