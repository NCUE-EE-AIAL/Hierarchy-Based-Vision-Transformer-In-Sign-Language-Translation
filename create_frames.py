import cv2
import os
from multiprocessing import Pool


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

video_directory = r"D:\user\Downloads\how2sign\test_rgb_front_clips\raw_videos"
output_directory = r"D:\user\Downloads\how2sign\test_rgb_front_clips\test_rgb_front_clips_output"

files = []
output_dirs = []
for video_file in os.listdir(video_directory):
    video_path = os.path.join(video_directory, video_file)
    video_output_folder = os.path.join(output_directory, video_file.split('.')[0])
    files.append(video_path)
    output_dirs.append(video_output_folder)
    # extract_frames(video_path, video_output_folder)

if __name__ == "__main__":
    num_workers = 4
    length = len(files)
    with Pool(num_workers) as pool:
        # 30 -> 10 fps
        pool.starmap(extract_frames, [(files[i], output_dirs[i], 3) for i in range(length)])
