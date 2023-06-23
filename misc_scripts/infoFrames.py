import os
import cv2
from tqdm import tqdm

min_total_frames = float('inf')
video_dir = 'Face2Face\\manipulated_sequences'

for video in tqdm(sorted(os.listdir(video_dir))):

    video_path = os.path.join(video_dir, video)
    video = cv2.VideoCapture(video_path)

    # Get the frame count and FPS of the video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Update the minimum total number of frames
    if frame_count < min_total_frames:
        min_total_frames = frame_count
        videoName = video_path

print("Minimum total number of frames:", min_total_frames)
print(F"Video: {videoName}")
