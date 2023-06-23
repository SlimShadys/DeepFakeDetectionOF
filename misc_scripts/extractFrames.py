import cv2
import os
from tqdm import tqdm
import re
import numpy as np

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

video_dir = 'original'

extracted_dir = os.path.join('extracted', video_dir)
if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

sortedFiles = sorted_alphanumeric(os.listdir(os.path.join(video_dir)))

# Specify the desired frame numbers to extract
desired_frames = np.append(np.arange(0, 286, 13), 285)

with tqdm(total=len(sortedFiles)*23, desc=F"Extracting {video_dir} sequences") as pbar:
    for video_name in sortedFiles:

        frame_count = 0

        # Set the video file path
        video_path = os.path.join(video_dir, video_name)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop over the frames in the video
        while cap.isOpened() and frame_count < max(desired_frames) + 1:

            # Read the next frame
            ret, frame = cap.read()

            # Check if there are no more frames
            if not ret:
                break

            # Check if this is a frame we want to extract
            if frame_count in desired_frames:
                # Save the frame as an image
                cv2.imwrite(
                    os.path.join(
                        extracted_dir, F"{video_name.split('.')[0]}-frame{frame_count}.jpg"
                    ),
                    frame,
                )

            # Increment the frame count
            frame_count += 1
            pbar.update(1)

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
