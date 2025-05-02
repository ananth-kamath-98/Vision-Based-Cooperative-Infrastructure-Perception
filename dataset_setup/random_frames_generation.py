import cv2
import random
import os
import argparse



def extract_random_frames(video_path, output_dir, n):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video loaded: {video_path}")
    print(f"Total Frames: {frame_count}, Frame Rate: {frame_rate} FPS")

    # Generate n random frame indices
    random_frames = sorted(random.sample(range(frame_count), n))

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames
    frame_idx = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in random_frames:
            output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
            extracted_count += 1

            # Stop if we've extracted all the frames we need
            if extracted_count >= n:
                break

        frame_idx += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract random frames from a video.")
    # parser.add_argument("video_path", type=str, help="Path to the input video.")
    # parser.add_argument("output_dir", type=str, help="Directory to save extracted frames.")
    # parser.add_argument("n", type=int, help="Number of random frames to extract.")
    #
    # args = parser.parse_args()
    video_path = "dataset/cam/6/camera_6.mp4"
    output_path = "dataset/frames/random/6"
    n = 1000

    extract_random_frames(video_path, output_path, n)
