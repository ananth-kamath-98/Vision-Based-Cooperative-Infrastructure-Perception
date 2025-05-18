import os
from pathlib import Path

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

if __name__ == '__main__':
    full_length_videos = Path(os.getcwd(), "..\\dataset\\cam")

    output_directory = Path(os.getcwd(), "..\\dataset\\video_clips")

    video_count = 1
    for directory in tqdm(list(full_length_videos.iterdir())):
        for video in directory.iterdir():
            ffmpeg_extract_subclip(video, 0, 10, targetname=Path
            (output_directory, video.name))
        video_count += 1
