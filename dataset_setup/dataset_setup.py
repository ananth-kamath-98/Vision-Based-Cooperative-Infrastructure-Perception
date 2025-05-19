from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    full_length_videos = project_root / "dataset" / "cam"
    output_directory = project_root / "dataset" / "video_clips"
    output_directory.mkdir(parents=True, exist_ok=True)

    for directory in tqdm(full_length_videos.iterdir()):
        for video_file in directory.iterdir():
            output_path = output_directory / video_file.name
            ffmpeg_extract_subclip(
                str(video_file),
                0,
                10,
                str(output_path)
            )
