"""
How to Run

Run the script by providing the path to your video folder:
python create_video_from_images.py "output/history_epoch/video18"

The final video will be saved as final_video.mp4 inside the video folder.
"""

import os
import json
import subprocess
import argparse
import sys
import shutil
import re
import math

def get_video_duration(folder):
    """
    Attempts to determine the video end time from subtitles.srt.
    Returns the time in seconds, or None if not found.
    """
    srt_path = os.path.join(folder, "subtitles.srt")
    last_time = None
    
    if os.path.exists(srt_path):
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Look for timestamps in format 00:00:00,000 --> 00:00:00,000
                matches = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", content)
                if matches:
                    # The regex captures 4 groups per timestamp.
                    # Standard SRT has pairs of timestamps. We want the very last one.
                    # It matches both start and end times linearly.
                    h, m, s, ms = matches[-1]
                    last_time = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                    print(f"Found subtitle end time: {last_time:.3f}s")
        except Exception as e:
            print(f"Error reading subtitles: {e}")
            
    return last_time

def create_video(folder_path):
    # Normalize paths
    folder_path = os.path.abspath(folder_path)
    prompts_path = os.path.join(folder_path, "prompts.json")
    images_dir = os.path.join(folder_path, "images")
    output_video = os.path.join(folder_path, "final_video.mp4")
    temp_dir = os.path.join(folder_path, "temp_clips")

    if not os.path.exists(prompts_path):
        print(f"Error: prompts.json not found in {folder_path}")
        return

    if not os.path.exists(images_dir):
        print(f"Error: images folder not found in {folder_path}")
        return

    # Clean and create temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Load prompts
    print("Loading prompts...")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Sort checks
    prompts.sort(key=lambda x: x["id"]) 
    # Actually user says id corresponds to image number.
    # But order of appearance in video is defined by start_time.
    prompts.sort(key=lambda x: x["start_time"])

    # Determine total duration
    sub_end_time = get_video_duration(folder_path)
    last_prompt = prompts[-1]
    
    # If we have subtitle time, use it. Otherwise guess.
    if sub_end_time:
        # buffer just in case
        total_duration = max(sub_end_time + 0.5, last_prompt["start_time"] + 5.0)
    else:
        # Fallback: assume last image lasts 5 seconds
        total_duration = last_prompt["start_time"] + 5.0
        print(f"No subtitle duration found. Defaulting video end to {total_duration:.3f}s")

    print(f"Total Video Duration: {total_duration:.3f}s")
    print(f"Processing {len(prompts)} clips...")

    file_list_path = os.path.join(temp_dir, "files.txt")
    
    # Target resolution and FPS
    WIDTH = 1920
    HEIGHT = 1080
    FPS = 30

    with open(file_list_path, "w", encoding="utf-8") as list_file:
        for i, p in enumerate(prompts):
            clip_id = p["id"]
            
            # Start time
            if i == 0:
                start_time = 0.0 # Force first image to start at 0
            else:
                start_time = p["start_time"]
            
            # End time
            if i < len(prompts) - 1:
                end_time = prompts[i+1]["start_time"]
            else:
                end_time = total_duration

            duration = end_time - start_time
            
            # Sanity check
            if duration <= 0.04: # Less than 1 frame typically 
                print(f"Skipping ID {clip_id}: duration too short ({duration:.4f}s)")
                # If we skip, we create a gap? 
                # Better to just let it be very short or merge?
                # For now warn and define min duration
                duration = 0.1

            # Image path
            # User format: image001, image002...
            # Ensure it matches the actual file extension if not PNG
            img_filename = f"image{clip_id:03d}.png"
            img_path = os.path.join(images_dir, img_filename)

            if not os.path.exists(img_path):
                print(f"Warning: {img_filename} missing. Looking for fallback...")
                # Try finding previous valid image
                found_fallback = False
                for prev_idx in range(i-1, -1, -1):
                    prev_id = prompts[prev_idx]["id"]
                    prev_name = f"image{prev_id:03d}.png"
                    prev_path = os.path.join(images_dir, prev_name)
                    if os.path.exists(prev_path):
                        img_path = prev_path
                        print(f"  -> Using {prev_name} instead.")
                        found_fallback = True
                        break
                
                if not found_fallback:
                    print(f"  -> No fallback found. Skipping this clip.")
                    continue

            # Output clip path
            clip_name = f"clip_{i:04d}.mp4"
            clip_path = os.path.join(temp_dir, clip_name)

            # Calculate total frames for this clip
            frames = int(duration * FPS)
            if frames < 1: 
                frames = 1

            # Zoom effect: 1.0 -> 1.18
            # zoompan filter:
            # d: duration in frames
            # z: zoom expression. 'on' is frame index (0..d-1)
            # z = 1 + 0.18 * (on / frames)
            # s: output size
            # fps: output fps
            
            # Note: We must ensure input is treated as a single image to generate 'd' frames.
            # Only -i image.png is needed, no -loop 1 inputs.
            
            zoom_expr = f"1+0.18*on/{frames}"
            
            # Scale logic in zoompan: x and y center the zoom
            # x='iw/2-(iw/zoom)/2', y='ih/2-(ih/zoom)/2'
            
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", img_path,
                "-vf", (
                    f"zoompan=z='{zoom_expr}':"
                    f"x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2':"
                    f"d={frames}:s=2560x1440:fps={FPS},"
                    f"scale={WIDTH}:{HEIGHT}"
                ),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "23",
                "-pix_fmt", "yuv420p",
                "-t", str(duration), # explicit duration to prevent hanging if zoompan behaves oddly
                clip_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
                # Write to concat list
                # Ffmpeg concat filter expects forward slashes
                safe_path = clip_path.replace("\\", "/")
                list_file.write(f"file '{safe_path}'\n")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process clip {i} (ID {clip_id}): {e}")

            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(prompts)} clips...")

    print("Concatenating clips...")
    concat_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        output_video
    ]
    
    try:
        subprocess.run(concat_cmd, check=True)
        print(f"Successfully created video: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Concatenation failed: {e}")

    # Cleanup optional
    # shutil.rmtree(temp_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from images and prompts.")
    parser.add_argument("folder", help="Path to the video folder (containing prompts.json and images/)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        create_video(args.folder)
    else:
        print(f"Invalid folder path: {args.folder}")
