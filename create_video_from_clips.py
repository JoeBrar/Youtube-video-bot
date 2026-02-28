"""
create_video_from_clips.py

Joins numbered video clips into a single final video, adjusting each clip's
playback speed so it fits the timing defined in prompts.json.

Usage:
    python create_video_from_clips.py "output/history_epoch/video18"
    python create_video_from_clips.py "output/history_epoch/video18" --keep-audio

The script expects:
    - <video_path>/prompts.json   (with id & start_time fields)
    - <video_path>/clips/         (with clip_001.mp4, clip_002.mp4, etc.)
    - <video_path>/voiceover.mp3  (used only to determine total video duration)

Output:
    - <video_path>/final_video.mp4
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import shutil


def get_media_duration(filepath):
    """Get the duration of a media file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting duration of {filepath}: {result.stderr}")
        sys.exit(1)
    return float(result.stdout.strip())


def get_video_properties(filepath):
    """Get width, height, fps of a video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting properties of {filepath}: {result.stderr}")
        sys.exit(1)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    width = stream["width"]
    height = stream["height"]
    # r_frame_rate is like "30/1" or "24000/1001"
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den
    return width, height, fps


def build_atempo_chain(speed_factor):
    """
    Build a chain of atempo filters for the given speed factor.
    Each atempo filter only supports values between 0.5 and 100.0,
    so we chain multiple filters for extreme values.
    """
    filters = []
    remaining = speed_factor

    if remaining >= 1.0:
        # Speeding up: each filter can do up to 100x
        while remaining > 100.0:
            filters.append("atempo=100.0")
            remaining /= 100.0
        filters.append(f"atempo={remaining}")
    else:
        # Slowing down: each filter can do down to 0.5x
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining}")

    return ",".join(filters)


def speed_adjust_clip(input_path, output_path, speed_factor, target_duration, keep_audio=False):
    """
    Re-encode a clip with adjusted speed, strongly enforcing total duration.
    speed_factor > 1 = faster playback, < 1 = slower playback.
    If keep_audio is True, audio is speed-adjusted with pitch correction via atempo.
    """
    # setpts: divide by speed_factor to speed up, multiply to slow down
    # PTS/speed_factor: speed_factor=2 -> plays 2x faster
    video_filter = f"setpts=PTS/{speed_factor}"

    if keep_audio:
        audio_filter = build_atempo_chain(speed_factor)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:v", video_filter,
            "-filter:a", audio_filter,
            "-t", str(target_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:v", video_filter,
            "-an",  # no audio
            "-t", str(target_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            output_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error processing {input_path}:")
        print(result.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Join numbered video clips into a final video with speed adjustments."
    )
    parser.add_argument("video_path", help="Path to the video folder (e.g. output/history_epoch/video18)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Keep clip audio (speed-adjusted). Default is silent video.")
    args = parser.parse_args()

    video_path = args.video_path
    keep_audio = args.keep_audio

    if keep_audio:
        print("Audio mode: KEEPING clip audio (speed-adjusted)")
    else:
        print("Audio mode: SILENT (no audio)")

    # Validate paths
    prompts_path = os.path.join(video_path, "prompts.json")
    clips_dir = os.path.join(video_path, "clips")
    voiceover_path = os.path.join(video_path, "voiceover.mp3")

    for path, label in [(prompts_path, "prompts.json"), (clips_dir, "clips/"), (voiceover_path, "voiceover.mp3")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found at {path}")
            sys.exit(1)

    # Load prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Sort by id
    prompts.sort(key=lambda p: p["id"])
    print(f"Loaded {len(prompts)} prompts (ids {prompts[0]['id']} to {prompts[-1]['id']})")

    # Get voiceover duration (= total video duration)
    total_duration = get_media_duration(voiceover_path)
    print(f"Voiceover duration: {total_duration:.2f}s")

    # Scan for existing clips
    existing_clips = {}
    for prompt in prompts:
        clip_name = f"clip_{prompt['id']:03d}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)
        if os.path.exists(clip_path):
            existing_clips[prompt["id"]] = clip_path

    print(f"Found {len(existing_clips)} clips out of {len(prompts)} prompts")
    if len(existing_clips) == 0:
        print("Error: No clips found!")
        sys.exit(1)

    # Build segments: each segment is a (clip_id, clip_path, start_time, end_time)
    # A present clip covers itself + any immediately following missing clips.
    segments = []
    sorted_clip_ids = sorted(existing_clips.keys())

    for i, clip_id in enumerate(sorted_clip_ids):
        clip_path = existing_clips[clip_id]

        # Find the start_time for this clip's prompt
        clip_prompt = next(p for p in prompts if p["id"] == clip_id)
        start_time = clip_prompt["start_time"]

        # Override: first segment always starts at 0
        if i == 0:
            start_time = 0.0

        # End time: the start_time of the next present clip, or total_duration for the last
        if i < len(sorted_clip_ids) - 1:
            next_clip_id = sorted_clip_ids[i + 1]
            next_prompt = next(p for p in prompts if p["id"] == next_clip_id)
            end_time = next_prompt["start_time"]
        else:
            end_time = total_duration

        target_duration = end_time - start_time
        if target_duration <= 0:
            print(f"Warning: Skipping clip {clip_id} (target_duration={target_duration:.2f}s)")
            continue

        segments.append({
            "clip_id": clip_id,
            "clip_path": clip_path,
            "start_time": start_time,
            "end_time": end_time,
            "target_duration": target_duration,
        })

    print(f"\nBuilt {len(segments)} segments:")
    for seg in segments:
        print(f"  Clip {seg['clip_id']:03d}: {seg['start_time']:.2f}s -> {seg['end_time']:.2f}s "
              f"(target: {seg['target_duration']:.2f}s)")

    # Create temp directory for speed-adjusted clips
    temp_dir = tempfile.mkdtemp(prefix="video_assembly_")
    print(f"\nTemp directory: {temp_dir}")

    try:
        adjusted_clips = []
        
        # Track actual video time to compensate for rounding errors
        current_actual_video_time = 0.0

        for idx, seg in enumerate(segments):
            clip_duration = get_media_duration(seg["clip_path"])

            # Calculate the target duration dynamically based on the exact end time
            # we need to hit, minus whatever actual length the video has accumulated so far.
            # This completely eliminates cumulative drift over the course of the video.
            target_duration = seg["end_time"] - current_actual_video_time

            print(f"\nProcessing clip {seg['clip_id']:03d}:")
            print(f"  Expected end time: {seg['end_time']:.3f}s")
            print(f"  Current video time: {current_actual_video_time:.3f}s")
            
            if target_duration <= 0.05: # safety threshold to avoid 0x speed crashes
                print(f"  -> Skipping! Drift pushed us past this clip's end time (target = {target_duration:.3f}s).")
                continue

            speed_factor = clip_duration / target_duration

            print(f"  Original duration: {clip_duration:.2f}s")
            print(f"  Compensated target duration: {target_duration:.3f}s")
            print(f"  Speed factor:      {speed_factor:.4f}x")

            if speed_factor > 1:
                print(f"  -> Fast motion ({speed_factor:.2f}x faster)")
            elif speed_factor < 1:
                print(f"  -> Slow motion ({1/speed_factor:.2f}x slower)")
            else:
                print(f"  -> Normal speed")

            adjusted_path = os.path.join(temp_dir, f"seg_{idx:04d}.mp4")
            speed_adjust_clip(seg["clip_path"], adjusted_path, speed_factor, target_duration, keep_audio)

            # Verify the adjusted clip duration
            actual_adjusted_duration = get_media_duration(adjusted_path)
            print(f"  Actual adjusted duration: {actual_adjusted_duration:.3f}s "
                  f"(target was {target_duration:.3f}s)")
                  
            # Update our running timer with reality, not theory.
            current_actual_video_time += actual_adjusted_duration

            adjusted_clips.append(adjusted_path)

        # Create concat list file
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for clip_path in adjusted_clips:
                # FFmpeg requires forward slashes or escaped backslashes
                escaped_path = clip_path.replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")

        # Get properties from first clip to ensure consistent output
        width, height, fps = get_video_properties(adjusted_clips[0])
        print(f"\nOutput resolution: {width}x{height} @ {fps:.2f} fps")

        # Concatenate all adjusted clips
        output_path = os.path.join(video_path, "final_video.mp4")
        print(f"\nConcatenating {len(adjusted_clips)} clips into {output_path}...")

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-movflags", "+faststart",
            output_path
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error concatenating clips:")
            print(result.stderr)
            sys.exit(1)

        # Verify final output
        final_duration = get_media_duration(output_path)
        print(f"\n{'='*50}")
        print(f"Final video created: {output_path}")
        print(f"Final duration:      {final_duration:.2f}s")
        print(f"Expected duration:   {total_duration:.2f}s")
        print(f"Difference:          {abs(final_duration - total_duration):.2f}s")
        print(f"{'='*50}")

    finally:
        # Clean up temp directory
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
