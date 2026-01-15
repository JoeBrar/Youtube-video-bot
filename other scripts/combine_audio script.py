#!/usr/bin/env python3
"""
Combines numbered audio files and creates an MP4 video with black screen.

Usage:
    python combine_audio.py                    # Use current directory
    python combine_audio.py "path/to/folder"   # Specify folder
"""

import os
import sys
import re
import subprocess
import tempfile
from pathlib import Path


def get_numbered_audio_files(folder: Path) -> list[Path]:
    """Find and sort audio files by their numeric prefix (1.wav, 2.wav, etc.)"""
    audio_extensions = {'.wav', '.mp3'}
    audio_files = []

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in audio_extensions:
            # Match files that start with a number
            match = re.match(r'^(\d+)', file.stem)
            if match:
                audio_files.append((int(match.group(1)), file))

    # Sort by numeric prefix
    audio_files.sort(key=lambda x: x[0])
    return [f[1] for f in audio_files]


def combine_audio(audio_files: list[Path], output_path: Path) -> bool:
    """Combine audio files using ffmpeg concat demuxer."""
    if not audio_files:
        print("No audio files found to combine.")
        return False

    # Create temporary file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for audio_file in audio_files:
            # Escape single quotes and use proper ffmpeg format
            escaped_path = str(audio_file).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        filelist_path = f.name

    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            str(output_path)
        ]

        print(f"Combining {len(audio_files)} audio files...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error combining audio: {result.stderr}")
            return False

        print(f"Created: {output_path}")
        return True
    finally:
        os.unlink(filelist_path)


def create_video(audio_path: Path, output_path: Path) -> bool:
    """Create MP4 video with black screen from audio."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', 'color=c=black:s=1920x1080:r=30',
        '-i', str(audio_path),
        '-c:v', 'libx264',
        '-tune', 'stillimage',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        str(output_path)
    ]

    print("Creating video with black screen...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error creating video: {result.stderr}")
        return False

    print(f"Created: {output_path}")
    return True


def main():
    # Get folder path from argument or use the script's directory
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folder = Path(__file__).parent.resolve()

    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        sys.exit(1)

    print(f"Processing folder: {folder}")

    # Find audio files
    audio_files = get_numbered_audio_files(folder)

    if not audio_files:
        print("No numbered audio files found (expected: 1.wav, 2.wav, etc.)")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  - {f.name}")

    # Output paths
    combined_audio = folder / "combined_voiceover.wav"
    video_output = folder / "voiceover_video.mp4"

    # Step 1: Combine audio
    if not combine_audio(audio_files, combined_audio):
        sys.exit(1)

    # Step 2: Create video
    if not create_video(combined_audio, video_output):
        sys.exit(1)

    print("\nDone!")
    print(f"  Audio: {combined_audio}")
    print(f"  Video: {video_output}")


if __name__ == "__main__":
    main()
