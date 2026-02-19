
import sys
import subprocess
import importlib.util

def check_whisper():
    spec = importlib.util.find_spec("whisper")
    return spec is not None

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_openai_api():
    spec = importlib.util.find_spec("openai")
    return spec is not None

print(f"Whisper installed: {check_whisper()}")
print(f"FFmpeg installed: {check_ffmpeg()}")
print(f"OpenAI installed: {check_openai_api()}")
