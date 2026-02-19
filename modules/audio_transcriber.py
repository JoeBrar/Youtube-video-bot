import os
import torch
import whisper
from pathlib import Path

class AudioTranscriber:
    def __init__(self, model_size="medium", device=None):
        """
        Initialize the AudioTranscriber with a local Whisper model.
        
        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large).
                              'medium' is recommended for good balance of speed/accuracy on GPU.
            device (str): Device to run on ('cuda' or 'cpu'). If None, auto-detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing Whisper model '{model_size}' on {self.device}...")
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to CPU if CUDA failed...")
            self.device = "cpu"
            self.model = whisper.load_model(model_size, device="cpu")

    def transcribe_file(self, audio_path: Path) -> str:
        """
        Transcribe an audio file to SRT format using the local Whisper model.
        
        Args:
            audio_path (Path): Path to the audio file.
            
        Returns:
            str: The generated SRT content.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Transcribing {audio_path.name}...")
        
        # Transcribe
        result = self.model.transcribe(str(audio_path), fp16=(self.device == "cuda"))
        
        # Generate SRT content
        srt_content = ""
        for i, segment in enumerate(result["segments"], start=1):
            start = self._format_timestamp(segment["start"])
            end = self._format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
            
        return srt_content

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into SRT timestamp format (HH:MM:SS,mmm)."""
        milliseconds = int((seconds - int(seconds)) * 1000)
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def save_srt(self, srt_content: str, output_path: Path):
        """Save SRT content to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f"SRT saved to: {output_path}")

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
        transcriber = AudioTranscriber()
        srt = transcriber.transcribe_file(audio_file)
        print(srt[:500])
    else:
        print("Usage: python audio_transcriber.py <audio_file>")
