"""
State Manager Module
Tracks progress and enables resumability for image generation
"""
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class VideoState(Enum):
    INITIALIZED = "initialized"
    SCRIPT_GENERATED = "script_generated"
    PROMPTS_GENERATED = "prompts_generated"
    IMAGES_IN_PROGRESS = "images_in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StateManager:
    def __init__(self, video_folder: Path):
        self.video_folder = Path(video_folder)
        self.state_file = self.video_folder / "state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load state from file or create new state."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._create_initial_state()

    def _create_initial_state(self) -> dict:
        """Create initial state structure."""
        # Extract channel_id from folder path (output/channel_id/videoX)
        channel_id = self.video_folder.parent.name if self.video_folder.parent != config.OUTPUT_DIR else None

        return {
            "video_id": self.video_folder.name,
            "channel_id": channel_id,
            "status": VideoState.INITIALIZED.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "topic": None,
            "script_word_count": 0,
            "estimated_duration_minutes": 0,
            "total_images_required": 0,
            "images": {
                "completed": [],  # List of completed image indices
                "in_progress": [],  # List of images currently generating
                "failed": [],  # List of failed image indices
                "pending": []  # List of pending image indices
            },
            "image_urls": {},  # {clip_number: url} for generated images
            "errors": []  # List of error messages
        }

    def save(self):
        """Save current state to file."""
        self.state["updated_at"] = datetime.now().isoformat()
        self.video_folder.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)

    def get_status(self) -> VideoState:
        """Get current status."""
        return VideoState(self.state["status"])

    def set_status(self, status: VideoState):
        """Set current status."""
        self.state["status"] = status.value
        self.save()

    def set_topic(self, topic: str):
        """Set the video topic."""
        self.state["topic"] = topic
        self.save()

    def set_script_info(self, word_count: int, duration_minutes: float, total_images: int):
        """Set script-related information."""
        self.state["script_word_count"] = word_count
        self.state["estimated_duration_minutes"] = duration_minutes
        self.state["total_images_required"] = total_images
        # Initialize pending list
        self.state["images"]["pending"] = list(range(1, total_images + 1))
        self.save()

    def mark_image_started(self, clip_number: int):
        """Mark an image as started generating."""
        if clip_number in self.state["images"]["pending"]:
            self.state["images"]["pending"].remove(clip_number)
        if clip_number not in self.state["images"]["in_progress"]:
            self.state["images"]["in_progress"].append(clip_number)
        self.save()

    def mark_image_completed(self, clip_number: int, image_url: str = None):
        """Mark an image as completed."""
        if clip_number in self.state["images"]["in_progress"]:
            self.state["images"]["in_progress"].remove(clip_number)
        if clip_number in self.state["images"]["pending"]:
            self.state["images"]["pending"].remove(clip_number)
        if clip_number not in self.state["images"]["completed"]:
            self.state["images"]["completed"].append(clip_number)
        if image_url:
            self.state["image_urls"][str(clip_number)] = image_url
        self.save()

    def mark_image_failed(self, clip_number: int, error: str = None):
        """Mark an image as failed."""
        if clip_number in self.state["images"]["in_progress"]:
            self.state["images"]["in_progress"].remove(clip_number)
        if clip_number not in self.state["images"]["failed"]:
            self.state["images"]["failed"].append(clip_number)
        if error:
            self.state["errors"].append({
                "type": "image",
                "clip_number": clip_number,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
        self.save()

    def get_pending_images(self) -> list[int]:
        """Get list of images that still need to be generated."""
        return sorted(self.state["images"]["pending"] + self.state["images"]["failed"])

    def get_in_progress_images(self) -> list[int]:
        """Get list of images currently being generated."""
        return self.state["images"]["in_progress"]

    def get_image_url(self, clip_number: int) -> Optional[str]:
        """Get the URL for a generated image."""
        return self.state["image_urls"].get(str(clip_number))

    def is_complete(self) -> bool:
        """Check if all processing is complete (including skipped/failed images)."""
        total = self.state["total_images_required"]
        if total == 0:
            return False
        images_completed = len(self.state["images"]["completed"])
        images_failed = len(self.state["images"]["failed"])
        # Consider complete if all images are either completed or failed (skipped)
        return (images_completed + images_failed) == total

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        total = self.state["total_images_required"]
        img_complete = len(self.state["images"]["completed"])
        img_progress = len(self.state["images"]["in_progress"])
        img_failed = len(self.state["images"]["failed"])
        channel_id = self.state.get('channel_id', 'unknown')

        return f"""
Progress for {channel_id}/{self.state['video_id']}:
  Channel: {channel_id}
  Topic: {self.state['topic']}
  Status: {self.state['status']}

  Images: {img_complete}/{total} completed, {img_progress} in progress, {img_failed} skipped/failed

  Last updated: {self.state['updated_at']}
"""

    def retry_failed(self):
        """Move failed items back to pending for retry."""
        for clip in self.state["images"]["failed"]:
            if clip not in self.state["images"]["pending"]:
                self.state["images"]["pending"].append(clip)
        self.state["images"]["failed"] = []
        self.save()


def get_next_video_number() -> int:
    """Get the next available video number.

    DEPRECATED: Use ChannelManager.get_next_video_number(channel) instead.
    This function is kept for backwards compatibility.
    """
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search across all channel directories
    all_numbers = []
    for channel_dir in output_dir.iterdir():
        if channel_dir.is_dir():
            for video_dir in channel_dir.iterdir():
                if video_dir.is_dir() and video_dir.name.startswith("video"):
                    try:
                        num = int(video_dir.name.replace("video", ""))
                        all_numbers.append(num)
                    except ValueError:
                        continue

    return max(all_numbers) + 1 if all_numbers else 1


def find_resumable_video(channel_id: str = None) -> Optional[Path]:
    """Find an incomplete video that can be resumed.

    Args:
        channel_id: Optional channel ID to search within. If None, searches all channels.

    Returns:
        Path to the video folder that can be resumed, or None if no resumable videos found.
    """
    output_dir = config.OUTPUT_DIR
    if not output_dir.exists():
        return None

    # Determine which channel directories to search
    if channel_id:
        channel_dirs = [output_dir / channel_id]
    else:
        channel_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]

    for channel_dir in channel_dirs:
        if not channel_dir.exists():
            continue

        for video_dir in sorted(channel_dir.iterdir()):
            if video_dir.is_dir() and video_dir.name.startswith("video"):
                state_file = video_dir / "state.json"
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    if state["status"] not in [VideoState.COMPLETED.value, VideoState.FAILED.value]:
                        return video_dir

    return None


if __name__ == "__main__":
    # Test state manager
    from modules.channel_manager import ChannelManager

    manager = ChannelManager()
    channel = manager.get_channel("history_epoch")

    if channel:
        test_folder = manager.get_channel_output_dir(channel) / "test_video"
        test_folder.mkdir(parents=True, exist_ok=True)

        sm = StateManager(test_folder)
        sm.set_topic("Test Topic")
        sm.set_script_info(3000, 20.0, 80)

        print(sm.get_progress_summary())

        # Simulate some progress
        sm.mark_image_started(1)
        sm.mark_image_completed(1, "http://example.com/image1.png")

        print(sm.get_progress_summary())

        # Cleanup test
        import shutil
        shutil.rmtree(test_folder)
    else:
        print("ERROR: Could not find history_epoch channel")
