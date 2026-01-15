"""
Channel Manager Module
Manages multiple YouTube channel configurations and iteration
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@dataclass
class Channel:
    """Represents a YouTube channel configuration."""
    id: str
    name: str
    niche: str
    description: str
    content_guidelines: str
    topic_categories: List[str]
    image_style_suffix: str
    target_video_length_minutes: int
    target_word_count: int
    images_per_minute: int

    @property
    def total_images(self) -> int:
        """Calculate total images needed for a video."""
        return self.target_video_length_minutes * self.images_per_minute

    @property
    def words_per_minute(self) -> int:
        """Calculate words per minute based on target."""
        return self.target_word_count // self.target_video_length_minutes

    @classmethod
    def from_dict(cls, data: dict) -> "Channel":
        """Create a Channel from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            niche=data["niche"],
            description=data.get("description", ""),
            content_guidelines=data.get("content_guidelines", ""),
            topic_categories=data.get("topic_categories", []),
            image_style_suffix=data.get("image_style_suffix", ""),
            target_video_length_minutes=data.get("target_video_length_minutes", 20),
            target_word_count=data.get("target_word_count", 3000),
            images_per_minute=data.get("images_per_minute", 4)
        )


class ChannelManager:
    """Manages channel configurations and iteration."""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = config.BASE_DIR / "channels.json"
        self.config_path = Path(config_path)
        self.channels: List[Channel] = self._load_channels()

    def _load_channels(self) -> List[Channel]:
        """Load channels from the configuration file."""
        if not self.config_path.exists():
            print(f"WARNING: Channels config not found at {self.config_path}")
            print("Creating default channels.json with History Epoch channel...")
            self._create_default_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        channels = [Channel.from_dict(ch) for ch in data]
        print(f"Loaded {len(channels)} channel(s): {', '.join(ch.name for ch in channels)}")
        return channels

    def _create_default_config(self):
        """Create a default channels.json with the History Epoch channel."""
        default_channel = {
            "id": "history_epoch",
            "name": "History Epoch",
            "niche": "history",
            "description": "Engaging history documentaries exploring fascinating events and figures",
            "content_guidelines": "Use dramatic, cinematic narration style. Focus on human stories and personal struggles. Include surprising facts and little-known details. Build tension and suspense. End with thought-provoking conclusions.",
            "topic_categories": [
                "historical figures",
                "major events",
                "ancient cities",
                "wars and battles",
                "empires and kingdoms"
            ],
            "image_style_suffix": "realistic, historically accurate, cinematic lighting, highly detailed, 16:9 aspect ratio, dramatic composition",
            "target_video_length_minutes": 20,
            "target_word_count": 3000,
            "images_per_minute": 4
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump([default_channel], f, indent=2)

    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a channel by its ID."""
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        return None

    def get_all_channels(self) -> List[Channel]:
        """Get all configured channels."""
        return self.channels

    def get_channel_ids(self) -> List[str]:
        """Get list of all channel IDs."""
        return [ch.id for ch in self.channels]

    def get_channel_output_dir(self, channel: Channel) -> Path:
        """Get the output directory for a specific channel."""
        return config.OUTPUT_DIR / channel.id

    def get_next_video_number(self, channel: Channel) -> int:
        """Get the next available video number for a channel."""
        channel_dir = self.get_channel_output_dir(channel)
        channel_dir.mkdir(parents=True, exist_ok=True)

        existing = [d.name for d in channel_dir.iterdir() if d.is_dir() and d.name.startswith("video")]
        if not existing:
            return 1

        numbers = []
        for name in existing:
            try:
                num = int(name.replace("video", ""))
                numbers.append(num)
            except ValueError:
                continue

        return max(numbers) + 1 if numbers else 1

    def list_channels(self):
        """Print a formatted list of all channels."""
        print("\nConfigured Channels:")
        print("-" * 60)
        for ch in self.channels:
            print(f"\n  ID: {ch.id}")
            print(f"  Name: {ch.name}")
            print(f"  Niche: {ch.niche}")
            print(f"  Video Length: {ch.target_video_length_minutes} minutes")
            print(f"  Word Count: {ch.target_word_count}")
            print(f"  Images: {ch.total_images} ({ch.images_per_minute}/min)")
            print(f"  Topics: {len(ch.topic_categories)} categories")
        print("-" * 60)


def create_video_folder_for_channel(channel: Channel) -> Path:
    """Create a new video folder for a specific channel."""
    manager = ChannelManager()
    video_num = manager.get_next_video_number(channel)
    channel_dir = manager.get_channel_output_dir(channel)
    video_folder = channel_dir / f"video{video_num}"
    video_folder.mkdir(parents=True, exist_ok=True)
    return video_folder


if __name__ == "__main__":
    # Test the channel manager
    manager = ChannelManager()
    manager.list_channels()

    # Test getting a specific channel
    channel = manager.get_channel("history_epoch")
    if channel:
        print(f"\nHistory Epoch channel found!")
        print(f"  Total images per video: {channel.total_images}")
        print(f"  Output dir: {manager.get_channel_output_dir(channel)}")
        print(f"  Next video number: {manager.get_next_video_number(channel)}")
