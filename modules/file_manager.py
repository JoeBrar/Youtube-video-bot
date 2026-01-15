"""
File Manager Module
Handles file organization and management for video projects
"""
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from modules.text_utils import clean_ai_text

if TYPE_CHECKING:
    from modules.channel_manager import Channel


class FileManager:
    def __init__(self, video_folder: Path):
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)

    def save_script(self, script: str):
        """Save the voiceover script."""
        script_file = self.video_folder / "script.txt"
        script = clean_ai_text(script)
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script)
        print(f"Script saved to {script_file}")

    def save_titles(self, titles: list[str]):
        """Save YouTube title ideas."""
        titles_file = self.video_folder / "titles.txt"
        titles = [clean_ai_text(title) for title in titles]
        with open(titles_file, 'w', encoding='utf-8') as f:
            f.write("YouTube Title Ideas\n")
            f.write("=" * 50 + "\n\n")
            for i, title in enumerate(titles, 1):
                f.write(f"{i}. {title}\n")
        print(f"Titles saved to {titles_file}")

    def save_description(self, description: str):
        """Save YouTube video description."""
        desc_file = self.video_folder / "description.txt"
        description = clean_ai_text(description)
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"Description saved to {desc_file}")

    def save_prompts(self, prompts: list[dict]):
        """Save image prompts to JSON file."""
        prompts_file = self.video_folder / "prompts.json"
        # Clean text in prompts
        cleaned_prompts = []
        for prompt in prompts:
            cleaned_prompt = prompt.copy()
            if 'prompt' in cleaned_prompt:
                cleaned_prompt['prompt'] = clean_ai_text(cleaned_prompt['prompt'])
            if 'segment_text' in cleaned_prompt:
                cleaned_prompt['segment_text'] = clean_ai_text(cleaned_prompt['segment_text'])
            cleaned_prompts.append(cleaned_prompt)
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_prompts, f, indent=2, ensure_ascii=False)
        print(f"Prompts saved to {prompts_file}")

    def load_prompts(self) -> list[dict]:
        """Load image prompts from JSON file."""
        prompts_file = self.video_folder / "prompts.json"
        if prompts_file.exists():
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def get_image_path(self, clip_number: int) -> Path:
        """Get the path for an image file."""
        return self.video_folder / "images" / f"image{clip_number:03d}.png"

    def ensure_images_folder(self):
        """Create images subfolder if needed."""
        images_folder = self.video_folder / "images"
        images_folder.mkdir(exist_ok=True)
        return images_folder

    def save_content(self, topic: str, script: str, titles: list[str], description: str, prompts: list[dict]):
        """Save all content files."""
        # Save topic
        topic_file = self.video_folder / "topic.txt"
        topic = clean_ai_text(topic)
        with open(topic_file, 'w', encoding='utf-8') as f:
            f.write(topic)

        self.save_script(script)
        self.save_titles(titles)
        self.save_description(description)
        self.save_prompts(prompts)

    def load_script(self) -> str:
        """Load script from file."""
        script_file = self.video_folder / "script.txt"
        if script_file.exists():
            with open(script_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def load_topic(self) -> str:
        """Load topic from file."""
        topic_file = self.video_folder / "topic.txt"
        if topic_file.exists():
            with open(topic_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""

    def get_video_info(self) -> dict:
        """Get summary of what's in this video folder."""
        info = {
            "folder": str(self.video_folder),
            "has_script": (self.video_folder / "script.txt").exists(),
            "has_titles": (self.video_folder / "titles.txt").exists(),
            "has_description": (self.video_folder / "description.txt").exists(),
            "has_prompts": (self.video_folder / "prompts.json").exists(),
            "images": []
        }

        # Count images
        images_folder = self.video_folder / "images"
        if images_folder.exists():
            for f in images_folder.glob("image*.png"):
                info["images"].append(f.name)
            info["images"].sort()
        info["image_count"] = len(info["images"])

        return info


def create_new_video_folder(channel: "Channel") -> Path:
    """Create a new video folder for a specific channel.

    Args:
        channel: The Channel to create a video folder for

    Returns:
        Path to the new video folder: output/{channel_id}/videoX/
    """
    from modules.channel_manager import ChannelManager

    manager = ChannelManager()
    video_num = manager.get_next_video_number(channel)
    channel_dir = manager.get_channel_output_dir(channel)
    video_folder = channel_dir / f"video{video_num}"
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"Created new video folder: {video_folder}")
    return video_folder


def list_video_folders(channel_id: str = None) -> list[dict]:
    """List all video folders with their info.

    Args:
        channel_id: Optional channel ID to filter by. If None, lists all channels.

    Returns:
        List of video info dictionaries
    """
    output_dir = config.OUTPUT_DIR
    if not output_dir.exists():
        return []

    videos = []

    # Get channel directories to search
    if channel_id:
        channel_dirs = [output_dir / channel_id]
    else:
        # Search all channel directories
        channel_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    for channel_dir in channel_dirs:
        if not channel_dir.exists():
            continue

        for folder in sorted(channel_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("video"):
                fm = FileManager(folder)
                info = fm.get_video_info()
                info["channel_id"] = channel_dir.name

                # Add state info if available
                state_file = folder / "state.json"
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    info["status"] = state.get("status", "unknown")
                    info["topic"] = state.get("topic", "")
                else:
                    info["status"] = "no_state"
                    info["topic"] = fm.load_topic()

                videos.append(info)

    return videos


if __name__ == "__main__":
    # Test file manager
    print("Existing video folders:")
    for v in list_video_folders():
        channel = v.get('channel_id', 'unknown')
        print(f"  [{channel}] {v['folder']}: {v.get('topic', 'No topic')[:40]} - {v['status']}")
