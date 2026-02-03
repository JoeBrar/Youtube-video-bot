"""
Retry Missing Images Script - Nano Banana Pro

Regenerates missing images from a video folder using Nano Banana Pro model.
This model is less restrictive than Seedream 4.5 and may successfully generate
images that were previously blocked.

Usage:
    python retry_missing_images.py <video_folder>
    python retry_missing_images.py output/history_epoch/video10

Or run from inside a video folder:
    cd output/history_epoch/video10
    python ../../../retry_missing_images.py
"""

import asyncio
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from modules.higgsfield_client import HiggsfieldClient, ImageTabManager
from modules.state_manager import StateManager
from modules.file_manager import FileManager


class NanoBananaImageTab(ImageTabManager):
    """Extended ImageTabManager for Nano Banana Pro model."""

    NANO_BANANA_URL = "https://higgsfield.ai/image/nano_banana_2"
    MAX_CONCURRENT = 4  # Nano Banana Pro limit

    async def switch_to_nano_banana(self):
        """Navigate to Nano Banana Pro and enable unlimited mode."""
        print("\n[4/5] Switching to Nano Banana Pro...")
        print(f"  Navigating to {self.NANO_BANANA_URL}")

        try:
            await self.page.goto(self.NANO_BANANA_URL, wait_until="load", timeout=60000)
            await asyncio.sleep(3)

            print("  Closing popups...")
            await self._close_popups()
            await asyncio.sleep(1)

            print("  Enabling Unlimited mode...")
            await self._enable_unlimited_mode()
            await asyncio.sleep(1)

            # Capture existing gallery count
            self.initial_gallery_count = await self._get_gallery_count()
            print(f"  Found {self.initial_gallery_count} existing images in gallery")

            print("  Nano Banana Pro ready!")

        except Exception as e:
            print(f"  ERROR: Failed to switch to Nano Banana Pro: {e}")
            raise


class MissingImageRetrier:
    """Main orchestration class for detecting and regenerating missing images."""

    def __init__(self, video_folder: Path):
        self.video_folder = video_folder.resolve()
        self.file_manager = FileManager(self.video_folder)
        self.state_manager = StateManager(self.video_folder)
        self.client: Optional[HiggsfieldClient] = None
        self.image_tab: Optional[NanoBananaImageTab] = None

    def load_prompts(self) -> List[Dict]:
        """Load prompts from prompts.json."""
        print("\n[1/5] Loading prompts from prompts.json...")
        prompts = self.file_manager.load_prompts()
        print(f"  Found {len(prompts)} prompts")
        return prompts

    def detect_missing_images(self, prompts: List[Dict]) -> List[int]:
        """
        Compare prompts.json against actual image files to find missing images.

        Returns:
            Sorted list of missing clip numbers
        """
        print("\n[2/5] Detecting missing images...")

        # Get required clip numbers from prompts
        required = set()
        for p in prompts:
            clip_num = p.get('clip_number', p.get('id'))
            if clip_num is not None:
                required.add(clip_num)

        # Get existing clip numbers from images/ folder
        existing = set()
        images_folder = self.video_folder / "images"

        if images_folder.exists():
            for file in images_folder.glob("image*.png"):
                # Extract number from image042.png → 42
                match = re.match(r'image(\d+)\.png', file.name)
                if match:
                    existing.add(int(match.group(1)))

        # Calculate missing
        missing = sorted(required - existing)

        print(f"  Found {len(existing)} existing images")
        print(f"  Missing {len(missing)} images: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

        return missing

    async def _generate_missing_images(self, prompts: List[Dict], missing_clip_numbers: List[int]):
        """Generate missing images with concurrency limit of 4."""
        print(f"\n[5/5] Generating {len(missing_clip_numbers)} missing images (max {NanoBananaImageTab.MAX_CONCURRENT} concurrent)...")
        print("-" * 60)

        # Create prompts map for quick lookup
        prompts_map = {}
        for p in prompts:
            clip_num = p.get('clip_number', p.get('id'))
            if clip_num is not None:
                prompts_map[clip_num] = p.get('prompt', '')

        # Initialize tracking
        missing_queue = list(missing_clip_numbers)
        in_progress: Dict[int, str] = {}
        completed = []
        failed = []

        # Track first submission for timing
        first_submission = True

        while missing_queue or in_progress:
            # Submit new images (respect MAX_CONCURRENT)
            while len(in_progress) < NanoBananaImageTab.MAX_CONCURRENT and missing_queue:
                clip_number = missing_queue.pop(0)
                prompt = prompts_map.get(clip_number)

                if not prompt:
                    print(f"  WARNING: No prompt found for image {clip_number}, skipping...")
                    continue

                # Calculate pre-wait
                pre_wait = 10 if first_submission else 0

                # Submit the prompt
                success = await self.image_tab.submit_prompt(clip_number, prompt, pre_wait_seconds=pre_wait)

                if success:
                    in_progress[clip_number] = prompt
                    self.state_manager.mark_image_started(clip_number)

                    if first_submission:
                        first_submission = False
                    
                    await asyncio.sleep(1)
                else:
                    # Submission failed, put back in queue
                    missing_queue.append(clip_number)
                    await asyncio.sleep(2)

            # Check for completions
            await self.image_tab.check_completions()

            # Process completed images
            for clip_number in list(in_progress.keys()):
                # Check if completed
                if clip_number in self.image_tab.completed:
                    url = self.image_tab.completed[clip_number]
                    image_path = self.file_manager.get_image_path(clip_number)

                    print(f"  [Image {clip_number}] Generation completed! Downloading...")

                    # Download the image
                    success = await self.client.download_file(url, image_path)

                    if success:
                        print(f"  [Image {clip_number}] Saved to {image_path.name}")
                        self.state_manager.mark_image_completed(clip_number, str(image_path))
                        completed.append(clip_number)
                        del in_progress[clip_number]

                        # Remove from image_tab tracking
                        del self.image_tab.completed[clip_number]
                        if clip_number in self.image_tab.in_flight:
                            del self.image_tab.in_flight[clip_number]
                    else:
                        print(f"  [Image {clip_number}] Download failed, will retry...")
                        await asyncio.sleep(5)

                # Check if blocked/skipped
                elif clip_number in self.image_tab.skipped:
                    reason = self.image_tab.skipped[clip_number]
                    print(f"  [Image {clip_number}] BLOCKED: {reason}")
                    self.state_manager.mark_image_failed(clip_number, f"Blocked: {reason}")
                    failed.append(clip_number)
                    del in_progress[clip_number]

                    # Remove from image_tab tracking
                    del self.image_tab.skipped[clip_number]
                    if clip_number in self.image_tab.in_flight:
                        del self.image_tab.in_flight[clip_number]

            # Brief pause before next iteration
            if in_progress:
                await asyncio.sleep(2)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total missing: {len(missing_clip_numbers)}")
        print(f"Successfully generated: {len(completed)}")
        print(f"Failed/blocked: {len(failed)}")

        if failed:
            print(f"\nFailed images: {failed}")
            print("Note: You may need to manually edit prompts for failed images and retry.")

    async def run(self):
        """Main execution flow."""
        try:
            # Load prompts
            prompts = self.load_prompts()

            # Detect missing images
            missing = self.detect_missing_images(prompts)

            if not missing:
                print("\nSUCCESS: No missing images. All images already exist.")
                return

            # Connect to browser
            print("\n[3/5] Connecting to browser...")
            self.client = HiggsfieldClient(
                state_manager=self.state_manager,
                download_dir=self.video_folder
            )
            await self.client.connect()

            # Create page and image tab
            page = await self.client.create_page()
            self.image_tab = NanoBananaImageTab(page, self.client)

            # Switch to Nano Banana Pro
            await self.image_tab.switch_to_nano_banana()

            # Generate missing images
            await self._generate_missing_images(prompts, missing)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress has been saved.")
            print("You can re-run this script to continue from where it left off.")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.client:
                await self.client.disconnect()


def resolve_video_folder() -> Path:
    """
    Resolve video folder from CLI args or current directory.

    Returns:
        Absolute path to video folder
    """
    if len(sys.argv) >= 2:
        video_path = Path(sys.argv[1])
        if video_path == Path('.'):
            video_path = Path.cwd()
    else:
        # No argument: check if cwd is a video folder
        if (Path.cwd() / "prompts.json").exists():
            video_path = Path.cwd()
        else:
            print("=" * 60)
            print("ERROR: No video folder specified")
            print("=" * 60)
            print("\nUsage:")
            print("  python retry_missing_images.py <video_folder>")
            print("  python retry_missing_images.py output/history_epoch/video10")
            print("\nOr run from inside a video folder:")
            print("  cd output/history_epoch/video10")
            print("  python ../../../retry_missing_images.py")
            print()
            sys.exit(1)

    # Convert to absolute path
    video_path = video_path.resolve()

    # Validate folder exists
    if not video_path.exists():
        print(f"ERROR: Folder not found: {video_path}")
        sys.exit(1)

    # Validate prompts.json exists
    if not (video_path / "prompts.json").exists():
        print(f"ERROR: prompts.json not found in {video_path}")
        print("This script requires an existing video folder with prompts.")
        sys.exit(1)

    return video_path


async def main():
    """Main entry point."""
    print("=" * 60)
    print("RETRY MISSING IMAGES - NANO BANANA PRO")
    print("=" * 60)

    # Resolve video folder path
    video_folder = resolve_video_folder()
    print(f"Video folder: {video_folder}")
    print("=" * 60)

    # Create retrier and run
    retrier = MissingImageRetrier(video_folder)
    await retrier.run()


if __name__ == "__main__":
    asyncio.run(main())
