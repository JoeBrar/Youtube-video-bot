"""
YouTube Video Creator Bot - Main Orchestrator
Creates complete YouTube video content packages for multiple channels
"""
import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

import config
from modules.script_generator import ScriptGenerator
from modules.prompt_generator import PromptGenerator
from modules.state_manager import StateManager, VideoState, find_resumable_video
from modules.file_manager import FileManager, create_new_video_folder, list_video_folders
from modules.higgsfield_client import HiggsfieldClient, HiggsfieldBatchProcessor
from modules.channel_manager import ChannelManager, Channel


def print_banner():
    """Print the application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           YouTube Video Creator Bot - Multi-Channel           ║
║                                                               ║
║    Automated content generation for faceless YouTube channels  ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_api_key():
    """Check if API key is configured for the selected provider."""
    provider = config.AI_PROVIDER.lower()

    if provider == "openai":
        if config.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            print("ERROR: OpenAI API key not configured!")
            print("\nPlease edit config.py and set your OPENAI_API_KEY")
            return False
    elif provider == "gemini":
        if config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            print("ERROR: Gemini API key not configured!")
            print("\nPlease edit config.py and set your GEMINI_API_KEY")
            return False
    elif provider == "grok":
        if config.GROK_API_KEY == "YOUR_GROK_API_KEY_HERE":
            print("ERROR: Grok API key not configured!")
            print("\nPlease edit config.py and set your GROK_API_KEY")
            return False
    else:
        print(f"ERROR: Unknown AI provider: {provider}")
        print("Valid options: openai, gemini, grok")
        return False

    return True


async def create_new_video(channel: Channel, topic: str = None, video_id: int = None, client: HiggsfieldClient = None) -> bool:
    """Create a new video from scratch for a specific channel.

    Args:
        channel: The Channel to create a video for
        topic: Optional topic for the video (if provided, video_id should also be provided)
        video_id: Optional video ID for the folder name
        client: Optional existing HiggsfieldClient to reuse browser connection

    Returns:
        True if video was created, False if skipped (no topic available)
    """
    print("\n" + "=" * 60)
    print(f"STARTING NEW VIDEO CREATION FOR: {channel.name}")
    print("=" * 60)

    # Phase 1: Generate Script (save each piece progressively so user can read while generating)
    print(f"\n[PHASE 1] Generating Script with {config.AI_PROVIDER.upper()} AI...")
    print("-" * 40)

    script_gen = ScriptGenerator(channel)

    # Get topic from channel_topics.json (no AI generation)
    if topic is None:
        print("Getting topic from channel_topics.json...")
        result = script_gen.get_topic()
        if result is None:
            print(f"No topics available for {channel.name} in channel_topics.json. Skipping...")
            return False
        topic, video_id = result
    print(f"Topic: {topic}")

    # Create video folder using the video_id from channel_topics.json
    video_folder = create_new_video_folder(channel, video_id=video_id)
    state = StateManager(video_folder)
    files = FileManager(video_folder)

    # Save topic
    files.save_topic(topic)
    state.set_topic(topic)

    # Generate and save script immediately (user can start reading now!)
    print("Generating script...")
    script = script_gen.generate_script(topic)
    word_count = len(script.split())
    duration = script_gen.estimate_duration_minutes(script)
    files.save_script(script)
    print(f"Script generated: {word_count} words (~{duration:.1f} minutes)")

    # Generate and save titles
    print("Generating titles...")
    titles = script_gen.generate_titles(topic, script)
    files.save_titles(titles)

    # Generate and save description
    print("Generating description...")
    description = script_gen.generate_description(topic, script)
    files.save_description(description)

    state.set_status(VideoState.SCRIPT_GENERATED)

    # Phase 2: Generate Image Prompts (script.txt is already available to read!)
    print("\n[PHASE 2] Generating Image Prompts...")
    print("-" * 40)

    prompt_gen = PromptGenerator(channel)
    prompts = prompt_gen.generate_all_prompts(topic, script)
    files.save_prompts(prompts)

    state.set_script_info(word_count, duration, len(prompts))
    state.set_status(VideoState.PROMPTS_GENERATED)

    print(f"\nGenerated {len(prompts)} image prompts")

    # Continue with image generation
    await generate_media(video_folder, channel, client)
    return True


async def resume_video(video_folder: Path, channel_manager: ChannelManager, continue_after: bool = False, client: HiggsfieldClient = None):
    """Resume video generation from where it left off.

    Args:
        video_folder: Path to the video folder to resume
        channel_manager: ChannelManager instance for channel lookup
        continue_after: If True, continue creating new videos after this one completes
        client: Optional existing HiggsfieldClient to reuse browser connection
    """
    # Extract channel_id from folder path (output/channel_id/videoX)
    channel_id = video_folder.parent.name
    channel = channel_manager.get_channel(channel_id)

    if not channel:
        print(f"ERROR: Channel '{channel_id}' not found in channels.json")
        return

    print("\n" + "=" * 60)
    print(f"RESUMING VIDEO: {channel.name}/{video_folder.name}")
    print("=" * 60)

    state = StateManager(video_folder)
    print(state.get_progress_summary())

    status = state.get_status()

    if status == VideoState.COMPLETED:
        print("This video is already completed!")
        if continue_after:
            await multi_channel_video_creation(channel_manager, client)
        return

    if status in [VideoState.INITIALIZED]:
        print("Video has no script yet. Starting fresh...")
        topic = state.state.get("topic")
        await create_new_video(channel, topic, client)
        if continue_after:
            await multi_channel_video_creation(channel_manager, client)
        return

    if status == VideoState.SCRIPT_GENERATED:
        # Need to generate prompts
        files = FileManager(video_folder)
        topic = files.load_topic()
        script = files.load_script()

        if not script:
            print("ERROR: Script file not found!")
            return

        print("\n[PHASE 2] Generating Image Prompts...")
        prompt_gen = PromptGenerator(channel)
        prompts = prompt_gen.generate_all_prompts(topic, script)
        files.save_prompts(prompts)

        state.set_script_info(
            len(script.split()),
            len(script.split()) / channel.words_per_minute,
            len(prompts)
        )
        state.set_status(VideoState.PROMPTS_GENERATED)

    # Retry any failed items
    state.retry_failed()

    # Continue with media generation
    await generate_media(video_folder, channel, client)

    # Continue creating new videos if requested
    if continue_after:
        await multi_channel_video_creation(channel_manager, client)


async def multi_channel_video_creation(channel_manager: ChannelManager, client: HiggsfieldClient = None, specific_channel: Channel = None, single_video: bool = False):
    """Create videos for multiple channels in round-robin fashion.

    Topics are strictly loaded from channel_topics.json.
    Channels with no available topics are skipped.

    Args:
        channel_manager: ChannelManager instance
        client: Optional existing HiggsfieldClient to reuse browser connection
        specific_channel: If provided, start round-robin from this channel
        single_video: If True, exit after creating one video
    """
    video_count = 1
    all_channels = channel_manager.get_all_channels()
    if specific_channel:
        # Rotate list to start from specified channel
        start_idx = next((i for i, c in enumerate(all_channels) if c.id == specific_channel.id), 0)
        channels = all_channels[start_idx:] + all_channels[:start_idx]
    else:
        channels = all_channels

    if not channels:
        print("ERROR: No channels configured. Please check channels.json")
        return

    # Track channels with no remaining topics
    exhausted_channels = set()

    while True:
        # Check if all channels are exhausted
        if len(exhausted_channels) >= len(channels):
            print("\n" + "=" * 60)
            print("ALL CHANNELS EXHAUSTED - No more topics in channel_topics.json")
            print("=" * 60)
            print("Add more topics to channel_topics.json to continue.")
            return

        for channel in channels:
            # Skip channels with no remaining topics
            if channel.id in exhausted_channels:
                continue

            print("\n" + "=" * 60)
            print(f"STARTING VIDEO #{video_count} FOR CHANNEL: {channel.name}")
            print("=" * 60)
            print("Press Ctrl+C to stop after current video completes.\n")

            try:
                # Topics come strictly from channel_topics.json
                success = await create_new_video(channel, client=client)

                if not success:
                    # No topic available for this channel
                    exhausted_channels.add(channel.id)
                    print(f"Channel {channel.name} has no more topics. Skipping in future rounds.")
                    continue

                video_count += 1

                # Exit after first video if single-video mode
                if single_video:
                    print("\n[SINGLE VIDEO MODE] One video completed. Exiting.")
                    return
            except KeyboardInterrupt:
                print("\n\nStopping continuous mode. Last video was saved.")
                return
            except Exception as e:
                print(f"\nError creating video for {channel.name}: {e}")
                print("Continuing to next channel in 10 seconds...")
                import time
                time.sleep(10)


async def run_continuous_with_persistent_browser(channel_manager: ChannelManager, channel_id: str = None, resume_folder: Path = None, single_video: bool = False):
    """Run continuous video creation with a single persistent browser session.

    Topics are strictly loaded from channel_topics.json.

    Args:
        channel_manager: ChannelManager instance
        channel_id: Optional specific channel ID to use (None = round-robin all channels)
        resume_folder: Optional video folder to resume before continuing
        single_video: If True, exit after creating one video
    """
    # Get specific channel if requested
    specific_channel = None
    if channel_id and channel_id != "all":
        specific_channel = channel_manager.get_channel(channel_id)
        if not specific_channel:
            print(f"ERROR: Channel '{channel_id}' not found in channels.json")
            print(f"Available channels: {', '.join(channel_manager.get_channel_ids())}")
            return

    # Create a single client that persists across all videos
    client = HiggsfieldClient(None, None)

    try:
        print("\n[BROWSER] Starting persistent browser session...")
        await client.connect()
        print("[BROWSER] Browser connected - will reuse for all videos\n")

        if resume_folder:
            # Resume the specified video first, then continue
            await resume_video(resume_folder, channel_manager, continue_after=not single_video, client=client)
        else:
            # Round-robin through all channels (starting from specific_channel if provided)
            # Topics come strictly from channel_topics.json
            await multi_channel_video_creation(channel_manager, client=client, specific_channel=specific_channel, single_video=single_video)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("\n[BROWSER] Closing browser session...")
        await client.disconnect()


async def generate_media(video_folder: Path, channel: Channel, client: HiggsfieldClient = None):
    """Generate images for a video project using the pipeline.

    Args:
        video_folder: Path to the video folder
        channel: The Channel this video belongs to
        client: Optional existing HiggsfieldClient to reuse browser connection
    """
    state = StateManager(video_folder)
    files = FileManager(video_folder)

    # Load prompts
    prompts = files.load_prompts()
    if not prompts:
        print("ERROR: No prompts found!")
        return

    print("\n[PHASE 3] Generating Images with Higgsfield Pipeline")
    print("-" * 40)
    print(f"Channel: {channel.name}")
    print(f"Total images needed: {len(prompts)}")
    print(f"Max concurrent: {config.MAX_CONCURRENT_IMAGES}")

    # Use existing client or create new one
    owns_client = client is None
    if owns_client:
        client = HiggsfieldClient(state, files.video_folder)

    processor = HiggsfieldBatchProcessor(client, state, files)

    try:
        # Only connect if we created the client (or it's not connected)
        if owns_client or not client.context:
            await client.connect()

        # Run the image pipeline
        state.set_status(VideoState.IMAGES_IN_PROGRESS)
        await processor.generate_all(prompts, lambda s: print(s))

        # Final status check
        if state.is_complete():
            state.set_status(VideoState.COMPLETED)
            print("\n" + "=" * 60)
            print(f"VIDEO CREATION COMPLETE FOR: {channel.name}")
            print("=" * 60)
            print(f"\nOutput folder: {video_folder}")
            print("\nFiles created:")
            print(f"  - script.txt (voiceover script)")
            print(f"  - titles.txt (YouTube title ideas)")
            print(f"  - description.txt (YouTube description)")
            print(f"  - images/image001.png through image{len(prompts):03d}.png")
        else:
            remaining_images = state.get_pending_images()
            if remaining_images:
                print(f"\nWarning: {len(remaining_images)} images still pending.")
            print(f"Re-run with --resume {channel.id}/{video_folder.name} to continue.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print(f"Resume with: python main.py --resume {channel.id}/{video_folder.name}")
    except Exception as e:
        print(f"\nError: {e}")
        state.set_status(VideoState.FAILED)
        raise
    finally:
        # Only disconnect if we created the client
        if owns_client:
            await client.disconnect()


def list_videos(channel_id: str = None):
    """List all video projects, optionally filtered by channel."""
    print("\nExisting Video Projects:")
    print("-" * 60)

    videos = list_video_folders(channel_id)
    if not videos:
        if channel_id:
            print(f"No videos found for channel: {channel_id}")
        else:
            print("No videos found in output folder.")
        return

    # Group by channel
    by_channel = {}
    for v in videos:
        ch = v.get('channel_id', 'unknown')
        if ch not in by_channel:
            by_channel[ch] = []
        by_channel[ch].append(v)

    for channel_id, channel_videos in by_channel.items():
        print(f"\n[{channel_id}]")
        for v in channel_videos:
            folder_name = Path(v['folder']).name
            topic = v.get('topic', 'No topic')[:40]
            status = v.get('status', 'unknown')
            images = v.get('image_count', 0)
            print(f"  {folder_name}: {topic}...")
            print(f"           Status: {status}, Images: {images}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Creator Bot - Multi-Channel Support"
    )
    parser.add_argument(
        "--channel", "-c",
        type=str,
        default="all",
        help="Channel to generate for: channel_id or 'all' for round-robin (default: all)"
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        nargs="?",
        const="auto",
        help="Resume a previous video (specify channel_id/videoX or 'auto' for most recent incomplete)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all video projects"
    )
    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="List all configured channels"
    )
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="Only generate script and prompts, skip image generation"
    )
    parser.add_argument(
        "--single-video",
        action="store_true",
        help="Create only one video and exit (don't continue round-robin)"
    )

    args = parser.parse_args()

    print_banner()

    # Initialize channel manager
    channel_manager = ChannelManager()

    # Handle list-channels command
    if args.list_channels:
        channel_manager.list_channels()
        return

    # Check API key
    if not check_api_key():
        sys.exit(1)

    # Handle list command
    if args.list:
        # Filter by channel if specified and not 'all'
        channel_filter = args.channel if args.channel != "all" else None
        list_videos(channel_filter)
        return

    # Handle resume
    if args.resume:
        if args.resume == "auto":
            # Find resumable video across all channels (or specific channel if specified)
            channel_filter = args.channel if args.channel != "all" else None
            video_folder = find_resumable_video(channel_filter)
            if not video_folder:
                print("No incomplete videos found to resume.")
                print("Starting new video instead...")
                asyncio.run(run_continuous_with_persistent_browser(channel_manager, args.channel, single_video=args.single_video))
                return
        else:
            # Parse channel_id/videoX format
            if "/" in args.resume:
                channel_id, video_name = args.resume.split("/", 1)
                video_folder = config.OUTPUT_DIR / channel_id / video_name
            else:
                # Legacy format: just videoX - search in all channels
                video_folder = None
                for channel_dir in config.OUTPUT_DIR.iterdir():
                    if channel_dir.is_dir():
                        potential = channel_dir / args.resume
                        if potential.exists():
                            video_folder = potential
                            break

            if not video_folder or not video_folder.exists():
                print(f"Video folder not found: {args.resume}")
                print("Use format: channel_id/videoX (e.g., history_epoch/video1)")
                sys.exit(1)

        # Resume the video and continue creating new ones after completion
        asyncio.run(run_continuous_with_persistent_browser(channel_manager, args.channel, video_folder, single_video=args.single_video))
        return

    # Create new video
    if args.script_only:
        # Only generate script and prompts
        # Get channel(s) to use
        if args.channel == "all":
            channels = channel_manager.get_all_channels()
            if not channels:
                print("ERROR: No channels configured. Please check channels.json")
                sys.exit(1)
            channel = channels[0]  # Use first channel for script-only mode
            print(f"Using first channel: {channel.name}")
        else:
            channel = channel_manager.get_channel(args.channel)
            if not channel:
                print(f"ERROR: Channel '{args.channel}' not found in channels.json")
                print(f"Available channels: {', '.join(channel_manager.get_channel_ids())}")
                sys.exit(1)

        print(f"Generating script and prompts only for {channel.name} (no images)...")

        script_gen = ScriptGenerator(channel)

        # Get topic from channel_topics.json (no AI generation)
        print("Getting topic from channel_topics.json...")
        result = script_gen.get_topic()
        if result is None:
            print(f"No topics available for {channel.name} in channel_topics.json.")
            print("Add more topics to channel_topics.json and try again.")
            sys.exit(1)
        topic, video_id = result
        print(f"Topic: {topic}")

        # Create video folder using the video_id from channel_topics.json
        video_folder = create_new_video_folder(channel, video_id=video_id)
        state = StateManager(video_folder)
        files = FileManager(video_folder)

        # Save topic
        files.save_topic(topic)
        state.set_topic(topic)

        # Generate and save script immediately (user can start reading now!)
        print("Generating script...")
        script = script_gen.generate_script(topic)
        word_count = len(script.split())
        duration = script_gen.estimate_duration_minutes(script)
        files.save_script(script)
        print(f"Script generated: {word_count} words (~{duration:.1f} minutes)")

        # Generate and save titles
        print("Generating titles...")
        titles = script_gen.generate_titles(topic, script)
        files.save_titles(titles)

        # Generate and save description
        print("Generating description...")
        description = script_gen.generate_description(topic, script)
        files.save_description(description)

        # Generate and save prompts
        print("Generating image prompts...")
        prompt_gen = PromptGenerator(channel)
        prompts = prompt_gen.generate_all_prompts(topic, script)
        files.save_prompts(prompts)

        state.set_script_info(word_count, duration, len(prompts))
        state.set_status(VideoState.PROMPTS_GENERATED)

        print(f"\nScript and prompts saved to: {video_folder}")
        print("Run without --script-only to generate images.")
    else:
        # Run continuous video creation with persistent browser
        # Topics come strictly from channel_topics.json
        asyncio.run(run_continuous_with_persistent_browser(channel_manager, args.channel, single_video=args.single_video))


if __name__ == "__main__":
    main()
