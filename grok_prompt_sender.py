"""
Grok Imagine Prompt Sender
──────────────────────────
Automates submitting video generation prompts to Grok Imagine by connecting
to an existing authenticated Chrome instance via Playwright CDP.

Flow:
  1. Launches/connects to Chrome with remote debugging (reuses login session)
  2. Navigates to https://grok.com/imagine
  3. Loads prompts from a JSON file
  4. Types each prompt into the TipTap editor and clicks Submit
  5. Waits a configurable delay between prompts (default 5s)

Usage:
    python grok_prompt_sender.py "Arctic Survival/video25" -1
    python grok_prompt_sender.py "Arctic Survival/video25" -1 --prompts prompts2.json
    python grok_prompt_sender.py "Arctic Survival/video25" -4 --start 10 --delay 8
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright


# ── Configuration ──────────────────────────────────────────────────────────────

GROK_IMAGINE_URL = "https://grok.com/imagine"
DEFAULT_DELAY = 6       # Seconds between prompts
DEFAULT_PROMPTS = "prompts.json"
DOWNLOAD_DELAY = 80     # Seconds to wait before starting download script

# ── Selectors (from live Grok Imagine page inspection) ─────────────────────────
#
# Text input: TipTap/ProseMirror contenteditable div
#   <div class="tiptap ProseMirror ..." contenteditable="true">
#
# Submit button:
#   <button aria-label="Submit">  (40×40px, SVG icon)
#
# Other controls (for reference):
#   Image/Video toggle:  buttons with text "Image" / "Video"
#   Aspect ratio:        button aria-label="Aspect Ratio" (text shows "16:9")
#   Resolution:          buttons with text "480p" / "720p"
#   Duration:            buttons with text "6s" / "10s"

INPUT_SELECTOR = 'div.tiptap.ProseMirror[contenteditable="true"]'


# ── JS: Clear editor and insert text atomically ──────────────────────────────

CLEAR_AND_INSERT_JS = """
({ inputSelector, text }) => {
    const el = document.querySelector(inputSelector);
    if (!el) return { success: false, reason: 'input not found' };

    // Focus, select all, and replace with new text in one shot
    el.focus();
    document.execCommand('selectAll');
    document.execCommand('insertText', false, text);

    return { success: true };
}
"""

# ── JS: Check if input exists ─────────────────────────────────────────────────

CHECK_INPUT_JS = """
(selector) => {
    const el = document.querySelector(selector);
    if (!el) return { found: false };
    const rect = el.getBoundingClientRect();
    return {
        found: true,
        visible: rect.width > 0 && rect.height > 0,
        width: Math.round(rect.width),
        height: Math.round(rect.height),
    };
}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_chrome() -> str:
    """Locate the Chrome executable on Windows."""
    common_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return "chrome.exe"


def launch_chrome_debug(profile_num=1):
    """Launch Chrome with remote debugging if not already running."""
    chrome_path = _find_chrome()

    user_data_dir = os.path.join(os.getcwd(), f"browser_data_downloader_{profile_num}")
    os.makedirs(user_data_dir, exist_ok=True)

    debug_port = 9333 + profile_num - 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', debug_port))
    sock.close()

    if result == 0:
        print(f"  Chrome profile {profile_num} already running on port {debug_port}")
        return debug_port

    print(f"  Launching Chrome (Profile {profile_num}) on port {debug_port}...")
    print(f"  Profile path: {user_data_dir}")

    subprocess.Popen([
        chrome_path,
        f"--remote-debugging-port={debug_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
    ])

    time.sleep(4)
    return debug_port


def load_prompts(video_path: str, prompts_file: str) -> list:
    """
    Load prompts from a JSON file.
    Supports both formats:
      - List of {id, prompt} objects
      - Object with "prompts" key containing such a list
    """
    prompts_path = os.path.join("output", video_path, prompts_file)
    if not os.path.exists(prompts_path):
        print(f"  Error: Prompts file not found at {prompts_path}")
        return []

    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("prompts", [])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit prompts to Grok Imagine for video generation"
    )
    parser.add_argument(
        "video_name",
        help="Relative path to video folder (e.g. 'Arctic Survival/video25')"
    )
    parser.add_argument(
        "--prompts", default=DEFAULT_PROMPTS,
        help=f"Name of prompts JSON file (default: {DEFAULT_PROMPTS})"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds to wait between prompts (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--download-delay", type=int, default=DOWNLOAD_DELAY,
        help=f"Seconds to wait before starting download script (default: {DOWNLOAD_DELAY})"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Automatically start the download script after finishing"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Prompt ID to start from (skip earlier prompts, default: 1)"
    )

    # Browser profile flags (same as download_clips_grok.py)
    parser.add_argument("-1", dest="profile", action="store_const", const=1, help="Use browser profile 1 (default)")
    parser.add_argument("-2", dest="profile", action="store_const", const=2, help="Use browser profile 2")
    parser.add_argument("-3", dest="profile", action="store_const", const=3, help="Use browser profile 3")
    parser.add_argument("-4", dest="profile", action="store_const", const=4, help="Use browser profile 4")
    parser.set_defaults(profile=1)

    args = parser.parse_args()

    # ── 1. Load Prompts ────────────────────────────────────────────────────
    print(f"\n  Loading prompts from {args.prompts} for {args.video_name}...")
    prompts = load_prompts(args.video_name, args.prompts)
    if not prompts:
        print("  No prompts found. Exiting.")
        return

    # Filter by --start
    prompts = [p for p in prompts if p['id'] >= args.start]
    if not prompts:
        print(f"  No prompts with id >= {args.start}. Exiting.")
        return

    # Skip prompts that already have a downloaded clip
    clips_dir = os.path.join("output", args.video_name, "clips")
    before_skip = len(prompts)
    prompts = [p for p in prompts if not os.path.exists(os.path.join(clips_dir, f"clip_{p['id']:03d}.mp4"))]
    skipped = before_skip - len(prompts)
    if skipped:
        print(f"  Skipped {skipped} prompts (clips already exist in {clips_dir})")
    if not prompts:
        print("  All clips already downloaded. Nothing to send.")
        return

    total = len(prompts)
    print(f"  Found {total} prompts to send (starting from ID {prompts[0]['id']})")

    # ── 2. Launch/Connect Browser ──────────────────────────────────────────
    debug_port = launch_chrome_debug(args.profile)

    with sync_playwright() as pw:
        print("  Connecting to Chrome...")
        browser = pw.chromium.connect_over_cdp(f"http://localhost:{debug_port}")
        context = browser.contexts[0]
        
        # Try to find an existing Grok page to reuse
        page = None
        for p in context.pages:
            try:
                if p.url and "grok.com" in p.url:
                    page = p
                    break
            except Exception:
                pass
        
        # If no Grok page, try to find an empty new tab
        if not page:
            for p in context.pages:
                try:
                    if p.url in ["about:blank", "chrome://newtab/", "edge://newtab/"]:
                        page = p
                        break
                except Exception:
                    pass
        
        # Otherwise, create a new page
        if not page:
            page = context.new_page()

        try:
            page.bring_to_front()
        except Exception as e:
            print(f"  Warning: Could not bring page to front: {e}")

        # ── 3. Wait for browser/extensions to initialize, then navigate ──
        print("  Waiting 10s for browser to initialize...")
        time.sleep(10)
        print(f"  Navigating to {GROK_IMAGINE_URL}...")
        page.goto(GROK_IMAGINE_URL)
        page.wait_for_load_state("domcontentloaded")
        time.sleep(5)  # Wait for Cloudflare + page render

        # ── 4. Verify the input is present ─────────────────────────────────
        print("  Checking for prompt input...")
        input_check = page.evaluate(CHECK_INPUT_JS, INPUT_SELECTOR)

        if not input_check or not input_check.get('found'):
            print("  WARNING: Prompt input not found!")
            print("  The page might be showing a Cloudflare challenge.")
            print("  Please solve it in the browser, then press Enter here.")
            input("  Press Enter when the page is ready...")

            # Wait a moment, then retry
            time.sleep(2)
            input_check = page.evaluate(CHECK_INPUT_JS, INPUT_SELECTOR)
            if not input_check or not input_check.get('found'):
                print("  Still cannot find input. Exiting.")
                return

        print(f"  ✓ Found TipTap editor ({input_check['width']}×{input_check['height']}px)")

        # ── 5. Submit prompts one by one ───────────────────────────────────
        print(f"\n  {'=' * 56}")
        print(f"  Grok Imagine Prompt Sender")
        print(f"  Sending {total} prompts | Delay: {args.delay}s")
        print(f"  Profile: {args.profile} | Port: {debug_port}")
        print(f"  {'=' * 56}\n")

        success_count = 0
        fail_count = 0

        for i, prompt_data in enumerate(prompts):
            pid = prompt_data['id']
            prompt_text = prompt_data['prompt']

            print(f"  [{i+1}/{total}] Prompt #{pid:03d}: {prompt_text[:70]}...")

            try:
                # Step 1: Clear editor and insert new prompt text (atomic JS call)
                result = page.evaluate(CLEAR_AND_INSERT_JS, {
                    "inputSelector": INPUT_SELECTOR,
                    "text": prompt_text,
                })

                if not result or not result.get('success'):
                    reason = result.get('reason', 'unknown') if result else 'no result'
                    print(f"    ✗ Failed ({reason}) — skipping")
                    fail_count += 1
                    continue

                # Step 2: Submit via Enter (CDP trusted event — React requires isTrusted=true)
                page.keyboard.press("Enter")

                success_count += 1
                print(f"    ✓ Sent!")

                # Extra wait after the very first prompt
                if i == 0:
                    print(f"    First prompt — extra 9s wait...")
                    time.sleep(9)

                # Step D: Wait between prompts (skip wait after last prompt)
                if i < total - 1:
                    print(f"    Waiting {args.delay}s...")
                    time.sleep(args.delay)

            except Exception as e:
                print(f"    ✗ Error: {e}")
                fail_count += 1
                time.sleep(2)

        # ── 6. Summary ─────────────────────────────────────────────────────
        print(f"\n  {'=' * 56}")
        print(f"  Done!")
        print(f"  Sent: {success_count}/{total}")
        if fail_count:
            print(f"  Failed: {fail_count}")
        print(f"  {'=' * 56}\n")

    # ── 7. Wait and Start Download ─────────────────────────────────────────
    if args.download:
        if success_count > 0:
            print(f"\n  Waiting {args.download_delay} seconds before starting the downloader...")
            time.sleep(args.download_delay)
        else:
            print("\n  No prompts sent, skipping delay.")
            
        print("  Starting download_clips_grok.py...")
        
        cmd = [
            sys.executable,
            "download_clips_grok.py",
            args.video_name,
            "--project", "1",
            f"-{args.profile}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  Downloader script exited with error: {e}")
        except KeyboardInterrupt:
            print("\n  Download interrupted by user.")


if __name__ == "__main__":
    main()