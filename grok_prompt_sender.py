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
import time
from playwright.sync_api import sync_playwright


# ── Configuration ──────────────────────────────────────────────────────────────

GROK_IMAGINE_URL = "https://grok.com/imagine"
DEFAULT_DELAY = 6       # Seconds between prompts
DEFAULT_PROMPTS = "prompts.json"

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
SUBMIT_BUTTON_SELECTOR = 'button[aria-label="Submit"]'


# ── JS: Focus and clear the TipTap editor ─────────────────────────────────────

FOCUS_AND_CLEAR_JS = """
(selector) => {
    const el = document.querySelector(selector);
    if (!el) return false;

    // Focus the contenteditable div
    el.focus();

    // Select all content and delete it
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(el);
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand('delete');

    // Ensure it's truly empty
    el.innerHTML = '<p></p>';
    el.focus();

    // Place cursor inside the <p>
    const p = el.querySelector('p');
    if (p) {
        const newRange = document.createRange();
        newRange.selectNodeContents(p);
        newRange.collapse(false);
        selection.removeAllRanges();
        selection.addRange(newRange);
    }

    return true;
}
"""

# ── JS: Click the Submit button ───────────────────────────────────────────────

CLICK_SUBMIT_JS = """
(selector) => {
    const btn = document.querySelector(selector);
    if (!btn) return { clicked: false, reason: 'not found' };
    if (btn.disabled) return { clicked: false, reason: 'disabled' };
    btn.click();
    return { clicked: true, reason: 'ok' };
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

    total = len(prompts)
    print(f"  Found {total} prompts to send (starting from ID {prompts[0]['id']})")

    # ── 2. Launch/Connect Browser ──────────────────────────────────────────
    debug_port = launch_chrome_debug(args.profile)

    with sync_playwright() as pw:
        print("  Connecting to Chrome...")
        browser = pw.chromium.connect_over_cdp(f"http://localhost:{debug_port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

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

        # Check submit button
        submit_check = page.evaluate(CHECK_INPUT_JS, SUBMIT_BUTTON_SELECTOR)
        if submit_check and submit_check.get('found'):
            print(f"  ✓ Found Submit button")
        else:
            print("  ⚠ Submit button not found — will use Enter key fallback")

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
                # Step A: Focus and clear the TipTap editor
                cleared = page.evaluate(FOCUS_AND_CLEAR_JS, INPUT_SELECTOR)
                if not cleared:
                    print(f"    ✗ Could not focus input — skipping")
                    fail_count += 1
                    continue

                time.sleep(0.3)

                # Step B: Paste the prompt via clipboard (instant)
                import pyperclip
                pyperclip.copy(prompt_text)
                page.keyboard.press("Control+v")

                time.sleep(0.5)

                # Step C: Click the Submit button
                click_result = page.evaluate(CLICK_SUBMIT_JS, SUBMIT_BUTTON_SELECTOR)
                if not click_result or not click_result.get('clicked'):
                    reason = click_result.get('reason', 'unknown') if click_result else 'no result'
                    print(f"    ⚠ Submit button issue ({reason}) — trying Enter key")
                    page.keyboard.press("Enter")

                success_count += 1
                print(f"    ✓ Sent!")

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


if __name__ == "__main__":
    main()