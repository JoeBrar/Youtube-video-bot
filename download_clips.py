"""
VEO 3.1 Clip Downloader
───────────────────────
Downloads generated clips from a Google Flow project URL by matching prompt text
to a local prompts.json file.

Usage:
    python download_clips.py "https://labs.google/fx/tools/flow/project/..." "Arctic Survival/video11"
"""

import argparse
import json
import os
import time
import urllib.request
import subprocess
from playwright.sync_api import sync_playwright, Page

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_SCROLL_STEPS = 300         # Maximum number of scroll attempts down the page
LAZY_LOAD_WAIT_SECONDS = 4     # Seconds to wait at the bottom for new clips to load

# ── Selectors & JS ─────────────────────────────────────────────────────────────

# Reuse robust JS from veo_auto.py for virtual scroll handling
FIND_SCROLL_CONTAINER_JS = """
    function findScrollContainer() {
        const candidates = document.querySelectorAll('div');
        for (const d of candidates) {
            const style = window.getComputedStyle(d);
            const hasOverflow = style.overflowY === 'auto' || style.overflowY === 'scroll'
                             || style.overflow === 'auto' || style.overflow === 'scroll';
            if (hasOverflow && d.querySelector('video') && d.scrollHeight > d.clientHeight + 10) {
                return d;
            }
        }
        const firstVideo = document.querySelector('video');
        if (!firstVideo) return null;
        let el = firstVideo.parentElement;
        while (el && el !== document.body) {
            if (el.scrollHeight > el.clientHeight + 10) return el;
            el = el.parentElement;
        }
        return null;
    }
"""

SCAN_ALL_CLIPS_JS = f"""
async () => {{
    {FIND_SCROLL_CONTAINER_JS}

    const container = findScrollContainer();
    if (!container) return {{}};

    const results = {{}};
    const maxScrollSteps = {MAX_SCROLL_STEPS}; // Allow extensive scrolling
    
    // 1. Reset to top
    container.scrollTop = 0;
    await new Promise(r => requestAnimationFrame(() => setTimeout(r, 300)));

    for (let step = 0; step < maxScrollSteps; step++) {{
        // 2. Scan visible videos
        const videos = container.querySelectorAll('video');
        for (const v of videos) {{
            let current = v.parentElement;
            let text = "";
            while (current && current.parentElement && current.parentElement.tagName !== 'BODY') {{
                const textNodes = Array.from(current.querySelectorAll('*')).filter(el => {{
                    return el.children.length === 0 && el.textContent.trim().length > 30;
                }});
                
                if (textNodes.length > 0) {{
                    let longest = "";
                    for (let node of textNodes) {{
                        if (node.textContent.trim().length > longest.length) {{
                            longest = node.textContent.trim();
                        }}
                    }}
                    if (longest.length > 50) {{
                        text = longest;
                        break;
                    }}
                }}
                current = current.parentElement;
            }}
            
            if (text && !results[text]) {{
                results[text] = v.src || v.currentSrc;
            }}
        }}

        // 3. Scroll down
        const prevScroll = container.scrollTop;
        container.scrollTop += container.clientHeight * 0.8;
        
        // 4. Check if we reached bottom
        if (Math.abs(container.scrollTop - prevScroll) < 5) {{
            // Wait for lazy loading to trigger and load more clips
            await new Promise(r => setTimeout(r, {LAZY_LOAD_WAIT_SECONDS * 1000}));
            
            // Try scrolling again after the wait
            const retryScroll = container.scrollTop;
            container.scrollTop += container.clientHeight * 0.8;
            
            if (Math.abs(container.scrollTop - retryScroll) < 5) {{
                break; // Still at bottom after wait, definitely the end
            }}
        }}
        
        // 5. Wait for render
        await new Promise(r => requestAnimationFrame(() => setTimeout(r, 300)));
    }}

    return results;
}}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_prompts(video_path: str) -> dict[str, int]:
    """
    Load prompts from json and return a map of {prompt_text: clip_id}.
    """
    prompts_path = os.path.join("output", video_path, "prompts.json")
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file not found at {prompts_path}")
        return {}
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            prompts = data
        else:
            prompts = data.get("prompts", [])
        
    return {p['prompt']: p['id'] for p in prompts}

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
    
    # Use a separate profile for the downloader to avoid conflicts with main bot
    # and to keep login session persistent without affecting main Chrome.
    user_data_dir = os.path.join(os.getcwd(), f"browser_data_downloader_{profile_num}")
    os.makedirs(user_data_dir, exist_ok=True)
    
    debug_port = 9333 + profile_num - 1 # Different port than veo_auto.py (9222)
    
    # Check if port is already in use (simple check)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', debug_port))
    sock.close()
    
    if result == 0:
        print(f"  Chrome downloader profile already running on port {debug_port}")
        return debug_port

    print(f"  Launching Chrome (Downloader Profile) on port {debug_port}...")
    print(f"  Profile path: {user_data_dir}")
    
    subprocess.Popen([
        chrome_path,
        f"--remote-debugging-port={debug_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
    ])
    
    # Give it time to launch
    time.sleep(4)
    return debug_port

def download_file(url: str, save_path: str) -> bool:
    try:
        urllib.request.urlretrieve(url, save_path)
        # Verify size
        if os.path.getsize(save_path) < 500 * 1024: # < 500KB
            os.remove(save_path)
            return False
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download VEO 3.1 clips from project URL")
    parser.add_argument("project_url", help="URL of the Google Flow project")
    parser.add_argument("video_name", help="Relative path to video folder (e.g. 'Arctic Survival/video11')")
    parser.add_argument("-1", dest="profile", action="store_const", const=1, help="Use browser profile 1 (default)")
    parser.add_argument("-2", dest="profile", action="store_const", const=2, help="Use browser profile 2")
    parser.add_argument("-3", dest="profile", action="store_const", const=3, help="Use browser profile 3")
    parser.add_argument("-4", dest="profile", action="store_const", const=4, help="Use browser profile 4")
    parser.set_defaults(profile=1)
    args = parser.parse_args()

    # 1. Load Prompts
    print(f"Loading prompts for {args.video_name}...")
    local_prompts = load_prompts(args.video_name) # {text: id}
    if not local_prompts:
        return
    
    print(f"Loaded {len(local_prompts)} prompts from local JSON.")

    # 2. Setup Output Dir
    output_dir = os.path.join("output", args.video_name, "clips")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Launch/Connect Browser
    debug_port = launch_chrome_debug(args.profile)
    
    with sync_playwright() as p:
        print("Connecting to Chrome...")
        browser = p.chromium.connect_over_cdp(f"http://localhost:{debug_port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

        print(f"Navigating to {args.project_url}...")
        if page.url != args.project_url:
            page.goto(args.project_url)
            page.wait_for_load_state("networkidle")
            time.sleep(3)
        else:
            print("Already on project page.")

        # 4. Scan Entire Page
        print("Scanning ALL clips on page (this may take a moment)...")
        found_clips = page.evaluate(SCAN_ALL_CLIPS_JS) # { text: url }
        print(f"Found {len(found_clips)} total generated clips on page.")

        # 5. Match & Download
        print("Matching against local prompts...")
        
        download_count = 0
        skip_count = 0
        missing_count = 0

        # Sort by ID for cleaner output
        sorted_prompts = sorted(local_prompts.items(), key=lambda x: x[1])

        for p_text, p_id in sorted_prompts:
            filename = f"clip_{p_id:03d}.mp4"
            save_path = os.path.join(output_dir, filename)

            # Skip if already exists
            if os.path.exists(save_path):
                # print(f"[{p_id:03d}] Skipping (already downloaded)")
                skip_count += 1
                continue

            # Check if found on page
            video_url = found_clips.get(p_text)
            
            if video_url:
                print(f"[{p_id:03d}] Downloading...")
                # Download using playwright with cookies
                try:
                    response = context.request.get(video_url)
                    if response.ok:
                        body = response.body()
                        if len(body) > 500 * 1024: # > 500KB
                            with open(save_path, "wb") as f:
                                f.write(body)
                            print(f"      > Saved {filename}")
                            download_count += 1
                        else:
                            print(f"      ! Download failed for {filename}: File too small ({len(body)} bytes)")
                    else:
                        print(f"      ! Download failed for {filename}: HTTP {response.status}")
                except Exception as e:
                    print(f"      ! Download exception: {e}")
            else:
                print(f"[{p_id:03d}] ! Not found on page")
                missing_count += 1
        
    print(f"\nSummary:")
    print(f"  Downloaded: {download_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Missing (not generated): {missing_count}")
    print("Done.")

if __name__ == "__main__":
    main()
