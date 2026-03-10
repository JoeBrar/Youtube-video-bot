"""
Grok Imagine Clip Downloader
─────────────────────────────
Downloads generated clips from a Grok Imagine project by matching prompt text
to a local prompts.json file.

Flow:
  1. Opens https://grok.com/imagine/favorites to establish auth session
  2. Calls Grok's REST API (with pagination) to fetch all clip data
  3. Matches clip prompts against local prompts.json
  4. Downloads each matched clip from public CDN or via Download button

Usage:
    python download_clips_grok.py "Arctic Survival/video24" --project 1 -1
    python download_clips_grok.py "Arctic Survival/video24" --project 1 -4
"""

import argparse
import json
import os
import time
import subprocess
import unicodedata
from playwright.sync_api import sync_playwright


# ── Configuration ──────────────────────────────────────────────────────────────
FAVORITES_URL = "https://grok.com/imagine/favorites"
API_URL = "https://grok.com/rest/media/post/list"
DOWNLOAD_BUTTON_SELECTOR = 'button[aria-label="Download"]'
VIDEO_SELECTOR = "video#sd-video"


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
    
    user_data_dir = os.path.join(os.getcwd(), f"browser_data_downloader_{profile_num}")
    os.makedirs(user_data_dir, exist_ok=True)
    
    debug_port = 9333 + profile_num - 1
    
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
    
    time.sleep(4)
    return debug_port


def normalize_text(text: str) -> str:
    """Normalize text for comparison: NFKC unicode, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.lower().split())


def match_prompt(scraped_text: str, local_prompts: dict[str, int]) -> tuple[str | None, int | None]:
    """
    Try to match scraped prompt text against local prompts.
    Uses only full-prompt matching to avoid mismatches between similar prompts.
    Returns (prompt_text, clip_id) or (None, None) if no match.
    """
    clean_scraped = scraped_text.strip().replace('\n', ' ')
    norm_scraped = normalize_text(clean_scraped)
    
    # Strategy 1: Exact match
    for p_text, p_id in local_prompts.items():
        if p_text == clean_scraped:
            return p_text, p_id
    
    # Strategy 2: Normalized exact match (lowercase, collapsed whitespace)
    for p_text, p_id in local_prompts.items():
        if normalize_text(p_text) == norm_scraped:
            return p_text, p_id
    
    return None, None


def fetch_all_project_clips(page, project_index: int) -> list[dict]:
    """
    Fetch ALL clips from a Grok Imagine project using the REST API.
    Uses paginated children endpoint to ensure all clips are returned.
    Returns a list of dicts with keys: 'id', 'prompt', 'mediaUrl'.
    """
    print("Fetching project data via Grok API...")
    
    # Step 1: Fetch project list to get the project ID
    api_response = page.evaluate("""async (apiUrl) => {
        try {
            const resp = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    limit: 200, 
                    filter: { source: 'MEDIA_POST_SOURCE_LIKED' } 
                })
            });
            if (!resp.ok) return { error: 'HTTP ' + resp.status, posts: [] };
            const data = await resp.json();
            return data;
        } catch (e) {
            return { error: e.message, posts: [] };
        }
    }""", API_URL)
    
    if not api_response or "error" in api_response:
        err = api_response.get('error', 'Unknown') if api_response else 'No response'
        print(f"API Error: {err}")
        return []
    
    posts = api_response.get("posts", [])
    if not posts:
        print("No projects found from API.")
        return []
    
    print(f"API returned {len(posts)} projects.")
    
    if project_index < 0 or project_index >= len(posts):
        print(f"Error: Project index {project_index + 1} out of range (found {len(posts)} projects)")
        return []
    
    project = posts[project_index]
    project_id = project.get("id", "unknown")
    print(f"Selected project: {project_id}")
    
    # Step 2: Get child posts from the initial response (may be incomplete)
    initial_children = project.get("childPosts", [])
    
    if not initial_children:
        # Single clip project
        prompt = project.get("prompt", "")
        media_url = project.get("mediaUrl", "")
        post_id = project.get("id", "")
        if prompt and post_id:
            return [{"id": post_id, "prompt": prompt, "mediaUrl": media_url}]
        print("No clips found in this project.")
        return []
    
    print(f"Initial response has {len(initial_children)} child clips.")
    
    # Step 3: Try paginated fetch to get ALL children
    # Use the children endpoint with cursor-based pagination
    print("Fetching ALL child clips with pagination...")
    all_children = page.evaluate("""async (args) => {
        const { apiUrl, projectId } = args;
        let allPosts = [];
        let cursor = null;
        let pageNum = 0;
        
        while (true) {
            pageNum++;
            try {
                const body = { 
                    limit: 200, 
                    parentPostId: projectId
                };
                if (cursor) body.cursor = cursor;
                
                const resp = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                if (!resp.ok) {
                    return { error: 'HTTP ' + resp.status, posts: allPosts, pages: pageNum };
                }
                const data = await resp.json();
                const posts = data.posts || [];
                allPosts = allPosts.concat(posts);
                
                // Check for more pages
                if (data.cursor && posts.length > 0) {
                    cursor = data.cursor;
                } else {
                    break;
                }
            } catch (e) {
                return { error: e.message, posts: allPosts, pages: pageNum };
            }
        }
        
        return { posts: allPosts, pages: pageNum };
    }""", {"apiUrl": API_URL, "projectId": project_id})
    
    paginated_children = []
    if all_children and not all_children.get("error"):
        paginated_children = all_children.get("posts", [])
        pages = all_children.get("pages", 0)
        print(f"Paginated fetch: {len(paginated_children)} clips across {pages} page(s).")
    else:
        err = all_children.get("error", "Unknown") if all_children else "No response"
        print(f"Paginated fetch failed ({err}), using initial child posts.")
    
    # Use whichever returned more clips
    child_posts = paginated_children if len(paginated_children) > len(initial_children) else initial_children
    print(f"Using {'paginated' if child_posts is paginated_children else 'initial'} data ({len(child_posts)} clips).")
    
    clips = []
    seen_ids = set()
    for cp in child_posts:
        clip_id = cp.get("id", "")
        if clip_id and clip_id not in seen_ids:
            seen_ids.add(clip_id)
            clips.append({
                "id": clip_id,
                "prompt": cp.get("prompt", ""),
                "mediaUrl": cp.get("mediaUrl", ""),
            })
    
    print(f"Found {len(clips)} unique clips in project.")
    return clips


def download_clip_direct(page, url: str, save_path: str) -> bool:
    """
    Download a clip using the browser's authenticated fetch API.
    This ensures all cookies (Cloudflare clearance, auth tokens) are sent,
    preventing 403 errors on successive downloads.
    """
    import base64

    download_url = f"{url}?cache=1&dl=1" if '?' not in url else url
    
    for attempt in range(3):
        try:
            result = page.evaluate("""async (url) => {
                try {
                    const resp = await fetch(url, { credentials: 'include' });
                    if (!resp.ok) return { error: resp.status };
                    const blob = await resp.blob();
                    const buffer = await blob.arrayBuffer();
                    const bytes = new Uint8Array(buffer);
                    let binary = '';
                    const chunkSize = 8192;
                    for (let i = 0; i < bytes.length; i += chunkSize) {
                        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
                    }
                    return { data: btoa(binary), size: bytes.length };
                } catch (e) {
                    return { error: e.message };
                }
            }""", download_url)
            
            if not result or "error" in result:
                err = result.get("error", "Unknown") if result else "No response"
                if isinstance(err, int) and err in (520, 429, 503) and attempt < 2:
                    time.sleep(2)
                    continue
                print(f"      ! Direct download HTTP {err}")
                return False
            
            data = base64.b64decode(result["data"])
            if len(data) > 500 * 1024:
                with open(save_path, 'wb') as f:
                    f.write(data)
                return True
            else:
                print(f"      ! File too small ({len(data)} bytes)")
                return False
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"      ! Direct download failed: {e}")
            return False
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Grok Imagine clips")
    parser.add_argument("video_name", help="Relative path to video folder (e.g. 'history_locale/video1')")
    parser.add_argument("--project", type=int, default=1, help="Index of the Grok Imagine project to open (1-based, default: 1)")
    parser.add_argument("-1", dest="profile", action="store_const", const=1, help="Use browser profile 1 (default)")
    parser.add_argument("-2", dest="profile", action="store_const", const=2, help="Use browser profile 2")
    parser.add_argument("-3", dest="profile", action="store_const", const=3, help="Use browser profile 3")
    parser.add_argument("-4", dest="profile", action="store_const", const=4, help="Use browser profile 4")
    parser.set_defaults(profile=1)
    args = parser.parse_args()

    # 1. Load Prompts
    print(f"Loading prompts for {args.video_name}...")
    local_prompts = load_prompts(args.video_name)
    if not local_prompts:
        return
    
    print(f"Loaded {len(local_prompts)} prompts from local JSON.")

    # 2. Setup Output Dir
    output_dir = os.path.join("output", args.video_name, "clips")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Launch/Connect Browser
    debug_port = launch_chrome_debug(args.profile)
    
    with sync_playwright() as pw:
        print("Connecting to Chrome...")
        browser = pw.chromium.connect_over_cdp(f"http://localhost:{debug_port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

        # Navigate to favorites to establish auth session
        print(f"Navigating to {FAVORITES_URL}...")
        page.goto(FAVORITES_URL)
        page.wait_for_load_state("domcontentloaded")
        time.sleep(3)

        # ── Step A: Fetch all clip data via API ────────────────────────────
        project_clips = fetch_all_project_clips(page, args.project - 1)
        
        if not project_clips:
            print("No clips to download.")
            return
        
        # Build a map of {normalized_prompt: [clip_info, ...]} to handle duplicates
        clip_map = {}
        for clip in project_clips:
            norm_key = normalize_text(clip["prompt"].strip().replace('\n', ' '))
            if norm_key not in clip_map:
                clip_map[norm_key] = []
            clip_map[norm_key].append(clip)
        
        # ── Diagnostic: dump API prompts for comparison ──────────────────
        debug_path = os.path.join("output", args.video_name, "api_prompts_debug.json")
        api_debug = []
        for i, clip in enumerate(project_clips):
            api_debug.append({
                "index": i + 1,
                "id": clip["id"],
                "prompt": clip["prompt"],
                "mediaUrl": clip.get("mediaUrl", "")[:80],
            })
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(api_debug, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(api_debug)} API prompts to {debug_path}")
        print(f"  API clips: {len(project_clips)}, Unique normalized keys: {len(clip_map)}, Local prompts: {len(local_prompts)}")
        
        # Show which local prompts have no match in API
        unmatched_local = []
        for p_text, p_id in sorted(local_prompts.items(), key=lambda x: x[1]):
            norm_local = normalize_text(p_text.strip().replace('\n', ' '))
            if norm_local not in clip_map:
                unmatched_local.append(p_id)
        if unmatched_local:
            print(f"  Local prompts with NO API match: {unmatched_local}")
        
        # Show which API prompts have no match in local
        local_norm_set = set()
        for p_text in local_prompts:
            local_norm_set.add(normalize_text(p_text.strip().replace('\n', ' ')))
        unmatched_api = []
        for clip in project_clips:
            norm_key = normalize_text(clip["prompt"].strip().replace('\n', ' '))
            if norm_key not in local_norm_set:
                unmatched_api.append(clip["prompt"][:80])
        if unmatched_api:
            print(f"  API prompts with NO local match ({len(unmatched_api)}):")
            for p in unmatched_api:
                print(f"    - {p}...")
        print()
        
        # Show sample data
        print(f"--- Sample clips ---")
        for i, clip in enumerate(project_clips[:3]):
            print(f"  Clip {i+1}: {clip['prompt'][:60]}...")
            print(f"    ID: {clip['id']}")
            print(f"    URL: {clip['mediaUrl'][:80] if clip['mediaUrl'] else 'N/A'}")
        print()

        # ── Step B: Match & Download ───────────────────────────────────────
        print("Matching against local prompts...\n")
        
        download_count = 0
        skip_count = 0
        missing_count = 0
        
        sorted_prompts = sorted(local_prompts.items(), key=lambda x: x[1])
        used_clips = set()

        for p_text, p_id in sorted_prompts:
            filename = f"clip_{p_id:03d}.mp4"
            save_path = os.path.join(output_dir, filename)

            if os.path.exists(save_path):
                skip_count += 1
                continue

            # Find matching clip from API data using normalized key
            matched_clip = None
            norm_local = normalize_text(p_text.strip().replace('\n', ' '))
            
            if norm_local in clip_map:
                # Direct normalized match found
                for clip_info in clip_map[norm_local]:
                    if clip_info["id"] not in used_clips:
                        matched_clip = clip_info
                        used_clips.add(clip_info["id"])
                        break
            
            if matched_clip:
                media_url = matched_clip.get("mediaUrl", "")
                print(f"[{p_id:03d}] Downloading...")
                
                success = False
                
                # Strategy 1: Direct HTTP download (fast, no browser needed)
                if media_url:
                    success = download_clip_direct(page, media_url, save_path)
                
                # Strategy 2: Fallback to browser Download button
                if not success:
                    print(f"      Falling back to browser download...")
                    try:
                        post_url = f"https://grok.com/imagine/post/{matched_clip['id']}"
                        page.goto(post_url)
                        download_btn = page.locator(DOWNLOAD_BUTTON_SELECTOR).first
                        download_btn.wait_for(timeout=20000)
                        
                        with page.expect_download(timeout=60000) as download_info:
                            download_btn.click()
                        
                        download = download_info.value
                        download.save_as(save_path)
                        
                        if os.path.exists(save_path) and os.path.getsize(save_path) > 500 * 1024:
                            success = True
                        else:
                            if os.path.exists(save_path):
                                os.remove(save_path)
                    except Exception as e:
                        print(f"      ! Browser download failed: {e}")
                
                if success:
                    size_mb = os.path.getsize(save_path) / (1024 * 1024)
                    print(f"      > Saved {filename} ({size_mb:.1f} MB)")
                    download_count += 1
            else:
                # Diagnostic: find closest API prompt to show where text diverges
                best_match_len = 0
                best_api_prompt = None
                for api_norm_key in clip_map:
                    # Find longest common prefix
                    common = 0
                    for a, b in zip(norm_local, api_norm_key):
                        if a == b:
                            common += 1
                        else:
                            break
                    if common > best_match_len:
                        best_match_len = common
                        best_api_prompt = api_norm_key
                
                print(f"[{p_id:03d}] ! Not found in project")
                if best_api_prompt and best_match_len > 20:
                    diverge_start = max(0, best_match_len - 10)
                    diverge_end = best_match_len + 40
                    print(f"      Closest API match ({best_match_len} chars shared):")
                    print(f"        LOCAL: ...{norm_local[diverge_start:diverge_end]}...")
                    print(f"        API  : ...{best_api_prompt[diverge_start:diverge_end]}...")
                    print(f"        LOCAL len={len(norm_local)}, API len={len(best_api_prompt)}")
                missing_count += 1
        
    print(f"\nSummary:")
    print(f"  Downloaded: {download_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Missing (not found): {missing_count}")
    print(f"  Total clips in project: {len(project_clips)}")
    print(f"  Total local prompts: {len(local_prompts)}")
    print("Done.")


if __name__ == "__main__":
    main()
