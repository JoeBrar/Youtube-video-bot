"""
Higgsfield Client Module
Browser automation for generating images on higgsfield.ai

Pipeline Architecture:
- Single tab for image generation
- Up to 4 concurrent image generations
- Images complete in any order, but download sequentially
"""
import asyncio
import time
import re
from pathlib import Path
from typing import Optional, Callable, Dict, List, Set
from urllib.parse import urlparse, parse_qs, unquote
import sys
import os
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from playwright.async_api import async_playwright, Browser, BrowserContext, Page


class HiggsfieldClient:
    """Low-level browser automation for Higgsfield."""

    def __init__(self, state_manager=None, download_dir: Path = None):
        self.context: Optional[BrowserContext] = None
        self.state_manager = state_manager

        # Folders
        self.video_folder = Path(download_dir) if download_dir else Path("./downloads")
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.images_folder = self.video_folder / "images"
        self.images_folder.mkdir(parents=True, exist_ok=True)

        # Browser profile
        self.automation_profile_dir = config.BASE_DIR / "browser_profile"

    def get_image_path(self, clip_number: int) -> Path:
        return self.images_folder / f"image{clip_number:03d}.png"

    async def connect(self):
        """Connect to browser."""
        print("Connecting to browser...")

        is_first_run = not self.automation_profile_dir.exists()

        if is_first_run:
            print("\n" + "=" * 60)
            print("FIRST TIME SETUP")
            print("=" * 60)
            print("A browser window will open. Please:")
            print("  1. Go to https://higgsfield.ai")
            print("  2. Log in to your account")
            print("  3. Once logged in, come back here and press ENTER")
            print("=" * 60 + "\n")

        self.playwright = await async_playwright().start()
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.automation_profile_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            viewport={"width": 1920, "height": 1080},
            accept_downloads=True,
        )

        print("Browser launched successfully!")

        if is_first_run:
            page = await self.context.new_page()
            await page.goto("https://higgsfield.ai", wait_until="load", timeout=60000)
            input("\nPress ENTER after you have logged in to higgsfield.ai...")
            await page.goto(config.HIGGSFIELD_IMAGE_URL, wait_until="load", timeout=60000)
            await asyncio.sleep(2)
            current_url = page.url
            if "login" in current_url.lower() or "signin" in current_url.lower():
                print("WARNING: You don't appear to be logged in. Please try again.")
            else:
                print("Login verified! Your session has been saved.")
            await page.close()

        print("Browser ready.")

    async def disconnect(self):
        """Disconnect from browser."""
        if self.context:
            await self.context.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        print("Browser disconnected.")

    async def create_page(self) -> Page:
        """Create a new page/tab."""
        return await self.context.new_page()

    @staticmethod
    async def js_click(page: Page, selector: str) -> bool:
        """
        Click an element using JavaScript to bypass Playwright's click timeout issues.
        Returns True if element was found and clicked.
        """
        try:
            return await page.evaluate(f'''() => {{
                const el = document.querySelector('{selector}');
                if (el) {{
                    el.click();
                    return true;
                }}
                return false;
            }}''')
        except:
            return False

    @staticmethod
    async def js_click_element(page: Page, element) -> bool:
        """
        Click a specific element using JavaScript.
        """
        try:
            await page.evaluate('(el) => el.click()', element)
            return True
        except:
            return False

    @staticmethod
    async def close_popups(page: Page, max_attempts: int = 5) -> bool:
        """
        Aggressively close any popups/dialogs/modals on the page.
        Returns True if any popup was closed.
        """
        closed_any = False

        for attempt in range(max_attempts):
            try:
                closed = await page.evaluate('''() => {
                    const dialog = document.querySelector('[role="dialog"]');
                    if (dialog) {
                        const buttons = dialog.querySelectorAll('button');
                        for (const btn of buttons) {
                            const text = btn.textContent?.trim() || '';
                            const hasSvg = btn.querySelector('svg') !== null;
                            if (hasSvg && text.length < 3) {
                                btn.click();
                                return true;
                            }
                        }
                        if (buttons.length > 0) {
                            const firstBtn = buttons[0];
                            const text = firstBtn.textContent?.trim() || '';
                            if (text.length < 10 && !text.toLowerCase().includes('get') && !text.toLowerCase().includes('discount')) {
                                firstBtn.click();
                                return true;
                            }
                        }
                    }
                    return false;
                }''')

                if closed:
                    closed_any = True
                    await asyncio.sleep(0.3)
                    continue

                dialog = await page.query_selector('[role="dialog"]')
                if dialog:
                    await page.keyboard.press("Escape")
                    closed_any = True
                    await asyncio.sleep(0.3)
                    continue

                break

            except Exception as e:
                try:
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.2)
                except:
                    pass
                break

        return closed_any

    async def download_file(self, url: str, save_path: Path) -> bool:
        """Download a file from URL using HTTP."""
        try:
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = "https://higgsfield.ai" + url

            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    return True
                else:
                    print(f"    HTTP error: {response.status_code}")
                    return False
        except Exception as e:
            print(f"    Download error: {e}")
            return False


class ImageTabManager:
    """Manages the image generation tab."""

    def __init__(self, page: Page, client: HiggsfieldClient):
        self.page = page
        self.client = client

        # Tracking: clip_number -> prompt text (for matching)
        self.in_flight: Dict[int, str] = {}

        # Completed images waiting for download: clip_number -> url
        self.completed: Dict[int, str] = {}

        # Skipped/blocked images: clip_number -> reason
        self.skipped: Dict[int, str] = {}

        # Track submission order for position-based matching
        # Gallery maintains this order, so we can match by position
        self.submission_order: List[int] = []

        # Count of existing gallery items when we started (to offset positions)
        self.initial_gallery_count: int = 0
        
        # Track processed gallery item IDs to avoid duplicates/re-processing
        self.processed_ids: Set[str] = set()

        # Track processed gallery item IDs to avoid duplicates/re-processing
        # Uses URL as unique ID since we reverted to legacy logic without data-asset-id
        self.seen_gallery_urls: Set[str] = set()

        # Lock for thread-safe operations
        self.lock = asyncio.Lock()

    async def initialize(self):
        """Navigate to image page and initialize state."""
        print("[Image Tab] Navigating to image generator...")
        await self.page.goto(config.HIGGSFIELD_IMAGE_URL, wait_until="load", timeout=90000)
        await asyncio.sleep(3)
        await self._close_popups()

        await self._close_popups()

        # Capture existing gallery items to ignore them
        initial_items = await self._get_gallery_items()
        for item in initial_items:
            if item.get('id'):
                self.processed_ids.add(item['id'])
                
        print(f"[Image Tab] Found {len(initial_items)} existing images (IDs tracked)")

    async def submit_prompt(self, clip_number: int, prompt: str, max_retries: int = 3, pre_wait_seconds: float = 0) -> bool:
        """Submit a prompt for image generation with retry logic for popup handling."""
        async with self.lock:
            # Optional wait before starting interaction (useful for first image to let page settle)
            if pre_wait_seconds > 0:
                print(f"  [Image {clip_number}] Waiting {pre_wait_seconds}s before starting...")
                await asyncio.sleep(pre_wait_seconds)

            for attempt in range(max_retries):
                try:
                    print(f"  [Image {clip_number}] Submitting prompt..." + (f" (attempt {attempt + 1})" if attempt > 0 else ""))

                    await self._close_popups()

                    # Find the textarea using Playwright
                    textarea = await self.page.query_selector('textarea')
                    if not textarea:
                        print(f"  [Image {clip_number}] Could not find textarea, checking for popups...")
                        await self._close_popups()
                        await asyncio.sleep(0.5)
                        if attempt < max_retries - 1:
                            continue
                        print(f"  [Image {clip_number}] ERROR: Could not find textarea after retries")
                        return False

                    # Use Playwright's fill() method which properly triggers React state updates
                    # First clear the textarea, then fill with new prompt
                    await textarea.click()
                    await asyncio.sleep(0.2)

                    # Select all and delete to clear
                    await self.page.keyboard.press("Control+a")
                    await asyncio.sleep(0.1)
                    await self.page.keyboard.press("Backspace")
                    await asyncio.sleep(0.2)

                    # Use fill() which properly simulates typing and triggers React events
                    await textarea.fill(prompt)
                    await asyncio.sleep(0.3)

                    await self._enable_unlimited_mode()
                    await self._close_popups()

                    # Click Generate button using JavaScript to avoid timeout issues
                    generate_btn = await self.page.query_selector('button[type="submit"]')
                    if generate_btn:
                        await HiggsfieldClient.js_click_element(self.page, generate_btn)
                    else:
                        print(f"  [Image {clip_number}] Could not find Generate button, trying Enter key...")
                        await self.page.keyboard.press("Enter")

                    self.in_flight[clip_number] = prompt
                    self.submission_order.append(clip_number)

                    print(f"  [Image {clip_number}] Prompt submitted: {prompt[:50]}...")
                    await asyncio.sleep(1.5)  # Slightly longer pause to ensure submission registers
                    return True

                except Exception as e:
                    print(f"  [Image {clip_number}] Error on attempt {attempt + 1}: {e}")
                    await self._close_popups()
                    await asyncio.sleep(1)
                    if attempt >= max_retries - 1:
                        print(f"  [Image {clip_number}] ERROR: Failed after {max_retries} attempts")
                        return False

            return False

    async def check_completions(self) -> List[int]:
        """Check for newly completed or blocked images using position-based matching.
        Returns list of clip numbers that completed (not including blocked ones).
        """
        async with self.lock:
            # We need to check if we have any active submissions or history
            if not self.in_flight and not self.submission_order:
                return []

            try:
                await self._close_popups()

                # Get all gallery items (images and error cards)
                gallery_items = await self._get_gallery_items()

                if not gallery_items:
                    return []

                # Gallery is NEWEST FIRST (prepended), so our new items are at the START
                # Take only as many items as we've submitted (tracked in submission_order)
                num_submitted = len(self.submission_order)
                our_items = gallery_items[:num_submitted]

                if not our_items:
                    return []

                completed_clips = []

                # Since gallery is newest-first, we need to reverse the matching
                # Gallery position 0 = most recent submission = submission_order[-1]
                # Gallery position N-1 = oldest submission = submission_order[0]
                for i, item in enumerate(our_items):
                    # Reverse index: gallery[0] matches submission_order[N-1]
                    submission_idx = num_submitted - 1 - i
                    if submission_idx < 0 or submission_idx >= len(self.submission_order):
                        continue

                    clip_number = self.submission_order[submission_idx]

                    # Skip if already processed (not in in_flight)
                    if clip_number not in self.in_flight:
                        continue

                    if item.get('isError'):
                        # This image was blocked
                        error_text = item.get('errorText', 'Content blocked')
                        await self.mark_as_skipped(clip_number, error_text)
                        print(f"  [Image {clip_number}] BLOCKED: {error_text[:50]}")
                    elif item.get('imgSrc'):
                        # This image completed successfully
                        gallery_url = item['imgSrc']
                        
                        # DUPLICATE GUARD: If we've seen this URL before, it's an old image shuffled into this slot.
                        # Skip it and wait for a new image to appear.
                        if gallery_url in self.seen_gallery_urls:
                            continue

                        # Try to get high-res URL
                        highres_url = await self._get_highres_url_from_gallery(gallery_url)
                        
                        self.completed[clip_number] = highres_url or gallery_url
                        
                        # Mark as seen so we don't use it again for another clip
                        self.seen_gallery_urls.add(gallery_url)
                        
                        del self.in_flight[clip_number]
                        completed_clips.append(clip_number)
                        print(f"  [Image {clip_number}] Generation completed!")

                return completed_clips

            except Exception as e:
                print(f"  [Image Tab] Error checking completions: {e}")
                await self._close_popups()
                return []

    async def _get_highres_url_from_gallery(self, gallery_url: str) -> Optional[str]:
        """Click a gallery image to get its high-res URL from the modal."""
        try:
            # Find and click the image by matching SRC
            # This logic from the repo is safer than generic selector clicking
            images = await self.page.query_selector_all('main img')
            target_img = None
            for img in images:
                src = await img.get_attribute('src')
                if src == gallery_url:
                    target_img = img
                    break

            if not target_img:
                return self._extract_original_image_url(gallery_url)

            await HiggsfieldClient.js_click_element(self.page, target_img)
            await asyncio.sleep(1.5)

            modal = await self.page.query_selector('[role="dialog"]')
            if not modal:
                await self._close_popups()
                return self._extract_original_image_url(gallery_url)

            # Wait for high-res image to load in modal
            modal_img = await self.page.query_selector('[role="dialog"] img')
            highres_url = None

            if modal_img:
                last_url = None
                stable_count = 0
                for _ in range(12):  # 6 seconds max
                    current_url = await modal_img.get_attribute('src')
                    if current_url and current_url == last_url:
                        stable_count += 1
                        if stable_count >= 2:
                            highres_url = current_url
                            break
                    else:
                        stable_count = 0
                        last_url = current_url
                    await asyncio.sleep(0.5)

                if not highres_url:
                    highres_url = await modal_img.get_attribute('src')
                
                # Cleanup URL
                highres_url = self._extract_original_image_url(highres_url)

            await self.page.keyboard.press("Escape")
            await asyncio.sleep(0.2)

            return highres_url

        except Exception as e:
            try:
                await self.page.keyboard.press("Escape")
            except:
                pass
            return self._extract_original_image_url(gallery_url)

    async def get_completed_url(self, clip_number: int) -> Optional[str]:
        """Get the URL of a completed image."""
        return self.completed.get(clip_number)

    async def mark_downloaded(self, clip_number: int):
        """Mark an image as downloaded (remove from completed buffer)."""
        if clip_number in self.completed:
            del self.completed[clip_number]

    async def _get_gallery_count(self) -> int:
        """Get the count of gallery items."""
        try:
            count = await self.page.evaluate('''() => {
                const scrollContainer = document.querySelector('.overflow-auto.hide-scrollbar');
                if (!scrollContainer) return 0;
                const innerDiv = scrollContainer.querySelector('div.w-full');
                if (!innerDiv) return 0;
                return innerDiv.children.length;
            }''')
            return count or 0
        except:
            return 0

    async def _get_gallery_items(self) -> List[dict]:
        """Get gallery items with their type (image or error) in order.
        Returns list of dicts with: index, isError, imgSrc, errorText
        Items are in gallery order (NEWEST FIRST - most recent submission at index 0).
        """
        try:
            await self._scroll_to_top()
            items = await self.page.evaluate('''() => {
                const results = [];

                // Higgsfield structure: .overflow-auto.hide-scrollbar > div.w-full > [cards]
                const scrollContainer = document.querySelector('.overflow-auto.hide-scrollbar');
                if (!scrollContainer) return results;

                const innerDiv = scrollContainer.querySelector('div.w-full');
                if (!innerDiv) return results;

                const cards = innerDiv.children;

                for (let i = 0; i < cards.length; i++) {
                    const card = cards[i];
                    if (!card || card.nodeType !== 1) continue;

                    // Get Asset ID (Crucial for tracking unique images)
                    // The card itself usually has data-asset-id, or a child div
                    let assetId = card.getAttribute('data-asset-id');
                    if (!assetId) {
                        const innerDiv = card.querySelector('[data-asset-id]');
                        if (innerDiv) assetId = innerDiv.getAttribute('data-asset-id');
                    }

                    const img = card.querySelector('img');
                    const imgSrc = img ? img.src : null;
                    const altText = (img ? img.alt : '').toLowerCase(); // Capture alt text
                    const text = (card.innerText || '').toLowerCase(); // This captures the prompt text displayed on the card. Error text usually here.
                    
                    // Check if this is an error card
                    const hasImage = imgSrc && imgSrc.startsWith('http');
                    // Check for various error types:
                    // 1. "Sensitive content" / "Restricted content detected"
                    // 2. "Failed to generate" / "Prompt or file includes unsupported content"
                    const hasErrorText = text.includes('sensitive') ||
                                        text.includes('restricted') ||
                                        text.includes('failed') ||
                                        text.includes('unsupported');

                    // It's an error if it has error text (blocked/failed cards have no image but have error text)
                    const isError = hasErrorText;

                    results.push({
                        index: i,
                        id: assetId,
                        isError: isError,
                        imgSrc: hasImage ? imgSrc : null,
                        errorText: isError ? card.innerText.trim().substring(0, 100) : null,
                        promptText: text,
                        altText: altText // Return alt text
                    });
                }
                return results;
            }''')

            # Debug: print what we found
            if items:
                error_count = sum(1 for item in items if item.get('isError'))
                image_count = sum(1 for item in items if item.get('imgSrc'))
                # print(f"    [Gallery] {len(items)} items: {image_count} images, {error_count} blocked")

            return items or []
        except Exception as e:
            print(f"    Error getting gallery items: {e}")
            return []

    async def _close_popups(self):
        """Close any popups or modals using the robust shared helper."""
        try:
            dialog = await self.page.query_selector('[role="dialog"]')
            if dialog:
                text = await dialog.text_content()
                if text and "credit" in text.lower():
                    print("    WARNING: Credits dialog - Unlimited may not be enabled!")

            closed = await HiggsfieldClient.close_popups(self.page)
            if closed:
                print("    Closed popup(s)")
        except:
            pass

    async def _scroll_to_top(self):
        """Scroll the gallery to top to ensure new images are visible."""
        try:
            await self.page.evaluate('''() => {
                window.scrollTo(0, 0);
                const main = document.querySelector('main');
                if (main) main.scrollTop = 0;
            }''')
            await asyncio.sleep(0.3)
        except:
            pass

    def is_skipped(self, clip_number: int) -> bool:
        """Check if an image was skipped/blocked."""
        return clip_number in self.skipped

    def get_skipped_reason(self, clip_number: int) -> Optional[str]:
        """Get the reason an image was skipped."""
        return self.skipped.get(clip_number)

    async def mark_as_skipped(self, clip_number: int, reason: str):
        """Mark an image as skipped and remove from in-flight."""
        self.skipped[clip_number] = reason
        if clip_number in self.in_flight:
            del self.in_flight[clip_number]
        print(f"  [Image {clip_number}] SKIPPED: {reason}")

    async def _enable_unlimited_mode(self):
        """Enable Unlimited mode using JavaScript to avoid click timeouts."""
        try:
            result = await self.page.evaluate('''() => {
                const switchBtn = document.querySelector('button[role="switch"]');
                if (switchBtn) {
                    const isChecked = switchBtn.getAttribute('aria-checked') === 'true';
                    if (!isChecked) {
                        switchBtn.click();
                        return 'clicked';
                    }
                    return 'already_on';
                }
                return 'not_found';
            }''')

            if result == 'clicked':
                await asyncio.sleep(0.5)
                is_checked = await self.page.evaluate('''() => {
                    const switchBtn = document.querySelector('button[role="switch"]');
                    return switchBtn?.getAttribute('aria-checked') === 'true';
                }''')
                if is_checked:
                    print("    Unlimited mode: ON")
                else:
                    print("    WARNING: Unlimited mode failed to enable!")
            elif result == 'already_on':
                print("    Unlimited mode: already ON")
            else:
                print("    WARNING: Unlimited switch not found!")
        except Exception as e:
            print(f"    WARNING: Error enabling Unlimited: {e}")

    def _extract_original_image_url(self, url: str) -> str:
        """
        Extract the original CloudFront image URL from optimization wrappers.

        Higgsfield uses two optimization layers:
        1. Next.js: /_next/image?url=<ORIGINAL_URL>&w=1920&q=75
        2. Cloudflare CDN: /cdn-cgi/image/.../https://cloudfront.net/...

        We want the original full-resolution image from CloudFront.
        """
        if not url:
            return url

        try:
            # Handle Next.js image optimization
            # Pattern: /_next/image?url=<ENCODED_URL>&w=1920&q=75
            if '/_next/image' in url:
                parsed = urlparse(url)
                query_params = parse_qs(parsed.query)
                if 'url' in query_params:
                    original_url = unquote(query_params['url'][0])
                    print(f"    Extracted original URL from Next.js optimization")
                    return original_url

            # Handle Cloudflare CDN image resizing
            # Pattern: /cdn-cgi/image/fit=scale-down,.../https://cloudfront.net/...
            if '/cdn-cgi/image/' in url:
                # The original URL is embedded after the options
                # Find the https:// part after /cdn-cgi/image/.../
                match = re.search(r'/cdn-cgi/image/[^/]+/(https?://[^\s]+)', url)
                if match:
                    original_url = unquote(match.group(1))
                    # Remove _min.webp suffix if present to get PNG
                    if original_url.endswith('_min.webp'):
                        original_url = original_url.replace('_min.webp', '.png')
                    print(f"    Extracted original URL from Cloudflare CDN")
                    return original_url

        except Exception as e:
            print(f"    Warning: Could not extract original URL: {e}")

        return url


class ImagePipelineProcessor:
    """
    Orchestrates image generation with concurrent processing.

    - Maintains up to 8 images generating concurrently (Seedream 4.5 limit)
    - Downloads in order (waits for correct sequence)
    """

    MAX_IMAGES_IN_FLIGHT = 8

    def __init__(self, client: HiggsfieldClient, state_manager, file_manager):
        self.client = client
        self.state_manager = state_manager
        self.file_manager = file_manager

        self.image_tab: Optional[ImageTabManager] = None

        # Pipeline state
        self.next_image_to_submit = 1
        self.next_image_to_download = 1

        # Total count
        self.total_images = 0

        # Completion flags
        self.all_images_submitted = False
        self.all_images_downloaded = False

    async def run(self, prompts: list, on_progress: Callable = None):
        """Run the image generation pipeline."""
        self.total_images = len(prompts)

        print(f"\n{'='*60}")
        print(f"IMAGE PIPELINE PROCESSOR")
        print(f"{'='*60}")
        print(f"Total images: {self.total_images}")
        print(f"Max concurrent: {self.MAX_IMAGES_IN_FLIGHT}")
        print(f"{'='*60}\n")

        # Create tab
        image_page = await self.client.create_page()
        self.image_tab = ImageTabManager(image_page, self.client)

        # Initialize tab
        await self.image_tab.initialize()

        # Run the pipeline
        try:
            await asyncio.gather(
                self._image_producer(prompts),
                self._image_downloader(),
                self._monitor_progress(on_progress)
            )
        finally:
            await image_page.close()

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}\n")

    async def _image_producer(self, prompts: list):
        """Submit image prompts, maintaining up to MAX_IMAGES_IN_FLIGHT."""
        while self.next_image_to_submit <= self.total_images:
            in_flight = len(self.image_tab.in_flight)

            while in_flight < self.MAX_IMAGES_IN_FLIGHT and self.next_image_to_submit <= self.total_images:
                clip_number = self.next_image_to_submit
                prompt_data = prompts[clip_number - 1]
                prompt = prompt_data.get("prompt", "")

                self.state_manager.mark_image_started(clip_number)

                # Add 10s wait for the very first image to allow page to settle
                pre_wait = 10 if clip_number == 1 else 0
                success = await self.image_tab.submit_prompt(clip_number, prompt, pre_wait_seconds=pre_wait)
                if success:
                    self.next_image_to_submit += 1
                    in_flight += 1
                    # Wait 10 sec after first image, 1 sec after subsequent
                    if self.next_image_to_submit == 2:
                        await asyncio.sleep(10)
                    else:
                        await asyncio.sleep(1)
                else:
                    await asyncio.sleep(5)

            completed = await self.image_tab.check_completions()

            await asyncio.sleep(2)

        self.all_images_submitted = True
        print("[Image Producer] All prompts submitted")

        # Keep checking for completions until all downloaded
        while not self.all_images_downloaded:
            await self.image_tab.check_completions()
            await asyncio.sleep(2)

    async def _image_downloader(self):
        """Download completed images in order."""
        while self.next_image_to_download <= self.total_images:
            clip_number = self.next_image_to_download

            # Check if this image was skipped (e.g., sensitive content blocked)
            if self.image_tab.is_skipped(clip_number):
                reason = self.image_tab.get_skipped_reason(clip_number)
                self.state_manager.mark_image_failed(clip_number, f"Skipped: {reason}")
                print(f"  [Image {clip_number}] Skipping (was blocked: {reason})")
                self.next_image_to_download += 1
                continue

            url = await self.image_tab.get_completed_url(clip_number)

            if url:
                download_path = self.file_manager.get_image_path(clip_number)

                print(f"  [Image {clip_number}] Downloading...")
                success = await self.client.download_file(url, download_path)

                if success:
                    await self.image_tab.mark_downloaded(clip_number)
                    self.state_manager.mark_image_completed(clip_number, str(download_path))
                    print(f"  [Image {clip_number}] Downloaded to {download_path.name}")
                    self.next_image_to_download += 1
                else:
                    print(f"  [Image {clip_number}] Download failed, will retry...")
                    await asyncio.sleep(5)
            else:
                await asyncio.sleep(1)

        self.all_images_downloaded = True
        print("[Image Downloader] All images downloaded")

    async def _monitor_progress(self, on_progress: Callable = None):
        """Monitor and report progress."""
        while not self.all_images_downloaded:
            images_done = self.next_image_to_download - 1
            images_in_flight = len(self.image_tab.in_flight) if self.image_tab else 0

            print(f"\r[Progress] Images: {images_done}/{self.total_images} ({images_in_flight} generating)", end="")

            if on_progress:
                on_progress(self.state_manager.get_progress_summary())

            await asyncio.sleep(10)

        print()


class HiggsfieldBatchProcessor:
    """
    High-level batch processor for image generation.
    """

    def __init__(self, client: HiggsfieldClient, state_manager, file_manager):
        self.client = client
        self.state_manager = state_manager
        self.file_manager = file_manager

    async def generate_all(self, prompts: list, on_progress: Callable = None):
        """Generate all images using the pipeline."""
        pipeline = ImagePipelineProcessor(self.client, self.state_manager, self.file_manager)
        await pipeline.run(prompts, on_progress)

    async def generate_all_images(self, prompts: list, on_progress: Callable = None):
        """Generate all images (alias for generate_all)."""
        await self.generate_all(prompts, on_progress)


if __name__ == "__main__":
    async def test():
        client = HiggsfieldClient()
        try:
            await client.connect()
            print("Connected! Browser should be visible.")
            await asyncio.sleep(300)
        finally:
            await client.disconnect()

    asyncio.run(test())
