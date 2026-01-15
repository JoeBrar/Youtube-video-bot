"""
Diagnostic script to analyze Higgsfield.ai website structure
Run this to capture screenshots and find the correct selectors
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Output folder for screenshots
DIAG_DIR = Path("./diagnostics")
DIAG_DIR.mkdir(exist_ok=True)

async def diagnose():
    print("Starting Higgsfield diagnostic...")
    print(f"Screenshots will be saved to: {DIAG_DIR.absolute()}")

    playwright = await async_playwright().start()

    # Use a fresh profile for diagnosis
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(Path("./browser_profile")),
        headless=False,
        viewport={"width": 1920, "height": 1080},
    )

    page = await context.new_page()

    try:
        # Step 1: Navigate to the image generator
        print("\n[Step 1] Navigating to image generator...")
        await page.goto("https://higgsfield.ai/image/nano_banana_2", wait_until="load", timeout=60000)
        await asyncio.sleep(5)  # Wait for JS
        await page.screenshot(path=DIAG_DIR / "01_initial_page.png", full_page=True)
        print(f"  Screenshot saved: 01_initial_page.png")

        # Step 2: Check for popups/modals/ads
        print("\n[Step 2] Looking for popups/modals/ads...")
        popup_selectors = [
            'button[aria-label*="close" i]',
            'button[aria-label*="dismiss" i]',
            '[class*="close"]',
            '[class*="modal"] button',
            '[class*="popup"] button',
            '[class*="dialog"] button',
            'button:has-text("×")',
            'button:has-text("X")',
            'button:has-text("Close")',
            'button:has-text("Skip")',
            'button:has-text("No thanks")',
            '[class*="overlay"] button',
        ]

        for selector in popup_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    print(f"  Found {len(elements)} elements matching: {selector}")
                    for i, el in enumerate(elements):
                        text = await el.text_content()
                        visible = await el.is_visible()
                        print(f"    [{i}] visible={visible}, text='{text[:50] if text else 'N/A'}'")
            except:
                pass

        await page.screenshot(path=DIAG_DIR / "02_after_popup_check.png", full_page=True)

        # Step 3: Find all input elements
        print("\n[Step 3] Looking for input elements (textareas, inputs)...")
        input_selectors = [
            'textarea',
            'input[type="text"]',
            'input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"])',
            '[contenteditable="true"]',
        ]

        for selector in input_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    print(f"  Found {len(elements)} elements matching: {selector}")
                    for i, el in enumerate(elements):
                        placeholder = await el.get_attribute("placeholder")
                        classname = await el.get_attribute("class")
                        visible = await el.is_visible()
                        print(f"    [{i}] visible={visible}, placeholder='{placeholder}', class='{classname[:50] if classname else 'N/A'}'")
            except Exception as e:
                print(f"  Error with {selector}: {e}")

        # Step 4: Find all buttons
        print("\n[Step 4] Looking for buttons...")
        try:
            buttons = await page.query_selector_all('button')
            print(f"  Found {len(buttons)} buttons")
            for i, btn in enumerate(buttons[:20]):  # First 20 buttons
                text = await btn.text_content()
                classname = await btn.get_attribute("class")
                visible = await btn.is_visible()
                print(f"    [{i}] visible={visible}, text='{text[:30] if text else 'N/A'}', class='{classname[:40] if classname else 'N/A'}'")
        except Exception as e:
            print(f"  Error: {e}")

        # Step 5: Find settings/options elements
        print("\n[Step 5] Looking for settings elements...")
        settings_keywords = ['aspect', 'ratio', 'quality', 'model', 'resolution', '16:9', '2K', '4K', 'unlimited']

        for keyword in settings_keywords:
            try:
                elements = await page.query_selector_all(f'text=/{keyword}/i')
                if elements:
                    print(f"  Found {len(elements)} elements containing '{keyword}'")
                    for i, el in enumerate(elements[:5]):
                        tag = await el.evaluate("el => el.tagName")
                        text = await el.text_content()
                        visible = await el.is_visible()
                        print(f"    [{i}] tag={tag}, visible={visible}, text='{text[:40] if text else 'N/A'}'")
            except:
                pass

        # Step 6: Find dropdowns/selects
        print("\n[Step 6] Looking for dropdowns/selects...")
        dropdown_selectors = [
            'select',
            '[role="listbox"]',
            '[role="combobox"]',
            '[class*="dropdown"]',
            '[class*="select"]',
        ]

        for selector in dropdown_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    print(f"  Found {len(elements)} elements matching: {selector}")
            except:
                pass

        await page.screenshot(path=DIAG_DIR / "03_elements_found.png", full_page=True)

        # Step 7: Let user interact
        print("\n" + "="*60)
        print("MANUAL INSPECTION TIME")
        print("="*60)
        print("The browser is now open. Please:")
        print("1. Look at the page and note what elements you see")
        print("2. Try clicking on settings to see what happens")
        print("3. Right-click on elements and 'Inspect' to see their selectors")
        print("\nWhen done, come back here and describe what you see.")
        print("Press ENTER to take a final screenshot and close...")
        input()

        await page.screenshot(path=DIAG_DIR / "04_final_state.png", full_page=True)
        print(f"Final screenshot saved: 04_final_state.png")

        # Get page HTML structure (simplified)
        print("\n[Step 8] Saving page HTML structure...")
        html = await page.content()
        with open(DIAG_DIR / "page_structure.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML saved to: page_structure.html")

    except Exception as e:
        print(f"Error during diagnosis: {e}")
        await page.screenshot(path=DIAG_DIR / "error_state.png", full_page=True)
    finally:
        await context.close()
        await playwright.stop()
        print("\nDiagnosis complete!")
        print(f"Check the screenshots in: {DIAG_DIR.absolute()}")


if __name__ == "__main__":
    asyncio.run(diagnose())
