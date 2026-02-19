
import asyncio
from modules.higgsfield_client import HiggsfieldClient, ImageTabManager

async def debug_gallery():
    print("Initializing Higgsfield Client...")
    client = HiggsfieldClient()
    
    try:
        await client.connect()
        page = await client.create_page()
        
        tab = ImageTabManager(page, client)
        await tab.initialize() # Navigates to page
        
        print("\n[DEBUG] Waiting 15s for gallery to load...")
        await asyncio.sleep(15)
        
        print("\n[DEBUG] Fetching gallery items...")
        items = await tab._get_gallery_items()
        
        if items:
            print(f"\n[DEBUG] Found {len(items)} items. Clicking first item to open modal...")
            
            # Click the first image using the function we already have access to via client? 
            # We are using tab manager, need to do it manually using page
            await page.evaluate(f'''() => {{
                const cards = document.querySelectorAll('.overflow-auto.hide-scrollbar div.w-full > div');
                if (cards.length > 0) {{
                    const img = cards[0].querySelector('img');
                    if (img) img.click();
                }}
            }}''')
            
            print("Clicked. Waiting for modal...")
            await asyncio.sleep(5)
            
            # Dump modal
            modal_html = await page.evaluate('''() => {
                const dialog = document.querySelector('[role="dialog"]');
                return dialog ? dialog.outerHTML : null;
            }''')
            
            if modal_html:
                print("\n[DEBUG] Saving Modal HTML to debug_modal.html...")
                with open('debug_modal.html', 'w', encoding='utf-8') as f:
                    f.write(modal_html)
                print("Saved.")
            else:
                print("No modal found!")
                
        else:
            print("No items found in gallery!")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_gallery())
