from google import genai
from google.genai import types
from PIL import Image
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
aspect_ratio = "16:9"
resolution = "2K"
model = "gemini-3-pro-image-preview"
max_retries = 3
retry_delay = 2  # seconds

# Paths (relative to script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(script_dir, "prompts.json")
images_folder = os.path.join(script_dir, "images")

# Initialize client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_prompts():
    """Load prompts from prompts.json"""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_existing_images():
    """Get list of existing image numbers in images folder"""
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        return set()

    existing = set()
    for filename in os.listdir(images_folder):
        if filename.startswith("image") and filename.endswith(".png"):
            # Extract number from imageXXX.png
            num_str = filename.replace("image", "").replace(".png", "")
            try:
                existing.add(int(num_str))
            except ValueError:
                continue
    return existing

def find_missing_images(prompts, existing_images):
    """Find which images are missing based on clip_number"""
    required_images = {item['clip_number'] for item in prompts}
    missing = sorted(required_images - existing_images)
    return missing

def generate_image(prompt_text, image_number, retry_count=0):
    """Generate a single image with retry logic"""
    try:
        print(f"  Attempting to generate image{image_number:03d}.png... ", end='', flush=True)

        response = client.models.generate_content(
            model=model,
            contents=[prompt_text],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution
                ),
            )
        )

        for part in response.parts:
            if part.text is not None:
                print(f"Response text: {part.text}")
            elif image := part.as_image():
                output_path = os.path.join(images_folder, f"image{image_number:03d}.png")
                image.save(output_path)
                print(f"✓ Saved successfully!")
                return True

        print(f"✗ No image in response")
        return False

    except Exception as e:
        print(f"✗ Error: {str(e)}")

        if retry_count < max_retries:
            print(f"  Retrying ({retry_count + 1}/{max_retries})...")
            time.sleep(retry_delay)
            return generate_image(prompt_text, image_number, retry_count + 1)
        else:
            print(f"  Failed after {max_retries} attempts. Skipping.")
            return False

def main():
    print("=" * 60)
    print("Missing Image Generator - Gemini Nano Banana")
    print("=" * 60)

    # Load prompts
    print("\n[1/4] Loading prompts from prompts.json...")
    prompts = load_prompts()
    print(f"  Found {len(prompts)} total prompts")

    # Check existing images
    print("\n[2/4] Checking existing images in images folder...")
    existing_images = get_existing_images()
    print(f"  Found {len(existing_images)} existing images")

    # Find missing images
    print("\n[3/4] Identifying missing images...")
    missing_images = find_missing_images(prompts, existing_images)

    if not missing_images:
        print("  ✓ No missing images! All images are already generated.")
        return

    print(f"  Found {len(missing_images)} missing images: {missing_images}")

    # Generate missing images
    print(f"\n[4/4] Generating {len(missing_images)} missing images...")
    print("-" * 60)

    success_count = 0
    failed_images = []

    for i, image_num in enumerate(missing_images, 1):
        # Find the prompt for this image number
        prompt_data = next((p for p in prompts if p['clip_number'] == image_num), None)

        if not prompt_data:
            print(f"[{i}/{len(missing_images)}] Warning: No prompt found for image{image_num:03d}")
            failed_images.append(image_num)
            continue

        print(f"\n[{i}/{len(missing_images)}] Processing image{image_num:03d}")
        print(f"  Prompt: {prompt_data['prompt'][:80]}...")

        if generate_image(prompt_data['prompt'], image_num):
            success_count += 1
        else:
            failed_images.append(image_num)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total missing images: {len(missing_images)}")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {len(failed_images)}")

    if failed_images:
        print(f"\nFailed images: {failed_images}")
    else:
        print("\n✓ All missing images generated successfully!")

if __name__ == "__main__":
    main()