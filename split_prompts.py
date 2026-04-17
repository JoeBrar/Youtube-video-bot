import sys
import os
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: python split_prompts.py <video_folder> -<num_parts>")
        print("Example: python split_prompts.py bible_dynamics/video1 -4")
        sys.exit(1)

    video_folder = sys.argv[1]
    num_parts_str = sys.argv[2]

    if not num_parts_str.startswith("-"):
        print("Error: The number of parts must be a negative flag, like -4")
        sys.exit(1)

    try:
        num_parts = int(num_parts_str[1:])
    except ValueError:
        print("Error: Invalid number flag")
        sys.exit(1)

    if num_parts <= 0:
        print("Error: Number of parts must be at least 1")
        sys.exit(1)

    # Automatically resolve path to output/ if not provided directly
    if not os.path.exists(video_folder) and os.path.exists(os.path.join("output", video_folder)):
        target_dir = os.path.join("output", video_folder)
    else:
        target_dir = video_folder

    if not os.path.exists(target_dir):
        print(f"Error: Directory not found -> {target_dir}")
        sys.exit(1)
        
    prompts_file = os.path.join(target_dir, "prompts.json")

    if not os.path.exists(prompts_file):
        print(f"Error: Could not find prompts.json at {prompts_file}")
        sys.exit(1)

    with open(prompts_file, "r", encoding="utf-8") as f:
        try:
            prompts = json.load(f)
        except json.JSONDecodeError:
            print("Error: prompts.json is not a valid JSON file")
            sys.exit(1)

    total_prompts = len(prompts)
    if total_prompts == 0:
        print("Error: prompts.json is empty")
        sys.exit(1)

    if num_parts > total_prompts:
        print(f"Warning: Number of parts ({num_parts}) is greater than total prompts ({total_prompts}). Setting parts to {total_prompts}.")
        num_parts = total_prompts

    print(f"Dividing {total_prompts} prompts into {num_parts} parts...")

    k, m = divmod(total_prompts, num_parts)
    
    for i in range(num_parts):
        start_idx = i * k + min(i, m)
        end_idx = (i + 1) * k + min(i + 1, m)
        
        chunk = prompts[start_idx:end_idx]
        out_filename = f"prompts{i+1}.json"
        out_path = os.path.join(target_dir, out_filename)
        
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(chunk, out_f, indent=2)
        
        # Determine the id range for logging
        first_id = chunk[0].get('id', 'N/A') if chunk else 'N/A'
        last_id = chunk[-1].get('id', 'N/A') if chunk else 'N/A'
            
        print(f"Created {out_filename} with {len(chunk)} prompts (IDs {first_id} to {last_id})")
        
    print("Successfully divided prompts!")

if __name__ == "__main__":
    main()
