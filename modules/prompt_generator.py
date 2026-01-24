"""
Prompt Generator Module
Generates image prompts based on script segments
Supports multiple AI providers: OpenAI (ChatGPT), Google Gemini, xAI Grok
"""
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from modules.script_generator import get_ai_client
from modules.text_utils import clean_ai_text
import modules.srt_utils as srt_utils
# from datetime import timedelta


if TYPE_CHECKING:
    from modules.channel_manager import Channel


class PromptGenerator:
    def __init__(self, channel: "Channel"):
        """Initialize the prompt generator for a specific channel.

        Args:
            channel: The Channel configuration to generate prompts for
        """
        print(f"Initializing prompt generator with AI provider: {config.AI_PROVIDER}")
        self.client = get_ai_client()
        self.channel = channel

    def calculate_required_images(self, script: str) -> int:
        """Calculate how many images are needed based on script length."""
        word_count = len(script.split())
        duration_minutes = word_count / self.channel.words_per_minute
        return int(duration_minutes * self.channel.images_per_minute)

    def segment_script(self, script: str, num_segments: int) -> list[str]:
        """Divide script into roughly equal segments."""
        words = script.split()
        words_per_segment = len(words) // num_segments

        segments = []
        for i in range(num_segments):
            start = i * words_per_segment
            if i == num_segments - 1:
                # Last segment gets remaining words
                segment = ' '.join(words[start:])
            else:
                end = start + words_per_segment
                segment = ' '.join(words[start:end])
            segments.append(segment)

        return segments

    def generate_prompts_batch(self, topic: str, segments: list[str], start_index: int) -> list[dict]:
        """Generate image prompts for a batch of segments."""
        segments_text = ""
        for i, segment in enumerate(segments):
            segments_text += f"\n\nSEGMENT {start_index + i + 1}:\n{segment}"

        style_suffix = self.channel.image_style_suffix

        prompt = f"""You are an expert at creating image generation prompts for AI art generators.

VIDEO TOPIC: {topic}
CHANNEL NICHE: {self.channel.niche}

I need you to create ONE detailed image prompt for EACH of the following script segments. Each image should visually represent what's being narrated in that segment.

{segments_text}

REQUIREMENTS FOR EACH PROMPT:
- Create imagery appropriate for {self.channel.niche} content
- Include specific visual details (setting, characters, objects, lighting, atmosphere)
- Describe the scene composition (foreground, background, perspective)
- Make each image distinct and visually interesting
- Add dramatic lighting and cinematic composition
- End each prompt with: "{style_suffix}"

IMPORTANT:
- Each prompt should be 2-4 sentences, detailed but not overly long
- Focus on what can be SHOWN visually, not abstract concepts
- If the segment discusses abstract ideas, visualize a related concrete scene

Respond in this EXACT JSON format:
{{
    "prompts": [
        {{"segment_index": 1, "prompt": "your detailed image prompt here"}},
        {{"segment_index": 2, "prompt": "your detailed image prompt here"}}
    ]
}}

Generate the prompts now:"""

        response_text = self.client.generate(prompt)
        if not response_text:
            print(f"Error: Failed to generate prompts for batch starting at {start_index} after retries.")
            # Record failure in state (requires access to state_manager, strictly we should return empty or error indicator)
            # For now, we return fallback/placeholder prompts to keep the pipeline moving
            print("Generating fallback placeholder prompts...")
            return self._fallback_parse("", segments, start_index)

        response_text = clean_ai_text(response_text)

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        try:
            data = json.loads(response_text)
            return data.get("prompts", [])
        except json.JSONDecodeError:
            # Fallback: try to extract prompts manually
            print(f"Warning: Could not parse JSON response, attempting fallback...")
            return self._fallback_parse(response_text, segments, start_index)

    def _fallback_parse(self, text: str, segments: list, start_index: int) -> list[dict]:
        """Fallback parser if JSON parsing fails."""
        prompts = []
        lines = text.split('\n')
        style_suffix = self.channel.image_style_suffix

        for i, segment in enumerate(segments):
            # Try to find a prompt for this segment
            for line in lines:
                if f"segment {start_index + i + 1}" in line.lower() or f"#{start_index + i + 1}" in line:
                    # Found a line referencing this segment, next meaningful line is probably the prompt
                    idx = lines.index(line)
                    if idx + 1 < len(lines):
                        prompt_text = lines[idx + 1].strip()
                        if prompt_text and len(prompt_text) > 20:
                            prompts.append({
                                "segment_index": start_index + i + 1,
                                "prompt": prompt_text
                            })
                            break

        # If we didn't get enough prompts, just add empty entries as requested
        while len(prompts) < len(segments):
            idx = len(prompts)
            prompts.append({
                "segment_index": start_index + idx + 1,
                "prompt": ""  # Empty as requested
            })

        return prompts

    def generate_all_prompts(self, topic: str, script: str) -> list[dict]:
        """Generate all image prompts for the entire script."""
        num_images = self.calculate_required_images(script)
        print(f"Generating {num_images} image prompts...")

        segments = self.segment_script(script, num_images)
        all_prompts = []
        style_suffix = self.channel.image_style_suffix

        # Process in batches of 10 to avoid token limits
        batch_size = 10
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            print(f"  Processing segments {i + 1} to {i + len(batch)}...")

            batch_prompts = self.generate_prompts_batch(topic, batch, i)

            # Add segment text reference to each prompt
            for j, prompt_data in enumerate(batch_prompts):
                prompt_data["segment_text"] = segments[i + j][:200] + "..."  # First 200 chars
                prompt_data["clip_number"] = i + j + 1
                all_prompts.append(prompt_data)

        # Ensure we have exactly the right number of prompts
        while len(all_prompts) < num_images:
            all_prompts.append({
                "segment_index": len(all_prompts) + 1,
                "clip_number": len(all_prompts) + 1,
                "prompt": f"Scene from {topic}, {style_suffix}",
                "segment_text": "..."
            })

        return all_prompts[:num_images]


    def generate_prompts_from_srt(self, topic: str, srt_path: Path) -> list[dict]:
        """
        Generate image prompts based on SRT subtitles.
        """
        print(f"Reading subtitles from: {srt_path}")
        subtitles = srt_utils.parse_srt(srt_path)
        segments = srt_utils.group_srt_by_sentences(subtitles)
        
        print(f"Found {len(segments)} segments (sentences) in subtitles.")
        
        all_prompts = []
        style_suffix = self.channel.image_style_suffix
        
        # Prepare text segments for batch processing
        segment_texts = [seg['text'] for seg in segments]
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(segments), batch_size):
            batch_texts = segment_texts[i:i + batch_size]
            print(f"  Processing segments {i + 1} to {i + len(batch_texts)}...")
            
            # Use existing batch generation logic
            batch_prompts = self.generate_prompts_batch(topic, batch_texts, i)
            
            # Map back to SRT segments
            for j, prompt_data in enumerate(batch_prompts):
                # Safe index check
                if i + j < len(segments):
                    segment_info = segments[i + j]
                    
                    # Format timestamp not needed for optimized json

                    # Simple formatting, srt_utils might need a reverse helper or we format manually
                    # Just generic string format for now or keep consistency with SRT
                    
                    new_entry = {
                        "id": i + j + 1,
                        "prompt": prompt_data.get("prompt", ""),
                        "start_time": segment_info['start']
                    }
                    all_prompts.append(new_entry)

        # Ensure we didn't miss any (or format/id fixup)
        return all_prompts



if __name__ == "__main__":
    # Test with a sample
    from modules.channel_manager import ChannelManager

    manager = ChannelManager()
    channel = manager.get_channel("history_epoch")

    if channel:
        generator = PromptGenerator(channel)

        sample_topic = "The Fall of Constantinople 1453"
        sample_script = """
        The year was 1453. The great city of Constantinople, the jewel of the Byzantine Empire,
        stood as it had for over a thousand years. But outside its massive walls, an army of
        over 80,000 Ottoman soldiers waited. Sultan Mehmed II, just 21 years old, had set his
        sights on the prize that had eluded conquerors for centuries. Inside the walls, Emperor
        Constantine XI prepared his meager force of 7,000 defenders for what would become one
        of history's most legendary sieges.
        """ * 20  # Repeat to simulate longer script

        prompts = generator.generate_all_prompts(sample_topic, sample_script)

        print(f"\nGenerated {len(prompts)} prompts for channel: {channel.name}")
        print("\nFirst 3 prompts:")
        for p in prompts[:3]:
            print(f"\nClip {p['clip_number']}:")
            print(f"  {p['prompt'][:100]}...")
    else:
        print("ERROR: Could not find history_epoch channel")
