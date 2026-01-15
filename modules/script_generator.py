"""
Script Generator Module
Supports multiple AI providers: OpenAI (ChatGPT), Google Gemini, xAI Grok
"""
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from modules.text_utils import clean_ai_text

if TYPE_CHECKING:
    from modules.channel_manager import Channel


class AIClient(ABC):
    """Abstract base class for AI clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        pass


class OpenAIClient(AIClient):
    """OpenAI (ChatGPT) client."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        print(f"Using OpenAI model: {self.model}")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


class GeminiClient(AIClient):
    """Google Gemini client."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        print(f"Using Gemini model: {config.GEMINI_MODEL}")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()


class GrokClient(AIClient):
    """xAI Grok client (uses OpenAI-compatible API)."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        self.model = config.GROK_MODEL
        print(f"Using Grok model: {self.model}")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


def get_ai_client() -> AIClient:
    """Factory function to get the appropriate AI client based on config."""
    provider = config.AI_PROVIDER.lower()

    if provider == "openai":
        return OpenAIClient()
    elif provider == "gemini":
        return GeminiClient()
    elif provider == "grok":
        return GrokClient()
    else:
        raise ValueError(f"Unknown AI provider: {provider}. Choose 'openai', 'gemini', or 'grok'")


class ScriptGenerator:
    def __init__(self, channel: "Channel"):
        """Initialize the script generator for a specific channel.

        Args:
            channel: The Channel configuration to generate content for
        """
        print(f"Initializing AI provider: {config.AI_PROVIDER}")
        self.client = get_ai_client()
        self.channel = channel
        print(f"Generating content for channel: {channel.name}")

    def generate_topic(self) -> str:
        """Generate an interesting topic for a video based on channel niche."""
        video_length = self.channel.target_video_length_minutes

        prompt = f"""You are a content strategist for a YouTube channel called "{self.channel.name}" that creates engaging {self.channel.niche} videos.

Channel description: {self.channel.description}

Generate ONE unique and fascinating {self.channel.niche} topic that would make a great {video_length}-minute YouTube video. The topic should be:
- Interesting and engaging for a general audience
- Have enough depth for a detailed exploration
- Be somewhat lesser-known or have a unique angle
- Have visual potential (can be illustrated with compelling images)

Categories to consider: {', '.join(self.channel.topic_categories)}

Respond with ONLY the topic title, nothing else. Make it specific, not generic.

Your topic:"""

        try:
            topic = self.client.generate(prompt)
            return clean_ai_text(topic)
        except Exception as e:
            print(f"ERROR generating topic: {e}")
            raise

    def generate_script(self, topic: str) -> str:
        """Generate a full voiceover script for the given topic."""
        video_length = self.channel.target_video_length_minutes
        word_count = self.channel.target_word_count

        prompt = f"""You are a professional scriptwriter for the YouTube channel "{self.channel.name}".
Write a complete voiceover script for a video about: "{topic}"

CHANNEL STYLE GUIDELINES:
{self.channel.content_guidelines}

REQUIREMENTS:
- Length: Approximately {word_count} words (for a {video_length}-minute video)
- Start with a compelling hook in the first 30 seconds to grab attention
- Use vivid descriptions and storytelling techniques
- Include interesting facts and details
- Build tension and maintain interest throughout
- End with a thought-provoking conclusion

IMPORTANT:
- Write ONLY the narration script (what will be spoken)
- Do NOT include stage directions, scene descriptions, or visual cues
- Do NOT include [brackets] or (parentheses) with instructions
- Do NOT include timestamps or section headers
- Write in a flowing, continuous narrative style
- Use dramatic pauses through sentence structure, not explicit markers

Write the complete script now:"""

        script = self.client.generate(prompt)
        return clean_ai_text(script)

    def generate_titles(self, topic: str, script: str) -> list[str]:
        """Generate catchy YouTube title ideas."""
        # Take first 500 words of script for context
        script_preview = ' '.join(script.split()[:500])

        prompt = f"""You are a YouTube title expert. Generate 10 compelling title ideas for this {self.channel.niche} video.

Channel: {self.channel.name}
Topic: {topic}

Script preview: {script_preview}...

TITLE REQUIREMENTS:
- Maximum 60 characters (YouTube limit)
- Create curiosity and urgency
- Use power words that drive clicks
- Some can use numbers ("The 3 Mistakes That...")
- Some can use questions
- Some can use dramatic statements
- Avoid clickbait that doesn't deliver

Generate 10 titles, one per line, numbered 1-10:"""

        response = self.client.generate(prompt)
        response = clean_ai_text(response)
        titles = []
        import re
        for line in response.strip().split('\n'):
            line = line.strip()
            if line:
                cleaned = re.sub(r'^[\d]+[.\):\-]\s*', '', line)
                if cleaned:
                    titles.append(cleaned)

        return titles[:10]

    def generate_description(self, topic: str, script: str) -> str:
        """Generate a YouTube video description."""
        script_preview = ' '.join(script.split()[:300])

        prompt = f"""You are a YouTube SEO expert. Write a compelling video description for this {self.channel.niche} video.

Topic: {topic}
Channel: {self.channel.name}

Script preview: {script_preview}...

DESCRIPTION REQUIREMENTS:
- First 2 lines should hook viewers (shown in preview)
- Include a brief summary of what viewers will learn
- Add relevant hashtags at the end (5-8 hashtags)
- Include a call-to-action to subscribe
- Total length: 150-300 words
- Make it SEO-friendly with relevant keywords

Write the description:"""

        description = self.client.generate(prompt)
        return clean_ai_text(description)

    def estimate_duration_minutes(self, script: str) -> float:
        """Estimate video duration based on word count."""
        word_count = len(script.split())
        return word_count / self.channel.words_per_minute

    def generate_all(self, topic: str = None) -> dict:
        """Generate all content for a video."""
        if topic is None:
            print("Generating topic...")
            topic = self.generate_topic()
            print(f"Topic: {topic}")

        print("Generating script...")
        script = self.generate_script(topic)
        duration = self.estimate_duration_minutes(script)
        print(f"Script generated: {len(script.split())} words (~{duration:.1f} minutes)")

        print("Generating titles...")
        titles = self.generate_titles(topic, script)

        print("Generating description...")
        description = self.generate_description(topic, script)

        return {
            "topic": topic,
            "script": script,
            "titles": titles,
            "description": description,
            "word_count": len(script.split()),
            "estimated_duration_minutes": duration
        }


if __name__ == "__main__":
    # Test the script generator
    from modules.channel_manager import ChannelManager

    manager = ChannelManager()
    channel = manager.get_channel("history_epoch")

    if channel:
        generator = ScriptGenerator(channel)
        result = generator.generate_all()

        print("\n" + "="*50)
        print(f"Channel: {channel.name}")
        print(f"Topic: {result['topic']}")
        print(f"Word count: {result['word_count']}")
        print(f"Estimated duration: {result['estimated_duration_minutes']:.1f} minutes")
        print("\nTitles:")
        for i, title in enumerate(result['titles'], 1):
            print(f"  {i}. {title}")
        print("\nDescription preview:")
        print(result['description'][:200] + "...")
    else:
        print("ERROR: Could not find history_epoch channel")
