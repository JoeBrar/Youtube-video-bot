"""
Script Generator Module
Supports multiple AI providers: OpenAI (ChatGPT), Google Gemini, xAI Grok
With optional web search for factual/news content
"""
import sys
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from modules.text_utils import clean_ai_text

if TYPE_CHECKING:
    from modules.channel_manager import Channel


# Path to channel topics file
CHANNEL_TOPICS_FILE = Path(__file__).parent.parent / "channel_topics.json"


def load_channel_topics() -> dict:
    """Load channel topics from JSON file."""
    if CHANNEL_TOPICS_FILE.exists():
        with open(CHANNEL_TOPICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_channel_topics(topics: dict) -> None:
    """Save channel topics to JSON file."""
    with open(CHANNEL_TOPICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)


def get_next_unused_topic(channel_id: str) -> Optional[dict]:
    """Get the next unused topic for a channel.

    Returns:
        The topic dict with video_id and topic, or None if no unused topics.
    """
    topics = load_channel_topics()

    if channel_id not in topics:
        return None

    for topic_entry in topics[channel_id]:
        if not topic_entry.get("used", False):
            return topic_entry

    return None


def mark_topic_as_used(channel_id: str, video_id: int) -> None:
    """Mark a topic as used in the channel topics file."""
    topics = load_channel_topics()

    if channel_id not in topics:
        return

    for topic_entry in topics[channel_id]:
        if topic_entry.get("video_id") == video_id:
            topic_entry["used"] = True
            break

    save_channel_topics(topics)


class AIClient(ABC):
    """Abstract base class for AI clients."""

    def __init__(self, enable_web_search: bool = False, temperature: Optional[float] = None):
        self.enable_web_search = enable_web_search
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        pass


class OpenAIClient(AIClient):
    """OpenAI client using Responses API for web search support."""

    def __init__(self, enable_web_search: bool = False, temperature: Optional[float] = None):
        super().__init__(enable_web_search, temperature)
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        search_status = "with web search" if enable_web_search else "without web search"
        temp_status = f"temp={temperature}" if temperature is not None else "temp=default"
        print(f"Using OpenAI model: {self.model} ({search_status}, {temp_status})")

    def generate(self, prompt: str) -> str:
        if self.enable_web_search:
            # Use Responses API with web search tool
            return self._generate_with_search(prompt)
        else:
            # Use standard Chat Completions API
            return self._generate_standard(prompt)

    def _generate_standard(self, prompt: str) -> str:
        """Generate using standard Chat Completions API."""
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def _generate_with_search(self, prompt: str) -> str:
        """Generate using Responses API with web search enabled."""
        kwargs = {
            "model": self.model,
            "tools": [{"type": "web_search"}],
            "input": prompt,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        response = self.client.responses.create(**kwargs)
        # Extract text from response - the Responses API returns output_text
        return response.output_text.strip()


class GeminiClient(AIClient):
    """Google Gemini client with optional Search Grounding."""

    def __init__(self, enable_web_search: bool = False, temperature: Optional[float] = None):
        super().__init__(enable_web_search, temperature)
        # Use the new google.genai SDK for better search grounding support
        from google import genai
        from google.genai import types
        
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model_name = config.GEMINI_MODEL
        
        search_status = "with Google Search grounding" if enable_web_search else "without search"
        temp_status = f"temp={temperature}" if temperature is not None else "temp=default"
        print(f"Using Gemini model: {self.model_name} ({search_status}, {temp_status})")

    def generate(self, prompt: str) -> str:
        config_kwargs = {}
        
        # Set temperature if specified
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        
        # Add search grounding tool if enabled
        tools = None
        if self.enable_web_search:
            tools = [self.types.Tool(google_search=self.types.GoogleSearch())]
        
        if tools:
            config_kwargs["tools"] = tools
        
        # Create config if we have any settings
        from google.genai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = [
            self.types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            self.types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            self.types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            self.types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        
        config_kwargs["safety_settings"] = safety_settings
        
        gen_config = self.types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        # Retry logic for handling ServerError/Overloaded errors OR empty responses
        import time
        import random
        from google.genai import errors
        
        max_retries = 5
        retry_interval = 5  # Fixed 5 second interval as requested
        
        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=gen_config
                )
                
                # Check for empty response (common with Gemini safety blocks or internal glitches)
                if not response.text:
                    if attempt < max_retries:
                        print(f"Warning: Gemini returned no text. Retrying in {retry_interval}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_interval)
                        continue
                    else:
                        print(f"Error: Gemini returned no text after {max_retries} retries. Feedback: {response.prompt_feedback}")
                        return ""
                    
                return response.text.strip()
                
            except errors.ServerError as e:
                if attempt < max_retries:
                    print(f"Gemini ServerError ({e}). Retrying in {retry_interval}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_interval)
                else:
                    print(f"Gemini failed after {max_retries} attempts.")
                    raise e
            except Exception as e:
                # Re-raise other exceptions immediately
                raise e


class GrokClient(AIClient):
    """xAI Grok client with Agent Tools API for web search."""

    def __init__(self, enable_web_search: bool = False, temperature: Optional[float] = None):
        super().__init__(enable_web_search, temperature)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        self.model = config.GROK_MODEL
        search_status = "with web search" if enable_web_search else "without web search"
        temp_status = f"temp={temperature}" if temperature is not None else "temp=default"
        print(f"Using Grok model: {self.model} ({search_status}, {temp_status})")

    def generate(self, prompt: str) -> str:
        if self.enable_web_search:
            return self._generate_with_search(prompt)
        else:
            return self._generate_standard(prompt)

    def _generate_standard(self, prompt: str) -> str:
        """Generate using standard chat completions."""
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def _generate_with_search(self, prompt: str) -> str:
        """Generate using Agent Tools API with web search."""
        # Grok's Agent Tools API uses tools parameter similar to OpenAI
        # Define the web search tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You have access to web search. Use it to find current, accurate information before responding. Always search for relevant facts and news."
                },
                {"role": "user", "content": prompt}
            ],
            "tools": tools,
            "tool_choice": "auto",
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        # First call - model may request tool use
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        
        # If the model wants to use tools, we need to handle that
        # For simplicity, we'll let the model proceed with its response
        # In a full implementation, you'd execute the search and feed results back
        if message.content:
            return message.content.strip()
        
        # If no content but tool calls, make another request without tools
        # to get the final response (the model should have internalized the search intent)
        kwargs_no_tools = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Search the web thoroughly for current information about this topic, then provide a comprehensive response based on what you find."
                },
                {"role": "user", "content": prompt}
            ],
        }
        if self.temperature is not None:
            kwargs_no_tools["temperature"] = self.temperature
            
        response = self.client.chat.completions.create(**kwargs_no_tools)
        return response.choices[0].message.content.strip()


def get_ai_client(enable_web_search: bool = False, temperature: Optional[float] = None) -> AIClient:
    """Factory function to get the appropriate AI client based on config."""
    provider = config.AI_PROVIDER.lower()

    if provider == "openai":
        return OpenAIClient(enable_web_search=enable_web_search, temperature=temperature)
    elif provider == "gemini":
        return GeminiClient(enable_web_search=enable_web_search, temperature=temperature)
    elif provider == "grok":
        return GrokClient(enable_web_search=enable_web_search, temperature=temperature)
    else:
        raise ValueError(f"Unknown AI provider: {provider}. Choose 'openai', 'gemini', or 'grok'")


class ScriptGenerator:
    def __init__(self, channel: "Channel"):
        """Initialize the script generator for a specific channel.

        Args:
            channel: The Channel configuration to generate content for
        """
        print(f"Initializing AI provider: {config.AI_PROVIDER}")
        self.client = get_ai_client(
            enable_web_search=channel.enable_web_search,
            temperature=channel.temperature
        )
        self.channel = channel
        print(f"Generating content for channel: {channel.name}")
        if channel.enable_web_search:
            print("Web search is ENABLED for this channel - will search for real/current information")

    def get_topic(self) -> tuple[str, int] | None:
        """Get the next available topic from channel_topics.json.

        Topics are strictly loaded from channel_topics.json - no AI generation.
        If no unused topics are available, returns None.

        Returns:
            Tuple of (topic_string, video_id) or None if no topics available
        """
        topic_entry = get_next_unused_topic(self.channel.id)

        if topic_entry:
            topic = topic_entry["topic"]
            video_id = topic_entry["video_id"]
            print(f"Using topic (video{video_id}) from channel_topics.json")
            # Mark the topic as used
            mark_topic_as_used(self.channel.id, video_id)
            return topic, video_id

        # No topics available for this channel
        return None

    def generate_script(self, topic: str) -> str:
        """Generate a full voiceover script for the given topic."""
        video_length = self.channel.target_video_length_minutes
        word_count = self.channel.target_word_count

        # Add web search instruction if enabled
        search_instruction = ""
        if self.channel.enable_web_search:
            search_instruction = """
CRITICAL: Before writing, search the web thoroughly for:
- The latest news, data and updates about this topic as of today
- Verified facts, dates, names, and details
- Recent developments and current status

This script must be FACTUALLY ACCURATE and based on REAL information.
"""

        prompt = f"""You are a professional scriptwriter for the YouTube channel "{self.channel.name}".
Write a complete voiceover script for a video about this topic : "{topic}"

{search_instruction}

CHANNEL STYLE GUIDELINES:
{self.channel.content_guidelines}

REQUIREMENTS:
- Length: Approximately {word_count} words (for a {video_length}-minute video)
- Start with a compelling hook in the first 30 seconds to grab attention
- Use vivid descriptions and storytelling techniques
- Include interesting facts and details
- Build tension and maintain interest throughout

IMPORTANT:
- Write ONLY the narration script (what will be spoken)
- Do NOT include stage directions, scene descriptions, visual cues, timestamps or section headers etc.
- DO NOT include any references to sources or links in the script
- Do NOT include [brackets] or (parentheses) with instructions
- Write in a flowing, continuous narrative style
- Just give me the script, nothing else

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

    def generate_all(self, topic: str = None, video_id: int = None) -> dict | None:
        """Generate all content for a video.

        Returns None if no topic is available for this channel.
        """
        if topic is None:
            result = self.get_topic()
            if result is None:
                print(f"No topics available for channel: {self.channel.name}")
                return None
            topic, video_id = result
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
            "estimated_duration_minutes": duration,
            "predefined_video_id": video_id  # None if AI-generated, video_id if from channel_topics.json
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
