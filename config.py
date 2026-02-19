"""
Configuration settings for the YouTube Video Creator Bot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# AI PROVIDER SETTINGS
# =============================================================================
# Choose your AI provider: "openai", "gemini", or "grok"
AI_PROVIDER = "gemini"

# =============================================================================
# API KEYS (set the one for your chosen provider)
# =============================================================================
# OpenAI (ChatGPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# xAI Grok
GROK_API_KEY = os.getenv("GROK_API_KEY")

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODULES_DIR = BASE_DIR / "modules"

# Chrome user data directory (for using existing session)
# Default paths for different OS
if os.name == 'nt':  # Windows
    CHROME_USER_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
else:  # Linux/Mac
    CHROME_USER_DATA_DIR = Path.home() / ".config" / "google-chrome"

# Chrome profile to use (usually "Default" or "Profile 1", etc.)
CHROME_PROFILE = "Default"

# =============================================================================
# VIDEO SETTINGS
# =============================================================================
TARGET_VIDEO_LENGTH_MINUTES = 20  # Target video length in minutes
WORDS_PER_MINUTE = 150  # Average speaking rate for TTS
TARGET_WORD_COUNT = TARGET_VIDEO_LENGTH_MINUTES * WORDS_PER_MINUTE  # ~3000 words

IMAGES_PER_MINUTE = 4  # How many images per minute of video
TOTAL_IMAGES = TARGET_VIDEO_LENGTH_MINUTES * IMAGES_PER_MINUTE  # ~80 images

# =============================================================================
# HIGGSFIELD SETTINGS
# =============================================================================
HIGGSFIELD_IMAGE_URL = "https://higgsfield.ai/image/seedream_v4_5"

# Image generation settings
IMAGE_MODEL = "Seedream 4.5"
IMAGE_QUALITY = "2K"
IMAGE_ASPECT_RATIO = "16:9"
IMAGE_UNLIMITED = True

# Concurrency limits (Pro plan - Seedream 4.5 supports 8 concurrent)
MAX_CONCURRENT_IMAGES = 8

# =============================================================================
# TIMING SETTINGS (for browser automation)
# =============================================================================
PAGE_LOAD_TIMEOUT = 60000  # 60 seconds
IMAGE_GENERATION_TIMEOUT = 300000  # 5 minutes per image
POLL_INTERVAL = 5000  # Check every 5 seconds for completion

# =============================================================================
# AI MODEL SETTINGS
# =============================================================================
# OpenAI models: "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4o", etc.
OPENAI_MODEL = "gpt-5.2"

# Gemini models: "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash", etc.
GEMINI_MODEL = "gemini-3-flash-preview"

# Grok models: "grok-3", "grok-2", etc.
GROK_MODEL = "grok-3"

# =============================================================================
# MULTI-CHANNEL SETTINGS
# =============================================================================
# Channel configurations are now stored in channels.json
# Each channel has its own settings for name, niche, topics, image style, etc.
CHANNELS_CONFIG_FILE = BASE_DIR / "channels.json"

# =============================================================================
# DEFAULT CONTENT SETTINGS (fallbacks if not specified in channel config)
# =============================================================================
DEFAULT_TARGET_VIDEO_LENGTH_MINUTES = 20
DEFAULT_TARGET_WORD_COUNT = 3000
DEFAULT_IMAGES_PER_MINUTE = 4
DEFAULT_IMAGE_STYLE_SUFFIX = "realistic, cinematic lighting, highly detailed, 16:9 aspect ratio, dramatic composition"

# =============================================================================
# WHISPER SETTINGS
# =============================================================================
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"

