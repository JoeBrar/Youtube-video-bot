# YouTube Video Creator Bot

## Overview
Automated system for creating faceless YouTube video content packages across **multiple channels**. Generates AI scripts, image prompts, and images via Higgsfield.ai.

## Tech Stack
- **Language**: Python 3.10+
- **AI Providers**: OpenAI, Gemini, Grok (configurable)
- **Image Generation**: Higgsfield.ai (browser automation via Playwright)

## Multi-Channel Architecture

### channels.json
Each channel has its own configuration:
```json
{
  "id": "history_epoch",           // Used for folder names
  "name": "History Epoch",         // Display name
  "niche": "history",              // Content category
  "description": "...",            // Channel description
  "content_guidelines": "...",     // AI script style instructions
  "topic_categories": [...],       // Topic ideas for AI
  "image_style_suffix": "...",     // Appended to all image prompts
  "target_video_length_minutes": 20,
  "target_word_count": 3000,
  "images_per_minute": 4
}
```

### Round-Robin Processing
The bot iterates through channels, generating one video per channel per iteration:
```
Channel 1 → Video → Channel 2 → Video → Channel 1 → Video → ...
```
Single browser session persists across all channels.

## Output Structure
```
output/
├── {channel_id}/
│   ├── video1/
│   │   ├── script.txt
│   │   ├── titles.txt
│   │   ├── description.txt
│   │   ├── prompts.json
│   │   ├── topic.txt
│   │   ├── state.json
│   │   └── images/
│   │       ├── image001.png
│   │       └── ...
│   └── video2/
└── another_channel/
    └── ...
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `channel_manager.py` | Loads channels.json, manages channel configs |
| `script_generator.py` | AI script generation (uses channel config) |
| `prompt_generator.py` | Image prompt generation (uses channel style) |
| `higgsfield_client.py` | Browser automation pipeline |
| `state_manager.py` | Progress tracking, resume support |
| `file_manager.py` | Folder structure, image naming |

## Higgsfield Pipeline
- **Concurrency**: 8 images in-flight simultaneously
- **Completion**: Images complete out-of-order, downloaded in sequence
- **Settings**: FLUX.2 Pro, 2K quality, 16:9 aspect ratio, Unlimited mode

## Usage
```bash
# Install
pip install -r requirements.txt
playwright install chromium

# Run (round-robin all channels)
python main.py

# Start round-robin from specific channel
python main.py --channel history_epoch

# Specific topic
python main.py --topic "The Siege of Constantinople 1453"

# Script only (no images)
python main.py --script-only

# Resume interrupted session
python main.py --resume auto
python main.py --resume history_epoch/video3

# List videos
python main.py --list
python main.py --list --channel history_epoch
```

## Configuration (config.py)
```python
AI_PROVIDER = "openai"  # "openai", "gemini", "grok"
OPENAI_API_KEY = "..."
GEMINI_API_KEY = "..."
GROK_API_KEY = "..."
```

## First-Time Setup
1. `pip install -r requirements.txt`
2. `playwright install chromium`
3. Configure API keys in `config.py`
4. Add channels to `channels.json`
5. Run bot - log into higgsfield.ai when browser opens
6. Session saved in `browser_profile/` for future runs

## Notes
- Delete `browser_profile/` to re-login if session expires
- State tracking enables resume from any interruption
- Each channel's `image_style_suffix` ensures visual consistency
