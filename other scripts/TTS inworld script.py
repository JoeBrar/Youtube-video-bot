import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://api.inworld.ai/tts/v1/voice"
inworld_api_key = os.getenv('INWORLD_API_KEY')
print("inworld_api_key: ", inworld_api_key)

headers = {
    "Authorization": f"Basic {inworld_api_key}",
    "Content-Type": "application/json"
}

payload = {
  "text": """ If you’ve ever turned an old radio dial late at night, you know the feeling.
That moment when the world goes quiet… and the static starts to sound like it’s breathing.
Now imagine this.
You’re alone, the room lit by nothing but the faint glow of a receiver. You sweep through the bands and catch something that shouldn’t be there. A child’s voice. Not singing to you, not talking to you… but counting. Calm. Mechanical. As if the numbers matter more than words ever could.
And then the melody begins.
A simple, looping tune. Bright enough to feel wrong in the dark. Familiar in a way you can’t place. Like a nursery rhyme remembered from a life you never lived.
This broadcast had a name, given by the people who found it and couldn’t stop listening.
Swedish Rhapsody.
No station identification. No explanation. Just a child’s voice, a music box innocence, and strings of numbers slipping into the night like coded confessions.
For decades, it haunted the shortwave spectrum… and it may have been a thread connected to a global spy network that never truly went away.
Tonight, we’re going inside the phantom radio station of numbers. We’re going to trace the sound, the theories, the eyewitness accounts, and the chilling possibility that someone, somewhere, was always meant to hear it.
Not everyone did.
But someone did.
And if you’re listening now… you might start wondering who the message was really for.
It starts with an old kind of magic.
Shortwave radio.
Before the internet, before satellites could put a live feed in your pocket, shortwave was how the world leaked into your home. With the right conditions, signals could skip across oceans, bouncing off the ionosphere like stones across black water. A voice from Europe could land in North America. A station in the Middle East could drift into a bedroom in the Midwest.
But shortwave also had a darker side.
Because the same physics that let people share news and music also allowed something else to travel invisibly, untraceably, and cheaply.
Instructions.
Orders.
Secrets.
If you’ve spent any time in the world of radio hobbyists, you’ve probably heard the phrase “numbers station.” It sounds harmless, almost academic. But the reality is stranger.
A numbers station is exactly what it sounds like. A broadcast, usually on shortwave, that transmits sequences of spoken numbers. Sometimes letters. Sometimes tones. Sometimes strange buzzing or bursts of digital noise. The transmissions often repeat. They don’t advertise. They don’t explain. They don’t sign off like normal stations.
They just appear… and vanish.
And the most unsettling detail is this.
They don’t seem to care if you hear them.
In fact, it can feel like they expect you to.
As if you’re allowed to listen… as long as you don’t understand.
The most widely accepted theory is also the simplest: espionage. Numbers stations are believed to be a way for intelligence agencies to communicate with agents in the field. One-way communication. The agent listens. Nobody transmits back. No direction-finding a sender from the receiver. No trail.
And if the agent has what’s called a one-time pad, a sheet of random numbers used only once, the code becomes essentially unbreakable. The broadcast can be heard by anyone, recorded by anyone… and still mean nothing to everyone except the person with the key.
That’s the genius of it.
And that’s the nightmare.

""",
  "voice_id": "default-ajvbelwdjdfgdpgmxa__my_voice1",
  "audio_config": {
    "audio_encoding": "MP3",
    "speaking_rate": 1
  },
  "temperature": 1.1,
  "model_id": "inworld-tts-1-max"
}

response = requests.post(url, json=payload, headers=headers)
print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")
response.raise_for_status()
result = response.json()
audio_content = base64.b64decode(result['audioContent'])

# Find available filename (avoid overwriting)
voice_id = payload["voice_id"]
filename = f"output_{voice_id}.mp3"
counter = 1
while os.path.exists(filename):
    filename = f"output_{voice_id}{counter}.mp3"
    counter += 1

with open(filename, "wb") as f:
    f.write(audio_content)
    
print(f"Audio saved to: {filename}")