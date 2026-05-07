import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# ── API clients ───────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=OPENAI_API_KEY,
)

# ── Server ────────────────────────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "your-server.com")
PORT: int = int(os.getenv("PORT", "8000"))
MAX_CALL_DURATION: int = int(os.getenv("MAX_CALL_DURATION", "3600"))
MAX_CONVERSATION_SECONDS: int = int(os.getenv("MAX_CONVERSATION_SECONDS", "300"))

# ── Audio sample rates ────────────────────────────────────────────────────────
TWILIO_RATE: int = 8000
WHISPER_RATE: int = 16000
TTS_RATE: int = 24000

# ── Prompts / copy ───────────────────────────────────────────────────────────
GREETING_TEXT: str = (
    "Hi! Let's start our English conversation practice. "
    "Tell me something about your day."
)

COACH_SYSTEM_PROMPT: str = """You are a friendly English conversation coach.
Keep responses SHORT (1-2 sentences max) and conversational.
Ask simple follow-up questions to keep the conversation flowing.
Gently correct major mistakes but don't overwhelm the user.
Be encouraging and supportive.
Focus on natural conversation rather than formal lessons."""
