# English Coaching Voice Agent

A real-time AI English conversation coach delivered over a phone call. Call +1 (405) 687-7983 to have a free-form spoken conversation with a GPT-powered coach and receive personalised spoken feedback on your English at the end of the session.
---

## How It Works

```
User calls Twilio number
        │
        ▼
Twilio fetches TwiML  ──►  POST /incoming-call
        │                  Returns <Connect><Stream> WebSocket URL
        ▼
Twilio opens WebSocket ──►  WS /audio-stream
        │
        │   mulaw 8 kHz audio chunks (20 ms each)
        ▼
 Silero VAD detects speech end
        │
        ▼
 OpenAI Whisper transcribes audio
        │
        ▼
 GPT-4o-mini generates a short coaching reply
        │
        ▼
 OpenAI TTS streams PCM audio back to Twilio
        │
        ▼  (after MAX_CONVERSATION_SECONDS)
 AutoGen feedback agent analyses full transcript
        │
        ▼
 Personalised feedback spoken over the call
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Telephony | Twilio Media Streams (WebSocket, mulaw 8 kHz) |
| Voice activity detection | Silero VAD (via `torch.hub`) |
| Transcription | OpenAI Whisper API (`whisper-1`) |
| Conversation LLM | OpenAI GPT-4o-mini |
| Text-to-speech | OpenAI TTS (`tts-1`, voice `nova`, PCM 24 kHz) |
| Feedback agent | AutoGen `AssistantAgent` + GPT-4o-mini |
| Audio conversion | `audioop` + NumPy (mulaw ↔ PCM ↔ float32) |

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py        # Environment variables, API clients, constants
│   ├── audio.py         # mulaw ↔ float32 audio conversion
│   ├── conversation.py  # ConversationManager + GPT streaming
│   ├── feedback.py      # AutoGen feedback agent
│   ├── session.py       # TwilioSession — VAD, transcription, TTS, main loop
│   └── main.py          # FastAPI routes: /health, /incoming-call, /audio-stream
├── .env                 # Local secrets (gitignored)
├── .env.example         # Template — copy to .env and fill in
├── .gitignore
├── requirements.txt     # Pinned dependencies
├── Dockerfile
├── Procfile             # For Railway / Heroku
├── railway.json         # Railway deployment config
├── render.yaml          # Render deployment config
└── test_call.py         # Trigger a test call via Twilio REST API
```

---

## Prerequisites

- Python 3.11+
- A [Twilio](https://twilio.com) account with a phone number that has Voice capability
- An [OpenAI](https://platform.openai.com) API key
- [ngrok](https://ngrok.com) for local testing

---

## Local Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd english-coaching-workflow
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
OPENAI_API_KEY=sk-...

TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
TO_PHONE_NUMBER=+1xxxxxxxxxx

SERVER_HOST=your-ngrok-subdomain.ngrok-free.app
PORT=8000

MAX_CALL_DURATION=3600
MAX_CONVERSATION_SECONDS=300
```

### 4. Start the server

```bash
python -m app.main
```

On startup the server will:
- Load the Silero VAD model (downloaded once, then cached)
- Pre-cache the greeting audio via OpenAI TTS
- Print `✅ Server ready`

### 5. Expose the server with ngrok

In a second terminal:

```bash
ngrok http 8000
```

Copy the `https://xxxx.ngrok-free.app` URL and set it as `SERVER_HOST` in `.env` (without `https://`). Restart the server so it picks up the new value.

### 6. Configure Twilio

In the [Twilio Console](https://console.twilio.com) → Phone Numbers → your number → Voice Configuration:

- **A call comes in** → Webhook → `https://your-ngrok-url/incoming-call` → HTTP POST

### 7. Run a test call

```bash
python test_call.py
```

Your phone will ring. Speak naturally — the agent will respond, and after the session limit you will hear personalised feedback spoken over the call.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `SERVER_HOST` | Yes | `your-server.com` | Public hostname (no `https://`, no trailing slash) |
| `PORT` | No | `8000` | Port the server listens on |
| `MAX_CONVERSATION_SECONDS` | No | `300` | Coaching session length in seconds before feedback |
| `MAX_CALL_DURATION` | No | `3600` | TwiML `<Pause>` fallback length in seconds |
| `TWILIO_ACCOUNT_SID` | test only | — | Needed by `test_call.py` |
| `TWILIO_AUTH_TOKEN` | test only | — | Needed by `test_call.py` |
| `TWILIO_PHONE_NUMBER` | test only | — | Caller ID for outbound test call |
| `TO_PHONE_NUMBER` | test only | — | Number to call in test |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/incoming-call` | Returns TwiML to connect Twilio Media Stream |
| `WS` | `/audio-stream` | WebSocket — handles the full call session |

---

## Session Flow (detail)

1. **Greeting** — Pre-cached TTS audio is sent the instant the WebSocket opens (zero API latency).
2. **VAD loop** — Incoming mulaw audio is upsampled 8 kHz → 16 kHz and fed to Silero VAD in a 50-chunk sliding window (≈ 1 second).
3. **Utterance detection** — After 0.8 s of silence following detected speech, the buffered audio is transcribed via Whisper.
4. **Reply** — GPT-4o-mini generates a short (1–2 sentence) response, streamed sentence-by-sentence to TTS and sent back to Twilio.
5. **Timer** — After `MAX_CONVERSATION_SECONDS`, the session flags `_time_up`. On the next media chunk, feedback is triggered.
6. **Feedback** — The AutoGen agent analyses the full transcript, covering grammar, vocabulary, fluency, and gives 3–5 personalised tips. The feedback is spoken over the call.

---

## Deployment

### Railway

1. Push your code to GitHub.
2. Create a new Railway project → **Deploy from GitHub repo**.
3. Railway detects `railway.json` and builds using the `Dockerfile`.
4. Set environment variables in Railway dashboard (see table above). Do **not** set `PORT` — Railway injects it automatically.
5. Copy your Railway public domain → set as `SERVER_HOST`.
6. Update the Twilio webhook to `https://<railway-domain>/incoming-call`.

### Render

1. Push your code to GitHub.
2. Create a new Render **Web Service** → connect your repo.
3. Render detects `render.yaml` automatically.
4. Set secret environment variables in the Render dashboard (`OPENAI_API_KEY`, `SERVER_HOST`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`). Do **not** set `PORT` — Render injects it.
5. Copy your Render public domain → set as `SERVER_HOST`.
6. Update the Twilio webhook to `https://<render-domain>/incoming-call`.

> **Note:** Render free tier spins down after inactivity. Use a paid plan or an uptime monitor (e.g. UptimeRobot hitting `/health` every 5 minutes) to prevent cold starts dropping the first call.

### Docker (self-hosted)

```bash
docker build -t english-coach .
docker run --env-file .env -p 8000:8000 english-coach
```

---

## Key Design Decisions

- **Greeting pre-caching** — The greeting is synthesised at startup and stored in memory. This eliminates the ~1 second TTS API latency that would otherwise cause the first words to be cut off.
- **Stale buffer reset** — While the agent is speaking, incoming audio chunks are discarded. This prevents the user's microphone from picking up the agent's voice and re-triggering VAD.
- **Feedback with live WebSocket drain** — During feedback generation, the main `iter_text()` loop keeps reading incoming Twilio messages. This prevents TCP backpressure that would cause Twilio to close the WebSocket before the feedback audio is fully delivered.
- **Keepalive marks** — A background task sends a WebSocket `mark` event every 25 seconds to prevent Twilio's idle WebSocket timeout.
