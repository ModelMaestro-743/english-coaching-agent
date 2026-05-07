import asyncio
import os
import sys
import time
import tempfile
import json
import base64
import audioop
import numpy as np
from scipy.io.wavfile import write as wav_write
from dotenv import load_dotenv
import torch
from openai import AsyncOpenAI
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import uvicorn

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_client = OpenAIChatCompletionClient(
    model='gpt-4o-mini',
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

COACH_SYSTEM_PROMPT = """You are a friendly English conversation coach.
Keep responses SHORT (1-2 sentences max) and conversational.
Ask simple follow-up questions to keep the conversation flowing.
Gently correct major mistakes but don't overwhelm the user.
Be encouraging and supportive.
Focus on natural conversation rather than formal lessons."""

# ========== Global Models (loaded once at startup) ==========
vad_model = None
get_speech_timestamps = None

_greeting_payload: str = None
_greeting_duration: float = 0.0
GREETING_TEXT = (
    "Hi! Let's start our English conversation practice. "
    "Tell me something about your day."
)

def load_models():
    global vad_model, get_speech_timestamps
    print("🔄 Loading VAD model...")
    _model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    vad_model = _model
    (get_speech_timestamps, _, _, _, _) = utils
    print("✅ VAD model loaded")

async def cache_greeting():
    global _greeting_payload, _greeting_duration
    print("🔄 Pre-caching greeting audio...")
    chunks = []
    async with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=GREETING_TEXT,
        response_format="pcm"
    ) as response:
        async for chunk in response.iter_bytes(chunk_size=4096):
            chunks.append(chunk)
    audio_24k = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    _greeting_payload = float32_to_mulaw_b64(audio_24k)
    _greeting_duration = len(audio_24k) / TTS_RATE
    print("✅ Greeting cached")

# ========== Audio Format Conversion ==========
TWILIO_RATE = 8000
WHISPER_RATE = 16000
TTS_RATE = 24000

def mulaw_to_float32(payload: str) -> np.ndarray:
    """Decode base64 mulaw 8kHz → float32 16kHz for Whisper/VAD."""
    mulaw_bytes = base64.b64decode(payload)
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)          # mulaw → int16
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # Upsample 8kHz → 16kHz by linear interpolation
    upsampled = np.interp(
        np.linspace(0, len(pcm) - 1, len(pcm) * 2),
        np.arange(len(pcm)),
        pcm
    )
    return upsampled.astype(np.float32)

def float32_to_mulaw_b64(audio_24k: np.ndarray) -> str:
    """Convert float32 24kHz PCM → base64 mulaw 8kHz for Twilio."""
    # Downsample 24kHz → 8kHz (keep every 3rd sample)
    audio_8k = audio_24k[::3]
    audio_8k = np.clip(audio_8k, -1.0, 1.0)
    audio_int16 = (audio_8k * 32767).astype(np.int16)
    mulaw = audioop.lin2ulaw(audio_int16.tobytes(), 2)
    return base64.b64encode(mulaw).decode('utf-8')

# ========== Conversation History Manager ==========
class ConversationManager:
    def __init__(self, max_recent_context=6):
        self.full_history = []
        self.recent_history = []
        self.max_recent_context = max_recent_context
        self.start_time = time.time()

    def add_message(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.full_history.append(message)
        self.recent_history.append(message)
        if len(self.recent_history) > self.max_recent_context:
            self.recent_history = self.recent_history[-self.max_recent_context:]

    def get_full_transcript(self) -> str:
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.full_history])

    def get_conversation_stats(self) -> dict:
        user_msgs = [m for m in self.full_history if m['role'] == 'user']
        return {
            'total_exchanges': len(user_msgs),
            'user_word_count': sum(len(m['content'].split()) for m in user_msgs),
            'total_conversation_time': int(time.time() - self.start_time)
        }

# ========== Streaming LLM Response ==========
async def stream_coach_response(messages):
    """Stream LLM tokens and yield complete sentences."""
    stream = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=messages,
        max_tokens=100,
        stream=True
    )
    buffer = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        while True:
            earliest = -1
            for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                idx = buffer.find(punct)
                if idx != -1 and (earliest == -1 or idx < earliest):
                    earliest = idx + len(punct)
            if earliest != -1:
                sentence = buffer[:earliest].strip()
                buffer = buffer[earliest:]
                if sentence:
                    yield sentence
            else:
                break
    if buffer.strip():
        yield buffer.strip()

# ========== Per-Call Session Handler ==========
class TwilioSession:
    CHUNK_DURATION = 0.02       # Twilio sends 20ms chunks
    END_SILENCE = 0.8           # Seconds of silence to end utterance
    MAX_SILENCE = 3.0           # Seconds before giving up with no speech
    NOISE_THRESHOLD = 0.008
    VAD_WINDOW = 50             # ~1 second of 20ms chunks
    MAX_CONVERSATION_SECONDS = 300

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.stream_sid = None
        self.conversation_manager = ConversationManager()

        # Audio state
        self.audio_buffer = []      # Accumulated 16kHz float32 audio
        self.recent_chunks = []     # Last ~1s of chunks for VAD
        self.speech_detected = False
        self.silence_time = 0.0
        self.speech_time = 0.0
        self.is_speaking = False    # True while TTS is playing
        self._time_up = False
        self._feedback_done = False

    # ---- VAD ----
    def _has_speech(self, audio: np.ndarray) -> bool:
        if len(audio) < WHISPER_RATE * 0.1:
            return False
        try:
            tensor = torch.from_numpy(audio).unsqueeze(0)
            stamps = get_speech_timestamps(
                tensor, vad_model,
                sampling_rate=WHISPER_RATE,
                threshold=0.4,
                min_speech_duration_ms=200,
                min_silence_duration_ms=50
            )
            return len(stamps) > 0
        except Exception:
            return np.mean(np.abs(audio)) > self.NOISE_THRESHOLD * 2

    # ---- Transcription ----
    async def _transcribe(self) -> str:
        if not self.audio_buffer:
            return ""
        audio = np.concatenate(self.audio_buffer)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        audio_int16 = np.int16(audio * 32767)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_write(f.name, WHISPER_RATE, audio_int16)
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as audio_file:
                result = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return result.text.strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ---- TTS + Send ----
    async def speak(self, text: str):
        if not text.strip():
            return
        print(f"🔊 Speaking: {text[:60]}...")
        self.is_speaking = True
        total_samples = 0
        leftover = b""
        try:
            async with openai_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=text,
                response_format="pcm"
            ) as response:
                async for raw_chunk in response.iter_bytes(chunk_size=4096):
                    data = leftover + raw_chunk
                    # Keep only complete int16 samples (2 bytes each)
                    remainder = len(data) % 2
                    leftover = data[-remainder:] if remainder else b""
                    data = data[:-remainder] if remainder else data
                    if not data:
                        continue
                    audio_24k = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    total_samples += len(audio_24k)
                    await self.ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": float32_to_mulaw_b64(audio_24k)}
                    }))

            # Wait for estimated playback to finish before listening again
            await asyncio.sleep(total_samples / TTS_RATE + 0.2)

        except WebSocketDisconnect:
            raise
        except Exception as e:
            print(f"❌ TTS error: {type(e).__name__}: {e}")
        finally:
            self.is_speaking = False

    # ---- LLM ----
    async def _get_response(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            *self.conversation_manager.recent_history,
            {"role": "user", "content": user_input}
        ]
        parts = []
        async for sentence in stream_coach_response(messages):
            parts.append(sentence)
        return " ".join(parts)

    # ---- Reset utterance state ----
    def _reset_utterance(self):
        self.audio_buffer = []
        self.recent_chunks = []
        self.speech_detected = False
        self.silence_time = 0.0
        self.speech_time = 0.0

    # ---- Process completed utterance ----
    async def _process_utterance(self):
        print("🔄 Processing utterance...")
        text = await self._transcribe()

        if not text or len(text.strip()) < 3:
            await self.speak("I didn't catch that. Could you say it again?")
            return

        print(f"📝 User: {text}")
        t0 = time.time()
        response = await self._get_response(text)
        print(f"⏱️ LLM: {time.time()-t0:.1f}s | Reply: {response}")

        self.conversation_manager.add_message("user", text)
        self.conversation_manager.add_message("assistant", response)
        await self.speak(response)

    # ---- End-of-session feedback ----
    async def _generate_feedback(self):
        if self._feedback_done:
            return
        self._feedback_done = True

        if not self.conversation_manager.full_history:
            print("ℹ️ No conversation history — skipping feedback")
            return

        # Speak intro immediately while connection is still alive
        try:
            await self.speak(
                "Great session! I'm preparing your personalised feedback now. "
                "Please hold on for just a moment."
            )
        except Exception:
            pass  # Connection may already be closing — continue to generate anyway

        stats = self.conversation_manager.get_conversation_stats()
        transcript = self.conversation_manager.get_full_transcript()

        print("🔄 Generating feedback...")
        feedback_agent = AssistantAgent(
            name="FeedbackAgent",
            model_client=model_client,
            system_message=f"""You are a warm and encouraging English coach giving direct feedback to a student after their conversation practice session.

You are speaking DIRECTLY to the student. Use "you" and "your" throughout — never refer to them as "the user" or in third person.
Only analyse the student's lines from the transcript — ignore the coach's lines completely.

Address these areas in flowing paragraphs (no bullet points or symbols):

Grammar and Sentence Structure: Point out patterns in how you constructed sentences — what you did well and where you can tighten things up. Give specific examples from what you said.

Vocabulary and Expression: Comment on the words and phrases you chose. Highlight good choices and suggest stronger or more varied alternatives where you repeated yourself or were unclear.

Fluency and Coherence: Reflect on how naturally and clearly your ideas flowed. Note where you expressed yourself confidently and where you could be more direct.

Personalized Tips: Give 3 to 5 specific, actionable things you can practise based on what came up in this conversation.

Session stats: You spoke {stats['user_word_count']} words across {stats['total_exchanges']} exchanges in about {stats['total_conversation_time']//60} minutes.

Be honest, specific, and encouraging. Speak as if you are sitting across from the student.

FULL TRANSCRIPT (read both sides for context, but only evaluate the lines marked "User:" when giving feedback on English):
{transcript}
"""
        )
        try:
            result = await feedback_agent.run(
                task="Give me direct, personalised feedback on my English based on what I said in the conversation."
            )
            feedback = result.messages[-1].content
            print("\n" + "="*60)
            print("📊 FEEDBACK")
            print("="*60)
            print(feedback)
            print("="*60)
        except Exception as e:
            print(f"❌ Feedback generation error: {e}")
            feedback = None

        try:
            if feedback:
                await self.speak(feedback)
            else:
                await self.speak("You did a great job today! Keep practising.")
            await asyncio.sleep(2)  # buffer so Twilio finishes playing before connection closes
        except Exception:
            print("ℹ️ Connection closed before feedback could be spoken — see console above")

    # ---- Drain incoming messages silently (prevents TCP backpressure close) ----
    async def _drain_incoming(self):
        try:
            while True:
                await self.ws.receive_text()
        except Exception:
            pass

    # ---- Session timer: signal time-up after MAX_CONVERSATION_SECONDS ----
    async def _session_timer(self):
        await asyncio.sleep(self.MAX_CONVERSATION_SECONDS)
        print(f"⏰ {self.MAX_CONVERSATION_SECONDS}s limit reached — wrapping up session")
        self._time_up = True

    # ---- Keep-alive: send a mark event every 25s to prevent Twilio idle timeout ----
    async def _keepalive(self):
        while True:
            await asyncio.sleep(25)
            if not self.stream_sid:
                continue
            try:
                await self.ws.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "keepalive"}
                }))
            except Exception:
                break

    # ---- Main session loop ----
    async def run(self):
        # Wait for Twilio's start event to get stream_sid before sending any audio
        async for raw in self.ws.iter_text():
            data = json.loads(raw)
            if data.get("event") == "start":
                self.stream_sid = data["start"]["streamSid"]
                print(f"📞 Call started: {self.stream_sid}")
                break

        # Send pre-cached greeting instantly — no API latency
        if _greeting_payload:
            self.is_speaking = True
            await self.ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": _greeting_payload}
            }))
            await asyncio.sleep(_greeting_duration + 0.2)
            self.is_speaking = False
            self._reset_utterance()
        else:
            await self.speak(GREETING_TEXT)

        keepalive = asyncio.create_task(self._keepalive())
        timer = asyncio.create_task(self._session_timer())
        try:
            async for raw in self.ws.iter_text():
                try:
                    data = json.loads(raw)
                    event = data.get("event")

                    if event == "media":
                        chunk = mulaw_to_float32(data["media"]["payload"])

                        if self.is_speaking:
                            self._reset_utterance()  # keep buffers clear so no stale audio triggers after speaking
                            continue

                        # Time is up — drain incoming chunks while speaking feedback
                        # so Twilio backpressure doesn't close the connection mid-speech
                        if self._time_up:
                            drain = asyncio.create_task(self._drain_incoming())
                            await self._generate_feedback()
                            drain.cancel()
                            break

                        self.audio_buffer.append(chunk)
                        self.recent_chunks.append(chunk)
                        if len(self.recent_chunks) > self.VAD_WINDOW:
                            self.recent_chunks.pop(0)

                        if len(self.recent_chunks) >= self.VAD_WINDOW:
                            window = np.concatenate(self.recent_chunks)
                            has_speech = self._has_speech(window)

                            if has_speech:
                                self.speech_detected = True
                                self.silence_time = 0.0
                                self.speech_time += self.CHUNK_DURATION
                            else:
                                self.speech_time = 0.0
                                self.silence_time += self.CHUNK_DURATION

                            if self.speech_detected and self.silence_time >= self.END_SILENCE:
                                await self._process_utterance()
                                self._reset_utterance()

                            elif not self.speech_detected and self.silence_time >= self.MAX_SILENCE:
                                self._reset_utterance()

                    elif event == "stop":
                        print("📞 Call ended by Twilio")
                        break  # finally block handles feedback

                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    print(f"❌ Session error: {e}")
        finally:
            timer.cancel()
            await self._generate_feedback()  # keepalive stays alive during feedback
            keepalive.cancel()             # cancel only after all audio is sent

# ========== FastAPI App ==========
app = FastAPI()

@app.on_event("startup")
async def startup():
    load_models()
    await cache_greeting()
    print("✅ Server ready")

@app.post("/incoming-call")
async def incoming_call():
    """Twilio calls this when a call comes in. Returns TwiML to open WebSocket stream."""
    host = os.getenv("SERVER_HOST", "your-server.com")
    max_call_duration = int(os.getenv("MAX_CALL_DURATION", "3600"))
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/audio-stream" />
    </Connect>
    <Pause length="{max_call_duration}"/>
</Response>"""
    return PlainTextResponse(xml, media_type="text/xml")

@app.websocket("/audio-stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    session = TwilioSession(websocket)
    try:
        await session.run()
    except WebSocketDisconnect as e:
        print(f"📞 WebSocket disconnected (code={e.code})")
        await session._generate_feedback()
    except Exception as e:
        print(f"❌ WebSocket error: {type(e).__name__}: {e}")
        await session._generate_feedback()

# ========== Entry Point ==========
if __name__ == "__main__":
    print("🚀 Starting English Coaching WebSocket server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
