import asyncio
import json
import os
import tempfile
import time

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from scipy.io.wavfile import write as wav_write

from .audio import float32_to_mulaw_b64, mulaw_to_float32
from .config import (
    COACH_SYSTEM_PROMPT,
    GREETING_TEXT,
    MAX_CONVERSATION_SECONDS,
    TTS_RATE,
    WHISPER_RATE,
    openai_client,
)
from .conversation import ConversationManager, stream_coach_response
from .feedback import run_feedback_agent

# ── Global models (loaded once at startup) ────────────────────────────────────
vad_model = None
get_speech_timestamps = None
_greeting_payload: str | None = None
_greeting_duration: float = 0.0


def load_models() -> None:
    global vad_model, get_speech_timestamps
    print("🔄 Loading VAD model...")
    _model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    vad_model = _model
    (get_speech_timestamps, _, _, _, _) = utils
    print("✅ VAD model loaded")


async def cache_greeting() -> None:
    global _greeting_payload, _greeting_duration
    print("🔄 Pre-caching greeting audio...")
    chunks: list[bytes] = []
    async with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=GREETING_TEXT,
        response_format="pcm",
    ) as response:
        async for chunk in response.iter_bytes(chunk_size=4096):
            chunks.append(chunk)
    audio_24k = (
        np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    )
    _greeting_payload = float32_to_mulaw_b64(audio_24k)
    _greeting_duration = len(audio_24k) / TTS_RATE
    print("✅ Greeting cached")


# ── Per-call session ──────────────────────────────────────────────────────────
class TwilioSession:
    CHUNK_DURATION: float = 0.02
    END_SILENCE: float = 0.8
    MAX_SILENCE: float = 3.0
    NOISE_THRESHOLD: float = 0.008
    VAD_WINDOW: int = 50

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.stream_sid: str | None = None
        self.conversation_manager = ConversationManager()

        self.audio_buffer: list[np.ndarray] = []
        self.recent_chunks: list[np.ndarray] = []
        self.speech_detected: bool = False
        self.silence_time: float = 0.0
        self.speech_time: float = 0.0
        self.is_speaking: bool = False
        self._time_up: bool = False
        self._feedback_done: bool = False

    # ── VAD ──────────────────────────────────────────────────────────────────

    def _has_speech(self, audio: np.ndarray) -> bool:
        if len(audio) < WHISPER_RATE * 0.1:
            return False
        try:
            tensor = torch.from_numpy(audio).unsqueeze(0)
            stamps = get_speech_timestamps(
                tensor,
                vad_model,
                sampling_rate=WHISPER_RATE,
                threshold=0.4,
                min_speech_duration_ms=200,
                min_silence_duration_ms=50,
            )
            return len(stamps) > 0
        except Exception:
            return np.mean(np.abs(audio)) > self.NOISE_THRESHOLD * 2

    # ── Transcription ─────────────────────────────────────────────────────────

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
                    language="en",
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

    # ── TTS + send ────────────────────────────────────────────────────────────

    async def speak(self, text: str) -> None:
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
                response_format="pcm",
            ) as response:
                async for raw_chunk in response.iter_bytes(chunk_size=4096):
                    data = leftover + raw_chunk
                    remainder = len(data) % 2
                    leftover = data[-remainder:] if remainder else b""
                    data = data[:-remainder] if remainder else data
                    if not data:
                        continue
                    audio_24k = (
                        np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    total_samples += len(audio_24k)
                    await self.ws.send_text(
                        json.dumps({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": float32_to_mulaw_b64(audio_24k)},
                        })
                    )
            await asyncio.sleep(total_samples / TTS_RATE + 0.2)
        except WebSocketDisconnect:
            raise
        except Exception as e:
            print(f"❌ TTS error: {type(e).__name__}: {e}")
        finally:
            self.is_speaking = False

    # ── LLM ──────────────────────────────────────────────────────────────────

    async def _get_response(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            *self.conversation_manager.recent_history,
            {"role": "user", "content": user_input},
        ]
        parts: list[str] = []
        async for sentence in stream_coach_response(messages):
            parts.append(sentence)
        return " ".join(parts)

    # ── Utterance helpers ─────────────────────────────────────────────────────

    def _reset_utterance(self) -> None:
        self.audio_buffer = []
        self.recent_chunks = []
        self.speech_detected = False
        self.silence_time = 0.0
        self.speech_time = 0.0

    async def _process_utterance(self) -> None:
        print("🔄 Processing utterance...")
        text = await self._transcribe()
        if not text or len(text.strip()) < 3:
            await self.speak("I didn't catch that. Could you say it again?")
            return
        print(f"📝 User: {text}")
        t0 = time.time()
        response = await self._get_response(text)
        print(f"⏱️ LLM: {time.time() - t0:.1f}s | Reply: {response}")
        self.conversation_manager.add_message("user", text)
        self.conversation_manager.add_message("assistant", response)
        await self.speak(response)

    # ── Feedback ──────────────────────────────────────────────────────────────

    async def _generate_feedback(self) -> None:
        if self._feedback_done:
            return
        self._feedback_done = True

        if not self.conversation_manager.full_history:
            print("ℹ️ No conversation history — skipping feedback")
            return

        try:
            await self.speak(
                "Great session! I'm preparing your personalised feedback now. "
                "Please hold on for just a moment."
            )
        except Exception:
            pass

        feedback = await run_feedback_agent(self.conversation_manager)

        try:
            await self.speak(feedback if feedback else "You did a great job today! Keep practising.")
            await asyncio.sleep(2)
        except Exception:
            print("ℹ️ Connection closed before feedback could be spoken — see console above")

    # ── Background tasks ──────────────────────────────────────────────────────

    async def _session_timer(self) -> None:
        await asyncio.sleep(MAX_CONVERSATION_SECONDS)
        print(f"⏰ {MAX_CONVERSATION_SECONDS}s limit reached — wrapping up session")
        self._time_up = True

    async def _keepalive(self) -> None:
        while True:
            await asyncio.sleep(25)
            if not self.stream_sid:
                continue
            try:
                await self.ws.send_text(
                    json.dumps({
                        "event": "mark",
                        "streamSid": self.stream_sid,
                        "mark": {"name": "keepalive"},
                    })
                )
            except Exception:
                break

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
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
            await self.ws.send_text(
                json.dumps({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": _greeting_payload},
                })
            )
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
                            self._reset_utterance()
                            continue

                        if self._time_up:
                            feedback_task = asyncio.create_task(self._generate_feedback())
                            # Keep reading so starlette never sees an idle WebSocket
                            # and Twilio never hits TCP backpressure while we speak feedback
                            async for drain_raw in self.ws.iter_text():
                                try:
                                    d = json.loads(drain_raw)
                                    if d.get("event") == "stop":
                                        print("📞 Call ended by Twilio")
                                        break
                                except Exception:
                                    pass
                                if feedback_task.done():
                                    break
                            try:
                                await feedback_task
                            except Exception:
                                pass
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
                        break

                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    print(f"❌ Session error: {e}")
        finally:
            timer.cancel()
            await self._generate_feedback()
            keepalive.cancel()
