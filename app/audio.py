import base64
import audioop
import numpy as np


def mulaw_to_float32(payload: str) -> np.ndarray:
    """Decode base64 mulaw 8 kHz → float32 16 kHz for Whisper / VAD."""
    mulaw_bytes = base64.b64decode(payload)
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    upsampled = np.interp(
        np.linspace(0, len(pcm) - 1, len(pcm) * 2),
        np.arange(len(pcm)),
        pcm,
    )
    return upsampled.astype(np.float32)


def float32_to_mulaw_b64(audio_24k: np.ndarray) -> str:
    """Convert float32 24 kHz PCM → base64 mulaw 8 kHz for Twilio."""
    audio_8k = audio_24k[::3]
    audio_8k = np.clip(audio_8k, -1.0, 1.0)
    audio_int16 = (audio_8k * 32767).astype(np.int16)
    mulaw = audioop.lin2ulaw(audio_int16.tobytes(), 2)
    return base64.b64encode(mulaw).decode("utf-8")
