import audioop
import base64

import numpy as np
import pytest

from app.audio import float32_to_mulaw_b64, mulaw_to_float32


def _make_mulaw_payload(num_samples: int = 160) -> str:
    """Create a valid base64 mulaw payload (silence) for testing."""
    pcm_silence = np.zeros(num_samples, dtype=np.int16)
    mulaw_bytes = audioop.lin2ulaw(pcm_silence.tobytes(), 2)
    return base64.b64encode(mulaw_bytes).decode("utf-8")


class TestMulawToFloat32:
    def test_returns_numpy_array(self):
        payload = _make_mulaw_payload(160)
        result = mulaw_to_float32(payload)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_is_float32(self):
        payload = _make_mulaw_payload(160)
        result = mulaw_to_float32(payload)
        assert result.dtype == np.float32

    def test_upsamples_8khz_to_16khz(self):
        num_mulaw_samples = 160
        payload = _make_mulaw_payload(num_mulaw_samples)
        result = mulaw_to_float32(payload)
        # 8 kHz → 16 kHz: output should be 2× the input sample count
        assert len(result) == num_mulaw_samples * 2

    def test_silence_decodes_near_zero(self):
        payload = _make_mulaw_payload(160)
        result = mulaw_to_float32(payload)
        assert np.max(np.abs(result)) < 0.1

    def test_values_clamped_to_minus_one_to_one(self):
        payload = _make_mulaw_payload(320)
        result = mulaw_to_float32(payload)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_handles_single_sample(self):
        payload = _make_mulaw_payload(1)
        result = mulaw_to_float32(payload)
        assert len(result) == 2

    def test_handles_large_payload(self):
        payload = _make_mulaw_payload(8000)  # 1 second at 8 kHz
        result = mulaw_to_float32(payload)
        assert len(result) == 16000  # 1 second at 16 kHz


class TestFloat32ToMulawB64:
    def test_returns_string(self):
        audio = np.zeros(240, dtype=np.float32)
        result = float32_to_mulaw_b64(audio)
        assert isinstance(result, str)

    def test_output_is_valid_base64(self):
        audio = np.zeros(240, dtype=np.float32)
        result = float32_to_mulaw_b64(audio)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_downsamples_24khz_to_8khz(self):
        num_samples = 2400  # 0.1 s at 24 kHz
        audio = np.zeros(num_samples, dtype=np.float32)
        result = float32_to_mulaw_b64(audio)
        # 24 kHz → 8 kHz: every 3rd sample kept → num_samples // 3
        decoded = base64.b64decode(result)
        assert len(decoded) == num_samples // 3

    def test_clips_values_above_one(self):
        audio = np.full(240, 2.0, dtype=np.float32)
        result = float32_to_mulaw_b64(audio)
        assert isinstance(result, str)

    def test_clips_values_below_minus_one(self):
        audio = np.full(240, -2.0, dtype=np.float32)
        result = float32_to_mulaw_b64(audio)
        assert isinstance(result, str)

    def test_roundtrip_preserves_silence(self):
        silence_24k = np.zeros(2400, dtype=np.float32)
        encoded = float32_to_mulaw_b64(silence_24k)
        decoded = mulaw_to_float32(encoded)
        assert np.max(np.abs(decoded)) < 0.1

    def test_empty_array(self):
        audio = np.zeros(0, dtype=np.float32)
        result = float32_to_mulaw_b64(audio)
        assert base64.b64decode(result) == b""
