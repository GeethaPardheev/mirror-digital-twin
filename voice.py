"""Local speech, free and on-device: Whisper (faster-whisper) hears you, Kokoro (ONNX) speaks."""
import io
import os
import re
import threading

import numpy as np
import soundfile as sf

STT_MODEL = os.getenv("MIRROR_STT", "small")  # "large-v3-turbo" is more accurate for Hinglish, slower
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
VOICES = {"mirror": os.getenv("MIRROR_VOICE", "af_heart"), "twin": os.getenv("MIRROR_TWIN_VOICE", "am_michael")}
_models, _lock = {}, threading.Lock()


def _whisper():
    if "stt" not in _models:
        from faster_whisper import WhisperModel
        _models["stt"] = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    return _models["stt"]


def _kokoro():
    if "tts" not in _models:
        from kokoro_onnx import Kokoro
        _models["tts"] = Kokoro(os.path.join(MODELS, "kokoro-v1.0.onnx"), os.path.join(MODELS, "voices-v1.0.bin"))
    return _models["tts"]


def transcribe(pcm):
    """pcm: mono float32 samples at 16 kHz."""
    if len(pcm) < 8000:  # under half a second
        return ""
    segments, _ = _whisper().transcribe(pcm, beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def speak(text, who="mirror", voice=""):
    """Returns WAV bytes. `voice` picks any Kokoro voice (e.g. bf_emma); otherwise the role's default."""
    text = re.sub(r"\[\d+\]|\*\*", "", text).strip()
    with _lock:
        tts = _kokoro()
        name = voice if voice in tts.get_voices() else VOICES.get(who, VOICES["mirror"])
        samples, rate = tts.create(text, voice=name, speed=1.0, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV")
    return buf.getvalue()


def warm_up():
    try:
        _whisper().transcribe(np.zeros(16000, dtype=np.float32))
        speak("Hi.")
    except Exception as e:  # voice is optional; text still works
        print(f"[voice] not available yet: {e}")
