from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Audio
    sample_rate: int = int(os.getenv("VA_SAMPLE_RATE", "16000"))
    frame_ms: int = int(os.getenv("VA_FRAME_MS", "30"))  # must be 10/20/30 for webrtcvad
    vad_aggressiveness: int = int(os.getenv("VA_VAD_AGGR", "2"))  # 0..3
    speech_start_ratio: float = float(os.getenv("VA_SPEECH_START_RATIO", "0.6"))
    speech_end_ratio: float = float(os.getenv("VA_SPEECH_END_RATIO", "0.9"))
    vad_window_ms: int = int(os.getenv("VA_VAD_WINDOW_MS", "300"))  # ring buffer window
    max_utterance_s: float = float(os.getenv("VA_MAX_UTTERANCE_S", "15.0"))
    min_utterance_s: float = float(os.getenv("VA_MIN_UTTERANCE_S", "0.35"))

    # STT (whisper.cpp via pywhispercpp)
    whisper_model: str = os.getenv("VA_WHISPER_MODEL", "base")  # tiny/base/small/...
    whisper_language: str = os.getenv("VA_WHISPER_LANG", "ru")  # "ru" or "auto"
    whisper_threads: int = int(os.getenv("VA_WHISPER_THREADS", "6"))

    # LLM (LM Studio OpenAI-compat)
    lm_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_api_key: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    lm_model: str = os.getenv("LMSTUDIO_MODEL", "qwen3-0.6b-mlx")
    lm_temperature: float = float(os.getenv("VA_LM_TEMPERATURE", "0.2"))
    history_turns: int = int(os.getenv("VA_HISTORY_TURNS", "10"))

    # TTS (macOS say)
    tts_voice: str = os.getenv("VA_TTS_VOICE", "")  # empty => system default
    tts_rate: str = os.getenv("VA_TTS_RATE", "")    # empty => default

    # Misc
    cache_dir: str = os.getenv("VA_CACHE_DIR", ".cache/voice_assistant")
