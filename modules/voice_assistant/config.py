from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # Audio
    sample_rate: int = int(os.getenv("VA_SAMPLE_RATE", "16000"))
    frame_ms: int = int(os.getenv("VA_FRAME_MS", "30"))  # must be 10/20/30 for webrtcvad
    input_device: str = os.getenv("VA_INPUT_DEVICE", "")  # index or substring
    vad_aggressiveness: int = int(os.getenv("VA_VAD_AGGR", "2"))  # 0..3
    speech_start_ratio: float = float(os.getenv("VA_SPEECH_START_RATIO", "0.6"))
    speech_end_ratio: float = float(os.getenv("VA_SPEECH_END_RATIO", "0.8"))
    vad_window_ms: int = int(os.getenv("VA_VAD_WINDOW_MS", "300"))  # ring buffer window
    max_utterance_s: float = float(os.getenv("VA_MAX_UTTERANCE_S", "15.0"))
    min_utterance_s: float = float(os.getenv("VA_MIN_UTTERANCE_S", "0.35"))
    min_text_chars: int = int(os.getenv("VA_MIN_TEXT_CHARS", "2"))
    duplicate_utt_window_s: float = float(os.getenv("VA_DUPLICATE_UTT_WINDOW_S", "2.2"))
    allow_barge_in: bool = _env_bool("VA_ALLOW_BARGE_IN", False)
    tts_echo_suppress_ms: int = int(os.getenv("VA_TTS_ECHO_SUPPRESS_MS", "900"))

    # STT (whisper.cpp via pywhispercpp)
    whisper_model: str = os.getenv("VA_WHISPER_MODEL", "medium")  # tiny/base/small/...
    whisper_language: str = os.getenv("VA_WHISPER_LANG", "ru")  # "ru" or "auto"
    whisper_threads: int = int(os.getenv("VA_WHISPER_THREADS", "6"))
    stt_save_utterances: bool = _env_bool("VA_STT_SAVE_UTTERANCES", False)
    stt_drop_noise_tags: bool = _env_bool("VA_STT_DROP_NOISE_TAGS", True)
    stt_normalize_tech_terms: bool = _env_bool("VA_STT_NORMALIZE_TECH_TERMS", True)
    stt_no_context: bool = _env_bool("VA_STT_NO_CONTEXT", True)
    stt_suppress_non_speech_tokens: bool = _env_bool("VA_STT_SUPPRESS_NON_SPEECH_TOKENS", True)
    stt_no_speech_thold: float = float(os.getenv("VA_STT_NO_SPEECH_THOLD", "0.7"))
    stt_initial_prompt: str = os.getenv(
        "VA_STT_INITIAL_PROMPT",
        "Русская речь. Технические термины: PHP, JavaScript, C++, C#, SQL.",
    )
    stt_drop_subtitle_hallucinations: bool = _env_bool(
        "VA_STT_DROP_SUBTITLE_HALLUCINATIONS",
        True,
    )

    # LLM
    lm_backend: str = os.getenv("VA_LLM_BACKEND", "local")  # auto | local | lmstudio
    lm_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_api_key: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    lm_model: str = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")
    lm_local_model_path: str = os.getenv(
        "VA_LLM_MODEL_PATH",
        ".cache/voice_assistant/models/llm/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )
    lm_local_model_url: str = os.getenv(
        "VA_LLM_MODEL_URL",
        "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )
    lm_local_ctx: int = int(os.getenv("VA_LLM_CTX", "4096"))
    lm_local_threads: int = int(os.getenv("VA_LLM_THREADS", "0"))  # 0 => auto
    lm_local_gpu_layers: int = int(os.getenv("VA_LLM_GPU_LAYERS", "-1"))  # -1 => all
    lm_local_max_tokens: int = int(os.getenv("VA_LLM_MAX_TOKENS", "384"))
    lm_local_use_mmap: bool = _env_bool("VA_LLM_USE_MMAP", True)
    lm_local_use_mlock: bool = _env_bool("VA_LLM_USE_MLOCK", False)
    lm_temperature: float = float(os.getenv("VA_LM_TEMPERATURE", "0.35"))
    lm_timeout_s: float = float(os.getenv("VA_LM_TIMEOUT_S", "45"))
    history_turns: int = int(os.getenv("VA_HISTORY_TURNS", "10"))

    # TTS (macOS say)
    tts_backend: str = os.getenv("VA_TTS_BACKEND", "piper")  # say | piper
    tts_lang: str = os.getenv("VA_TTS_LANG", "ru")
    tts_voice: str = os.getenv("VA_TTS_VOICE", "")  # empty => system default
    tts_rate: str = os.getenv("VA_TTS_RATE", "")    # empty => default
    tts_strip_emoji: bool = _env_bool("VA_TTS_STRIP_EMOJI", True)
    tts_strip_markdown: bool = _env_bool("VA_TTS_STRIP_MARKDOWN", True)
    tts_max_chars: int = int(os.getenv("VA_TTS_MAX_CHARS", "420"))
    tts_piper_bin: str = os.getenv("VA_TTS_PIPER_BIN", "piper")
    tts_piper_model: str = os.getenv(
        "VA_TTS_PIPER_MODEL",
        ".cache/voice_assistant/models/piper/ru_RU-irina-medium.onnx",
    )
    tts_piper_model_url: str = os.getenv(
        "VA_TTS_PIPER_MODEL_URL",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    )
    tts_piper_config_url: str = os.getenv("VA_TTS_PIPER_CONFIG_URL", "")

    # Misc
    cache_dir: str = os.getenv("VA_CACHE_DIR", ".cache/voice_assistant")
