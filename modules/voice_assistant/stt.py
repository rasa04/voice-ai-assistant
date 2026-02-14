from __future__ import annotations

import os
import re
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from pywhispercpp.model import Model


_NOISE_TAG_RE = re.compile(r"^\s*[\[(]\s*([^\]\)]+)\s*[\])]\s*$")
_ASTERISK_TAG_RE = re.compile(r"^\s*\*+\s*([^*]+?)\s*\*+\s*$")
_NOISE_WORDS = {
    "музыка",
    "music",
    "шум",
    "noise",
    "смех",
    "клик",
    "click",
    "аплодисменты",
    "applause",
    "laughter",
}

_SUBTITLE_HALLUCINATION_RE = re.compile(
    r"\b(редактор\s+субтитров|корректор)\b",
    re.IGNORECASE,
)

_TECH_TERM_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(пи\s*эйч\s*пи|п[еэ]\s*ха\s*п[еэ]|п[иы]эйчпи|пэйчпи|пэичпи|пхп|php)\b",
            re.IGNORECASE,
        ),
        "PHP",
    ),
    (re.compile(r"\b(джава\s*скрипт|ява\s*скрипт|жс|джейс)\b", re.IGNORECASE), "JavaScript"),
    (re.compile(r"\b(си\s*плюс\s*плюс|c\s*плюс\s*плюс)\b", re.IGNORECASE), "C++"),
    (re.compile(r"\b(си\s*шарп|c\s*sharp|c\s*шарп)\b", re.IGNORECASE), "C#"),
    (re.compile(r"\b(эс\s*кью\s*эл|сиквел)\b", re.IGNORECASE), "SQL"),
)

_WHISPER_PROFILES: dict[str, dict[str, str]] = {
    "tiny": {
        "params": "~39M",
        "disk": "~75 MB",
        "speed": "очень быстро",
        "quality": "низкая",
    },
    "base": {
        "params": "~74M",
        "disk": "~145 MB",
        "speed": "быстро",
        "quality": "базовая",
    },
    "small": {
        "params": "~244M",
        "disk": "~465 MB",
        "speed": "средняя",
        "quality": "хорошая",
    },
    "medium": {
        "params": "~769M",
        "disk": "~1.5 GB",
        "speed": "медленнее",
        "quality": "очень хорошая",
    },
    "large-v2": {
        "params": "~1550M",
        "disk": "~2.9 GB",
        "speed": "медленно",
        "quality": "максимальная",
    },
    "large-v3": {
        "params": "~1550M",
        "disk": "~2.9 GB",
        "speed": "медленно",
        "quality": "максимальная",
    },
}


def get_whisper_profile(model_name: str) -> Optional[dict[str, str]]:
    name = (model_name or "").strip().lower()
    if not name:
        return None

    # try exact, then fuzzy (for custom paths like ggml-small.bin)
    if name in _WHISPER_PROFILES:
        return _WHISPER_PROFILES[name]

    for key in sorted(_WHISPER_PROFILES.keys(), key=len, reverse=True):
        if key in name:
            return _WHISPER_PROFILES[key]
    return None


@dataclass
class STTConfig:
    cache_dir: str
    sample_rate: int
    model_name: str
    language: str
    n_threads: int
    save_utterances: bool
    drop_noise_tags: bool
    normalize_tech_terms: bool
    no_context: bool
    suppress_non_speech_tokens: bool
    no_speech_thold: float
    initial_prompt: str
    drop_subtitle_hallucinations: bool


class WhisperCppSTT:
    """
    Offline STT using whisper.cpp via pywhispercpp.
    pywhispercpp can auto-download ggml models into cache. 
    """
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._unsupported_params: set[str] = set()
        os.makedirs(cfg.cache_dir, exist_ok=True)
        if cfg.save_utterances:
            os.makedirs(os.path.join(cfg.cache_dir, "utterances"), exist_ok=True)

        # Note: pywhispercpp will download the ggml model automatically if you pass a known name
        # (e.g., "base", "small", ...). 
        self.model = Model(
            cfg.model_name,
            n_threads=cfg.n_threads,
            print_progress=False,
            print_realtime=False,
        )

    def transcribe_pcm16(self, pcm16: bytes) -> str:
        media: str | np.ndarray
        transient_wav_path = ""

        if self.cfg.save_utterances:
            media = self._write_wav(pcm16, persist=True)
        else:
            media = self._pcm16_to_float32(pcm16)

        try:
            segments = self._transcribe_with_compat(media)
        except Exception:
            if isinstance(media, str):
                raise

            # Fallback to WAV path mode if backend rejects ndarray for any reason.
            transient_wav_path = self._write_wav(pcm16, persist=False)
            segments = self._transcribe_with_compat(transient_wav_path)
        finally:
            if transient_wav_path:
                try:
                    os.remove(transient_wav_path)
                except OSError:
                    pass

        text = " ".join((s.text or "").strip() for s in segments).strip()
        return self._cleanup_text(
            text,
            drop_noise_tags=self.cfg.drop_noise_tags,
            normalize_tech_terms=self.cfg.normalize_tech_terms,
            drop_subtitle_hallucinations=self.cfg.drop_subtitle_hallucinations,
        )

    def _write_wav(self, pcm16: bytes, *, persist: bool) -> str:
        if persist:
            fn = f"utt_{int(time.time() * 1000)}.wav"
            path = os.path.join(self.cfg.cache_dir, "utterances", fn)
        else:
            fd, path = tempfile.mkstemp(prefix="utt_", suffix=".wav", dir=self.cfg.cache_dir)
            os.close(fd)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm16)

        return path

    @staticmethod
    def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    @staticmethod
    def _cleanup_text(
        text: str,
        *,
        drop_noise_tags: bool,
        normalize_tech_terms: bool,
        drop_subtitle_hallucinations: bool,
    ) -> str:
        # a bit of normalization for voice use
        text = text.replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")

        if drop_noise_tags and text:
            m = _NOISE_TAG_RE.match(text) or _ASTERISK_TAG_RE.match(text)
            token = (m.group(1) if m else text).strip().lower()
            if token in _NOISE_WORDS:
                return ""

        if normalize_tech_terms and text:
            for pattern, repl in _TECH_TERM_RULES:
                text = pattern.sub(repl, text)
            text = text.replace("  ", " ").strip()

        if drop_subtitle_hallucinations and _SUBTITLE_HALLUCINATION_RE.search(text):
            return ""

        return text

    def _transcribe_with_compat(self, media: str | np.ndarray):
        params: dict[str, Any] = {
            "language": self.cfg.language,
            "no_context": self.cfg.no_context,
            "suppress_blank": True,
            "suppress_non_speech_tokens": self.cfg.suppress_non_speech_tokens,
            "no_speech_thold": self.cfg.no_speech_thold,
            "initial_prompt": self.cfg.initial_prompt,
        }
        for key in self._unsupported_params:
            params.pop(key, None)

        # Some pywhispercpp bindings don't expose every PARAMS_SCHEMA key.
        # If a key is unsupported, drop only that key and retry.
        max_retries = len(params) + 1
        for _ in range(max_retries):
            try:
                # Some bindings return a lazy iterator and can raise only on iteration.
                # Materialize here so compatibility retries still work.
                return list(self.model.transcribe(media, **params))
            except Exception as e:  # noqa: BLE001
                bad_key = self._extract_unsupported_param(str(e), params)
                if bad_key is None:
                    raise

                self._unsupported_params.add(bad_key)
                params.pop(bad_key, None)
                print(f"[stt] Param '{bad_key}' unsupported by local whisper binding; disabled.", flush=True)

        raise RuntimeError("STT compatibility retries exhausted")

    @staticmethod
    def _extract_unsupported_param(err: str, params: dict[str, Any]) -> Optional[str]:
        m = re.search(r"has no attribute ['\"]([^'\"]+)['\"]", err)
        if m and m.group(1) in params:
            return m.group(1)

        m = re.search(r"has no attribute ([A-Za-z_][A-Za-z0-9_]*)", err)
        if m and m.group(1) in params:
            return m.group(1)

        m = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", err)
        if m and m.group(1) in params:
            return m.group(1)

        return None
