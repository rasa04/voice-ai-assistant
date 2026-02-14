from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from typing import Optional

from pywhispercpp.model import Model


@dataclass
class STTConfig:
    cache_dir: str
    sample_rate: int
    model_name: str
    language: str
    n_threads: int


class WhisperCppSTT:
    """
    Offline STT using whisper.cpp via pywhispercpp.
    pywhispercpp can auto-download ggml models into cache. 
    """
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        os.makedirs(cfg.cache_dir, exist_ok=True)
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
        wav_path = self._write_wav(pcm16)
        segments = self.model.transcribe(
            wav_path,
            language=self.cfg.language,
        )
        text = " ".join((s.text or "").strip() for s in segments).strip()
        return self._cleanup_text(text)

    def _write_wav(self, pcm16: bytes) -> str:
        fn = f"utt_{int(time.time() * 1000)}.wav"
        path = os.path.join(self.cfg.cache_dir, "utterances", fn)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm16)

        return path

    @staticmethod
    def _cleanup_text(text: str) -> str:
        # a bit of normalization for voice use
        text = text.replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return text
