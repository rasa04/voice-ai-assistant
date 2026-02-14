from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSConfig:
    voice: str
    rate: str


class MacSayTTS:
    """
    Local TTS via macOS `say`.
    Supports stop() by terminating the subprocess (for barge-in).
    """
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        text = self._normalize(text)
        if not text:
            return

        with self._lock:
            self.stop()

            args = ["say"]
            if self.cfg.voice:
                args += ["-v", self.cfg.voice]
            if self.cfg.rate:
                args += ["-r", self.cfg.rate]
            args.append(text)

            self._proc = subprocess.Popen(args)

    def is_speaking(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            if self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None

    @staticmethod
    def _normalize(text: str) -> str:
        # TTS-friendly cleanup
        text = text.replace("```", " ")
        text = text.replace("\n", ". ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()
