from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional, Protocol


_SPACE_RE = re.compile(r"\s+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_URL_RE = re.compile(r"https?://\S+")
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_EMOJI_ALIAS_RE = re.compile(r":[a-zA-Z0-9_+\-]{2,}:")
_EMOJI_RE = re.compile(r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]")
_MD_NOISE_RE = re.compile(r"[*_#>|~]+")
_MULTI_PUNCT_RE = re.compile(r"([!?.,])\1+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?])")


class TTSLike(Protocol):
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def is_speaking(self) -> bool: ...


@dataclass
class TTSConfig:
    backend: str
    lang: str
    voice: str
    rate: str
    strip_emoji: bool
    strip_markdown: bool
    max_chars: int
    piper_bin: str
    piper_model: str
    cache_dir: str


class MacSayTTS:
    """
    Local TTS via macOS `say`.
    Supports stop() by terminating the subprocess (for barge-in).
    """

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._voice = (cfg.voice or "").strip() or self._pick_auto_voice(cfg.lang)

    def speak(self, text: str) -> None:
        text = _normalize_tts_text(text, self.cfg)
        if not text:
            return

        with self._lock:
            self._stop_locked()

            args = ["say"]
            if self._voice:
                args += ["-v", self._voice]
            if self.cfg.rate:
                args += ["-r", self.cfg.rate]
            args.append(text)

            self._proc = subprocess.Popen(args)

    def is_speaking(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        self._proc = None

    @staticmethod
    def _pick_auto_voice(lang: str) -> str:
        voices = _list_say_voices()
        if not voices:
            return ""

        lang_key = (lang or "").lower()
        preferred: list[str]
        if lang_key.startswith("ru"):
            preferred = ["Milena", "Yuri", "Katya"]
        elif lang_key.startswith("en"):
            preferred = ["Samantha", "Alex"]
        else:
            preferred = []

        for candidate in preferred:
            if candidate in voices:
                return candidate

        return ""


class PiperTTS:
    """
    Local neural TTS via external `piper` binary + `afplay`.
    """

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._piper_proc: Optional[subprocess.Popen] = None
        self._player_proc: Optional[subprocess.Popen] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._tmp_wav: Optional[str] = None
        self._piper_exec = self._resolve_piper_exec(cfg.piper_bin)
        self._fallback_say = MacSayTTS(
            TTSConfig(
                backend="say",
                lang=cfg.lang,
                voice="",  # auto-pick language voice
                rate=cfg.rate,
                strip_emoji=cfg.strip_emoji,
                strip_markdown=cfg.strip_markdown,
                max_chars=cfg.max_chars,
                piper_bin=cfg.piper_bin,
                piper_model=cfg.piper_model,
                cache_dir=cfg.cache_dir,
            )
        )

        if not cfg.piper_model:
            raise RuntimeError(
                "Piper backend requires VA_TTS_PIPER_MODEL (path to .onnx)."
            )
        if self._piper_exec is None:
            raise RuntimeError(
                f"Piper backend requested, but binary not found: {cfg.piper_bin}"
            )
        if shutil.which("afplay") is None:
            raise RuntimeError("Piper backend requires macOS afplay.")

    def speak(self, text: str) -> None:
        text = _normalize_tts_text(text, self.cfg)
        if not text:
            return

        with self._lock:
            self._stop_locked()
            self._stop_event = threading.Event()
            worker = threading.Thread(
                target=self._run_pipeline,
                args=(text, self._stop_event),
                daemon=True,
            )
            self._worker = worker
            worker.start()

    def is_speaking(self) -> bool:
        with self._lock:
            if self._piper_proc is not None and self._piper_proc.poll() is None:
                return True
            if self._player_proc is not None and self._player_proc.poll() is None:
                return True
            if self._worker is not None and self._worker.is_alive():
                return True
        return self._fallback_say.is_speaking()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop_event.set()

        if self._player_proc is not None and self._player_proc.poll() is None:
            try:
                self._player_proc.terminate()
                self._player_proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                try:
                    self._player_proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        self._player_proc = None

        if self._piper_proc is not None and self._piper_proc.poll() is None:
            try:
                self._piper_proc.terminate()
                self._piper_proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                try:
                    self._piper_proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        self._piper_proc = None

        tmp = self._tmp_wav
        self._tmp_wav = None
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass

        # Do not keep stale fallback audio playing.
        self._fallback_say.stop()

    def _run_pipeline(self, text: str, stop_event: threading.Event) -> None:
        fd, wav_path = tempfile.mkstemp(prefix="piper_tts_", suffix=".wav", dir=self.cfg.cache_dir)
        os.close(fd)

        with self._lock:
            if stop_event is not self._stop_event:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                return
            self._tmp_wav = wav_path

        try:
            piper_args = [
                self._piper_exec,
                "--model",
                self.cfg.piper_model,
                "--output_file",
                wav_path,
            ]
            piper_proc = subprocess.Popen(
                piper_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._lock:
                if stop_event is not self._stop_event:
                    try:
                        piper_proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                    return
                self._piper_proc = piper_proc

            piper_stderr = ""
            try:
                _, piper_stderr = piper_proc.communicate(text, timeout=max(20.0, len(text) / 8.0))
            except subprocess.TimeoutExpired:
                try:
                    piper_proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                piper_proc.wait()
                piper_stderr = "timeout"

            with self._lock:
                if self._piper_proc is piper_proc:
                    self._piper_proc = None

            if stop_event.is_set() or piper_proc.returncode not in (0, None):
                if not stop_event.is_set():
                    self._report_runtime_error("piper", piper_proc.returncode or 1, piper_stderr)
                    self._speak_fallback(text, stop_event)
                return

            player_proc = subprocess.Popen(
                ["afplay", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with self._lock:
                if stop_event is not self._stop_event:
                    try:
                        player_proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                    return
                self._player_proc = player_proc

            while player_proc.poll() is None and not stop_event.is_set():
                stop_event.wait(0.05)

            afplay_stderr = ""
            if stop_event.is_set() and player_proc.poll() is None:
                try:
                    player_proc.terminate()
                    player_proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    try:
                        player_proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            else:
                _, afplay_stderr = player_proc.communicate()

            with self._lock:
                if self._player_proc is player_proc:
                    self._player_proc = None

            if not stop_event.is_set() and player_proc.returncode not in (0, None):
                self._report_runtime_error("afplay", player_proc.returncode or 1, afplay_stderr)
                self._speak_fallback(text, stop_event)

        finally:
            with self._lock:
                tmp = self._tmp_wav
                self._tmp_wav = None
                if self._worker is not None and threading.current_thread() is self._worker:
                    self._worker = None

            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _speak_fallback(self, text: str, stop_event: threading.Event) -> None:
        if stop_event.is_set():
            return
        print("[tts] Falling back to macOS say.", flush=True)
        self._fallback_say.speak(text)

    @staticmethod
    def _report_runtime_error(stage: str, code: int, stderr_text: str) -> None:
        details = (stderr_text or "").strip()
        if len(details) > 220:
            details = details[:220] + "..."
        msg = f"[tts error] {stage} failed (rc={code})"
        if details:
            msg += f": {details}"
        print(msg, flush=True)

    @staticmethod
    def _resolve_piper_exec(piper_bin: str) -> Optional[str]:
        if not piper_bin:
            return None

        direct = shutil.which(piper_bin)
        if direct:
            return direct

        if os.path.isabs(piper_bin) and os.path.exists(piper_bin):
            return piper_bin

        # Fallback: if running from venv python, look for sibling script.
        py_bin_dir = os.path.dirname(sys.executable)
        candidate = os.path.join(py_bin_dir, piper_bin)
        if os.path.exists(candidate):
            return candidate

        return None


def build_tts(cfg: TTSConfig) -> TTSLike:
    backend = (cfg.backend or "say").strip().lower()
    if backend == "piper":
        return PiperTTS(cfg)
    if backend == "say":
        return MacSayTTS(cfg)
    raise RuntimeError(f"Unknown TTS backend: {cfg.backend}")


def _list_say_voices() -> set[str]:
    try:
        out = subprocess.run(
            ["say", "-v", "?"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:  # noqa: BLE001
        return set()

    voices: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        voices.add(line.split()[0])
    return voices


def _normalize_tts_text(text: str, cfg: TTSConfig) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)

    if cfg.strip_markdown:
        text = _MD_NOISE_RE.sub(" ", text)

    if cfg.strip_emoji:
        text = _EMOJI_ALIAS_RE.sub(" ", text)
        text = _EMOJI_RE.sub(" ", text)

    text = text.replace("\n", ". ")
    text = text.replace("\t", " ")
    text = text.replace(";", ", ")
    text = text.replace(":", ". ")
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_RE.sub(" ", text).strip()
    text = _truncate_tts_text(text, cfg.max_chars)

    if text and text[-1] not in ".!?":
        text += "."
    return text


def _truncate_tts_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    best = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind(", "))
    if best > int(max_chars * 0.6):
        cut = cut[: best + 1].strip()
    else:
        cut = cut.rstrip()
    return cut + "..."
