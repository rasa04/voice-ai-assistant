from __future__ import annotations

import os
import re
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol

from openai import OpenAI


@dataclass
class LLMConfig:
    backend: str
    base_url: str
    api_key: str
    model: str
    local_model_path: str
    local_model_url: str
    local_ctx: int
    local_threads: int
    local_gpu_layers: int
    local_max_tokens: int
    local_use_mmap: bool
    local_use_mlock: bool
    temperature: float
    timeout_s: float
    history_turns: int


class ChatLike(Protocol):
    @property
    def model_label(self) -> str: ...
    def add_user(self, text: str) -> None: ...
    def add_assistant(self, text: str) -> None: ...
    def reply(self) -> str: ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class _BaseChat:
    def __init__(self, cfg: LLMConfig, backend_hint: str):
        self.cfg = cfg
        self._backend_hint = backend_hint
        self._model_reported = False
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": _build_system_prompt(backend_hint),
            }
        ]

    @property
    def model_label(self) -> str:
        return self._backend_hint

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self) -> None:
        keep = 1 + (self.cfg.history_turns * 2)
        if len(self.history) > keep:
            self.history = [self.history[0]] + self.history[-(keep - 1) :]

    @staticmethod
    def _cleanup_text(text: str) -> str:
        text = _THINK_RE.sub("", text or "")
        text = text.replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return text


class LMStudioChat(_BaseChat):
    """
    OpenAI-compatible LM Studio backend.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg, f"{cfg.model} (LM Studio)")
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
        )

    def reply(self) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=self.history,
                temperature=self.cfg.temperature,
                stream=False,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LM Studio недоступен по {self.cfg.base_url}. "
                f"Проверь сервер LM Studio. Детали: {e}"
            ) from e

        remote_model = getattr(resp, "model", "") or self.cfg.model
        if not self._model_reported:
            print(f"[llm] Модель backend: {remote_model} (LM Studio)", flush=True)
            self._model_reported = True

        text = self._cleanup_text(resp.choices[0].message.content or "")
        print(text, flush=True)
        return text


class LocalLlamaChat(_BaseChat):
    """
    Fully local LLM backend via llama.cpp Python bindings.
    """

    def __init__(self, cfg: LLMConfig):
        model_path = _ensure_local_llm_model(cfg.local_model_path, cfg.local_model_url)

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Локальный LLM backend требует пакет llama-cpp-python. "
                f"Детали: {e}"
            ) from e

        threads = cfg.local_threads if cfg.local_threads > 0 else max(1, (os.cpu_count() or 4) - 1)
        super().__init__(cfg, f"{Path(model_path).name} (local llama.cpp)")

        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=max(512, cfg.local_ctx),
            n_threads=threads,
            n_gpu_layers=cfg.local_gpu_layers,
            use_mmap=cfg.local_use_mmap,
            use_mlock=cfg.local_use_mlock,
            verbose=False,
        )

    def reply(self) -> str:
        if not self._model_reported:
            print(f"[llm] Модель backend: {self.model_label}", flush=True)
            self._model_reported = True

        try:
            resp = self.llm.create_chat_completion(
                messages=self.history,
                temperature=self.cfg.temperature,
                max_tokens=max(64, self.cfg.local_max_tokens),
                stream=False,
            )
            raw_text = _extract_chat_message_text(resp)
        except Exception:
            # Some GGUF builds have incomplete chat metadata; fallback to manual ChatML prompt.
            prompt = _history_to_chatml(self.history)
            try:
                completion = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=max(64, self.cfg.local_max_tokens),
                    temperature=self.cfg.temperature,
                    stop=["<|im_end|>", "</s>"],
                )
                raw_text = _extract_completion_text(completion)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Локальный LLM вызов завершился ошибкой: {e}") from e

        text = self._cleanup_text(raw_text)
        print(text, flush=True)
        return text


def build_chat(cfg: LLMConfig) -> ChatLike:
    backend = (cfg.backend or "auto").strip().lower()
    if backend in {"lmstudio", "openai"}:
        return LMStudioChat(cfg)
    if backend in {"local", "llamacpp", "llama.cpp"}:
        return LocalLlamaChat(cfg)
    if backend == "auto":
        try:
            return LocalLlamaChat(cfg)
        except Exception as local_err:  # noqa: BLE001
            print(f"[llm] Local backend unavailable: {local_err}", flush=True)
            print("[llm] Falling back to LM Studio backend.", flush=True)
            return LMStudioChat(cfg)
    raise RuntimeError(f"Unknown LLM backend: {cfg.backend}")


def _build_system_prompt(backend_hint: str) -> str:
    return (
        "Ты локальный голосовой ассистент. Отвечай по-русски, кратко и по делу. "
        "Не используй эмодзи, смайлики и markdown-разметку. "
        f"Текущая языковая модель backend: {backend_hint}. "
        "Никогда не выдумывай название компании-разработчика, если не уверен. "
        "Если спрашивают «кто тебя разработал» или «какая ты модель», "
        "честно говори, что ты локальный ассистент, запущенный пользователем, "
        f"а LLM backend сейчас: {backend_hint}. "
        "Никогда не показывай скрытые рассуждения. "
        "НЕ используй теги <think> и не выводи размышления."
    )


def _ensure_local_llm_model(path_raw: str, url: str) -> str:
    if not path_raw:
        raise RuntimeError("VA_LLM_MODEL_PATH is empty")

    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return str(path)

    if not url:
        raise RuntimeError(f"LLM model not found: {path} and VA_LLM_MODEL_URL is empty")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    url_variants = _download_url_variants(url)
    for idx, candidate_url in enumerate(url_variants, start=1):
        if idx > 1:
            print(f"[llm] Trying fallback URL: {candidate_url}", flush=True)

        for attempt in range(1, 4):
            try:
                print(f"[llm] Downloading local model (attempt {attempt}/3)...", flush=True)
                ssl_ctx = _download_ssl_context()
                with urllib.request.urlopen(candidate_url, timeout=180, context=ssl_ctx) as response, open(tmp_path, "wb") as out:
                    shutil.copyfileobj(response, out)
                tmp_path.replace(path)
                return str(path)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 404:
                    break
                if _is_ssl_error(e):
                    curl_ok, curl_err = _download_with_curl(candidate_url, tmp_path)
                    if curl_ok:
                        tmp_path.replace(path)
                        return str(path)
                    last_err = RuntimeError(f"{e}; curl fallback failed: {curl_err}")  # noqa: TRY004
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_ssl_error(e):
                    curl_ok, curl_err = _download_with_curl(candidate_url, tmp_path)
                    if curl_ok:
                        tmp_path.replace(path)
                        return str(path)
                    last_err = RuntimeError(f"{e}; curl fallback failed: {curl_err}")  # noqa: TRY004
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass

            time.sleep(0.8 * attempt)

    raise RuntimeError(
        f"Failed to download local LLM model from {url}. "
        f"Tried {len(url_variants)} URL variant(s). Last error: {last_err}"
    ) from last_err


def _download_ssl_context() -> ssl.SSLContext:
    ca_bundle = os.getenv("VA_DOWNLOAD_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)

    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _is_ssl_error(err: Exception) -> bool:
    if isinstance(err, ssl.SSLCertVerificationError):
        return True
    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return True
    return "CERTIFICATE_VERIFY_FAILED" in str(err).upper()


def _download_with_curl(url: str, tmp_path: Path) -> tuple[bool, str]:
    curl_bin = shutil.which("curl")
    if not curl_bin:
        return False, "curl not found in PATH"

    cmd = [curl_bin, "-L", "--fail", "--retry", "3", "-o", str(tmp_path), url]
    ca_bundle = os.getenv("VA_DOWNLOAD_CA_BUNDLE", "").strip()
    if ca_bundle:
        cmd.extend(["--cacert", ca_bundle])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _download_url_variants(url: str) -> list[str]:
    variants = [url]
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc != "huggingface.co":
            return variants
        parts = parsed.path.strip("/").split("/")
        # /<owner>/<repo>/resolve/<revision>/<filename>
        if len(parts) < 5 or parts[2] != "resolve":
            return variants

        repo_name = parts[1]
        filename = parts[-1]
        repo_base = repo_name[:-5] if repo_name.endswith("-GGUF") else repo_name
        core_name = filename[:-5] if filename.endswith(".gguf") else filename
        model_part = core_name.split("-Q", 1)[0]

        alt_names: list[str] = []
        prefix = ""
        if model_part and repo_base.endswith(model_part):
            prefix = repo_base[: -len(model_part)]

        if prefix and not filename.startswith(prefix):
            alt_names.append(f"{prefix}{filename}")
        if prefix and filename.startswith(prefix):
            alt_names.append(filename[len(prefix) :])

        for alt_name in alt_names:
            alt_parts = parts.copy()
            alt_parts[-1] = alt_name
            alt_path = "/" + "/".join(alt_parts)
            alt_url = urllib.parse.urlunparse(parsed._replace(path=alt_path))
            if alt_url not in variants:
                variants.append(alt_url)
    except Exception:  # noqa: BLE001
        return variants

    return variants


def _extract_chat_message_text(resp: object) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content")
    return _normalize_content(content)


def _extract_completion_text(resp: object) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    return _normalize_content((choices[0] or {}).get("text", ""))


def _normalize_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _history_to_chatml(history: List[Dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in history:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
