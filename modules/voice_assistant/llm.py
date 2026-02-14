from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from openai import OpenAI


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    history_turns: int


class LMStudioChat:
    """
    LM Studio OpenAI-compatible client.
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Ты локальный голосовой ассистент. Отвечай по-русски, кратко и по делу. "
                    "Если вопрос неоднозначный — уточни одним коротким вопросом."
                ),
            }
        ]

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def reply(self) -> str:
        # Streaming output to console; return full text for TTS.
        out = []
        try:
            stream = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=self.history,
                temperature=self.cfg.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue
                print(delta, end="", flush=True)
                out.append(delta)
            print("", flush=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LM Studio недоступен по {self.cfg.base_url}. "
                f"Проверь что сервер запущен (lms server start) и модель загружена. "
                f"Детали: {e}"
            ) from e

        text = "".join(out).strip()
        return self._cleanup_text(text)

    def _trim(self) -> None:
        # keep system + last N turns (user+assistant pairs)
        keep = 1 + (self.cfg.history_turns * 2)
        if len(self.history) > keep:
            self.history = [self.history[0]] + self.history[-(keep - 1):]

    @staticmethod
    def _cleanup_text(text: str) -> str:
        text = text.replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return text
