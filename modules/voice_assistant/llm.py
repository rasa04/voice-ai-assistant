from __future__ import annotations

import re
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


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


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
                    "Никогда не показывай скрытые рассуждения. "
                    "НЕ используй теги <think> и не выводи размышления."
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
        # Для MVP делаем stream=False, чтобы гарантированно вырезать <think>
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
                f"Проверь сервер: lms server start. Детали: {e}"
            ) from e

        text = resp.choices[0].message.content or ""
        text = self._cleanup_text(text)

        # печатаем уже очищенный текст
        print(text, flush=True)
        return text

    def _trim(self) -> None:
        keep = 1 + (self.cfg.history_turns * 2)
        if len(self.history) > keep:
            self.history = [self.history[0]] + self.history[-(keep - 1):]

    @staticmethod
    def _cleanup_text(text: str) -> str:
        text = _THINK_RE.sub("", text)
        text = text.replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return text
