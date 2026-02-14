from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass

from .audio_io import MicStream
from .config import Config
from .vad import VADConfig, VADSegmenter
from .stt import STTConfig, WhisperCppSTT
from .llm import LLMConfig, LMStudioChat
from .tts import TTSConfig, MacSayTTS


@dataclass
class Utterance:
    pcm16: bytes


def _norm_cmd(text: str) -> str:
    t = text.lower().strip()
    for ch in ["!", "?", ".", ",", "…", "—", "-", ":", ";", ")", "(", "[", "]", "{", "}", "\"", "'"]:
        t = t.replace(ch, " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t.strip()


class AssistantWorker(threading.Thread):
    def __init__(
        self,
        utter_q: "queue.Queue[Utterance]",
        stt: WhisperCppSTT,
        llm: LMStudioChat,
        tts: MacSayTTS,
        stop_event: threading.Event,
        disable_tts: bool,
    ):
        super().__init__(daemon=True)
        self.utter_q = utter_q
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.stop_event = stop_event
        self.disable_tts = disable_tts

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                utt = self.utter_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                text = self.stt.transcribe_pcm16(utt.pcm16)
                if not text:
                    continue

                print(f"\n🧏  Ты сказал: {text}\n", flush=True)

                cmd = _norm_cmd(text)

                # команды делаем "безопасными": только с префиксом "ассистент"
                if cmd in {"ассистент стоп", "ассистент стопни"}:
                    self.tts.stop()
                    continue

                if cmd in {"ассистент выход", "ассистент завершись", "ассистент выключись"}:
                    self.tts.stop()
                    self.stop_event.set()
                    return

                self.llm.add_user(text)
                print("🤖  Ответ: ", end="", flush=True)
                answer = self.llm.reply()
                self.llm.add_assistant(answer)

                if answer and not self.disable_tts:
                    print("🔊  Озвучиваю...", flush=True)
                    self.tts.speak(answer)

            except Exception as e:  # noqa: BLE001
                print(f"\n[worker error] {e}\n", flush=True)


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.cache_dir, exist_ok=True)

    disable_tts = os.getenv("VA_DISABLE_TTS", "0") == "1"

    frame_samples = int(cfg.sample_rate * (cfg.frame_ms / 1000.0))

    mic = MicStream(sample_rate=cfg.sample_rate, frame_samples=frame_samples)
    mic.start()

    vad = VADSegmenter(
        VADConfig(
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
            aggressiveness=cfg.vad_aggressiveness,
            window_ms=cfg.vad_window_ms,
            speech_start_ratio=cfg.speech_start_ratio,
            speech_end_ratio=cfg.speech_end_ratio,
            max_utterance_s=cfg.max_utterance_s,
            min_utterance_s=cfg.min_utterance_s,
        )
    )

    stt = WhisperCppSTT(
        STTConfig(
            cache_dir=cfg.cache_dir,
            sample_rate=cfg.sample_rate,
            model_name=cfg.whisper_model,
            language=cfg.whisper_language,
            n_threads=cfg.whisper_threads,
        )
    )

    llm = LMStudioChat(
        LLMConfig(
            base_url=cfg.lm_base_url,
            api_key=cfg.lm_api_key,
            model=cfg.lm_model,
            temperature=cfg.lm_temperature,
            history_turns=cfg.history_turns,
        )
    )

    tts = MacSayTTS(TTSConfig(voice=cfg.tts_voice, rate=cfg.tts_rate))

    utter_q: "queue.Queue[Utterance]" = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    worker = AssistantWorker(
        utter_q=utter_q,
        stt=stt,
        llm=llm,
        tts=tts,
        stop_event=stop_event,
        disable_tts=disable_tts,
    )
    worker.start()

    print(
        "\nГотово. Говори в микрофон.\n"
        "- Команда: «ассистент стоп» — прервать озвучку\n"
        "- Команда: «ассистент выход» — завершить\n",
        flush=True,
    )

    try:
        while not stop_event.is_set():
            frame = mic.queue.get()
            events = vad.process_frame(frame.pcm16)

            for et, payload in events:
                if et == "speech_start":
                    # barge-in: если ты начал говорить — прерываем TTS
                    if tts.is_speaking():
                        tts.stop()

                elif et == "utterance" and payload:
                    try:
                        utter_q.put_nowait(Utterance(pcm16=payload))
                    except queue.Full:
                        pass

            time.sleep(0.0)

    except KeyboardInterrupt:
        print("\nОстановлено.", flush=True)
    finally:
        stop_event.set()
        mic.stop()
        tts.stop()
