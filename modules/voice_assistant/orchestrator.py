from __future__ import annotations

import os
import queue
import shutil
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .audio_io import MicStream
from .config import Config
from .env_bootstrap import bootstrap_env, resolve_runtime_root
from .vad import VADConfig, VADSegmenter
from .stt import STTConfig, WhisperCppSTT, get_whisper_profile
from .llm import ChatLike, LLMConfig, build_chat
from .tts import TTSConfig, TTSLike, build_tts


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


def _meta_answer(cmd: str, llm_model: str) -> str:
    if any(
        q in cmd
        for q in {
            "какая ты модель",
            "что ты за модель",
            "какая ты нейронка",
            "какая у тебя модель",
            "кто ты",
        }
    ):
        return (
            f"Я локальный голосовой ассистент. Сейчас backend LLM: {llm_model} "
            "Запуск локальный."
        )

    if any(
        q in cmd
        for q in {
            "кто тебя разработал",
            "кто твой создатель",
            "кем ты разработан",
        }
    ):
        return (
            "Это локальный ассистент в вашем проекте. "
            f"Текущая языковая модель backend: {llm_model}. "
            "Конкретного разработчика ассистента я не выдумываю."
        )

    return ""


class AssistantWorker(threading.Thread):
    def __init__(
        self,
        utter_q: "queue.Queue[Utterance]",
        stt: WhisperCppSTT,
        llm: ChatLike,
        tts: TTSLike,
        stop_event: threading.Event,
        disable_tts: bool,
        min_text_chars: int,
        duplicate_utt_window_s: float,
    ):
        super().__init__(daemon=True)
        self.utter_q = utter_q
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.stop_event = stop_event
        self.disable_tts = disable_tts
        self.min_text_chars = max(1, min_text_chars)
        self.duplicate_utt_window_s = max(0.0, duplicate_utt_window_s)
        self._last_norm_text = ""
        self._last_text_ts = 0.0

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

                if len(cmd.replace(" ", "")) < self.min_text_chars:
                    continue

                now = time.monotonic()
                if (
                    self.duplicate_utt_window_s > 0
                    and cmd == self._last_norm_text
                    and (now - self._last_text_ts) <= self.duplicate_utt_window_s
                ):
                    print("[stt] Повтор распознавания, пропускаю.", flush=True)
                    continue

                self._last_norm_text = cmd
                self._last_text_ts = now

                meta = _meta_answer(cmd, self.llm.model_label)
                if meta:
                    print("🤖  Ответ: ", end="", flush=True)
                    print(meta, flush=True)
                    if meta and not self.disable_tts:
                        print("🔊  Озвучиваю...", flush=True)
                        self.tts.speak(meta)
                    continue

                self.llm.add_user(text)
                print("🤖  Ответ: ", end="", flush=True)
                answer = self.llm.reply()
                self.llm.add_assistant(answer)

                if answer and not self.disable_tts:
                    print("🔊  Озвучиваю...", flush=True)
                    self.tts.speak(answer)

            except Exception as e:  # noqa: BLE001
                print(f"\n[worker error] {e}\n", flush=True)


def _ensure_piper_assets(cfg: Config) -> None:
    if (cfg.tts_backend or "").strip().lower() != "piper":
        return

    model_path = (cfg.tts_piper_model or "").strip()
    if not model_path:
        raise RuntimeError("Piper backend requires VA_TTS_PIPER_MODEL")

    model_url = (cfg.tts_piper_model_url or "").strip()
    config_url = (cfg.tts_piper_config_url or "").strip() or (f"{model_url}.json" if model_url else "")
    config_path = f"{model_path}.json"

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    if not os.path.exists(model_path):
        _download_file(model_url, model_path, "Piper voice model")

    if not os.path.exists(config_path):
        _download_file(config_url, config_path, "Piper voice config")


def _download_file(url: str, dst_path: str, label: str) -> None:
    if not url:
        raise RuntimeError(f"{label} missing and download URL is empty.")

    tmp_path = f"{dst_path}.tmp"
    last_err: Exception | None = None

    for attempt in range(1, 4):
        try:
            print(f"[tts] Downloading {label} (attempt {attempt}/3)...", flush=True)
            ssl_ctx = _download_ssl_context()
            with urllib.request.urlopen(url, timeout=120, context=ssl_ctx) as response, open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out)
            os.replace(tmp_path, dst_path)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_ssl_error(e):
                curl_ok, curl_err = _download_with_curl(url, tmp_path)
                if curl_ok:
                    os.replace(tmp_path, dst_path)
                    return
                last_err = RuntimeError(f"{e}; curl fallback failed: {curl_err}")  # noqa: TRY004
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            time.sleep(0.6 * attempt)

    raise RuntimeError(f"Failed to download {label} from {url}: {last_err}") from last_err


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


def _download_with_curl(url: str, tmp_path: str) -> tuple[bool, str]:
    curl_bin = shutil.which("curl")
    if not curl_bin:
        return False, "curl not found in PATH"

    cmd = [curl_bin, "-L", "--fail", "--retry", "3", "-o", tmp_path, url]
    ca_bundle = os.getenv("VA_DOWNLOAD_CA_BUNDLE", "").strip()
    if ca_bundle:
        cmd.extend(["--cacert", ca_bundle])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> None:
    runtime_root = resolve_runtime_root()
    os.chdir(runtime_root)
    bootstrap_env(runtime_root)
    cfg = Config()
    os.makedirs(cfg.cache_dir, exist_ok=True)

    disable_tts = os.getenv("VA_DISABLE_TTS", "0") == "1"
    tts_backend = (cfg.tts_backend or "say").strip().lower()
    tts_voice_hint = cfg.tts_voice or (f"auto({cfg.tts_lang})" if tts_backend == "say" else "n/a")
    tts_model_hint = os.path.basename(cfg.tts_piper_model) if tts_backend == "piper" and cfg.tts_piper_model else "n/a"
    stt_profile = get_whisper_profile(cfg.whisper_model)
    stt_profile_line = ""
    if stt_profile:
        stt_profile_line = (
            f"- STT profile: params={stt_profile['params']} "
            f"disk={stt_profile['disk']} "
            f"speed={stt_profile['speed']} "
            f"quality={stt_profile['quality']}\n"
        )
    tts_status = (
        "off (VA_DISABLE_TTS=1)"
        if disable_tts
        else (
            f"on (backend={tts_backend}, "
            f"voice={tts_voice_hint}, "
            f"tts_model={tts_model_hint}, "
            f"rate={cfg.tts_rate or 'system-default'}, "
            f"strip_emoji={int(cfg.tts_strip_emoji)})"
        )
    )
    llm_backend = (cfg.lm_backend or "auto").strip().lower()
    if llm_backend in {"local", "llamacpp", "llama.cpp"}:
        llm_target = f"{os.path.basename(cfg.lm_local_model_path)} (local llama.cpp)"
    elif llm_backend in {"lmstudio", "openai"}:
        llm_target = f"{cfg.lm_model} @ {cfg.lm_base_url}"
    else:
        llm_target = (
            f"auto: local={os.path.basename(cfg.lm_local_model_path)} "
            f"fallback={cfg.lm_model} @ {cfg.lm_base_url}"
        )

    print(
        "\nКонфиг запуска:\n"
        f"- LLM: backend={llm_backend} target={llm_target}\n"
        f"- LLM timeout: {cfg.lm_timeout_s}s\n"
        f"- STT: whisper.cpp/{cfg.whisper_model} "
        f"lang={cfg.whisper_language} threads={cfg.whisper_threads} "
        f"save_wav={int(cfg.stt_save_utterances)} "
        f"normalize_terms={int(cfg.stt_normalize_tech_terms)} "
        f"no_context={int(cfg.stt_no_context)} "
        f"no_speech_thold={cfg.stt_no_speech_thold}\n"
        f"{stt_profile_line}"
        f"- VAD: aggr={cfg.vad_aggressiveness} "
        f"window={cfg.vad_window_ms}ms "
        f"start={cfg.speech_start_ratio} end={cfg.speech_end_ratio}\n"
        f"- Echo guard: suppress={cfg.tts_echo_suppress_ms}ms "
        f"barge_in={int(cfg.allow_barge_in)}\n"
        f"- Mic device: {cfg.input_device or 'system-default'}\n"
        f"- Filters: min_text_chars={cfg.min_text_chars} "
        f"duplicate_window={cfg.duplicate_utt_window_s}s\n"
        f"- TTS: {tts_status}\n",
        flush=True,
    )
    if disable_tts:
        print("[warn] Озвучка отключена: VA_DISABLE_TTS=1", flush=True)
    else:
        try:
            _ensure_piper_assets(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"\n[tts asset error]\n{e}\n", flush=True)
            return

    frame_samples = int(cfg.sample_rate * (cfg.frame_ms / 1000.0))

    mic = MicStream(
        sample_rate=cfg.sample_rate,
        frame_samples=frame_samples,
        input_device=cfg.input_device,
    )
    try:
        mic.start()
    except Exception as e:  # noqa: BLE001
        print(f"\n[audio error]\n{e}\n", flush=True)
        return

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
            save_utterances=cfg.stt_save_utterances,
            drop_noise_tags=cfg.stt_drop_noise_tags,
            normalize_tech_terms=cfg.stt_normalize_tech_terms,
            no_context=cfg.stt_no_context,
            suppress_non_speech_tokens=cfg.stt_suppress_non_speech_tokens,
            no_speech_thold=cfg.stt_no_speech_thold,
            initial_prompt=cfg.stt_initial_prompt,
            drop_subtitle_hallucinations=cfg.stt_drop_subtitle_hallucinations,
        )
    )

    try:
        llm = build_chat(
            LLMConfig(
                backend=cfg.lm_backend,
                base_url=cfg.lm_base_url,
                api_key=cfg.lm_api_key,
                model=cfg.lm_model,
                local_model_path=cfg.lm_local_model_path,
                local_model_url=cfg.lm_local_model_url,
                local_ctx=cfg.lm_local_ctx,
                local_threads=cfg.lm_local_threads,
                local_gpu_layers=cfg.lm_local_gpu_layers,
                local_max_tokens=cfg.lm_local_max_tokens,
                local_use_mmap=cfg.lm_local_use_mmap,
                local_use_mlock=cfg.lm_local_use_mlock,
                temperature=cfg.lm_temperature,
                timeout_s=cfg.lm_timeout_s,
                history_turns=cfg.history_turns,
            )
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n[llm error]\n{e}\n", flush=True)
        return

    try:
        tts = build_tts(
            TTSConfig(
                backend=cfg.tts_backend,
                lang=cfg.tts_lang,
                voice=cfg.tts_voice,
                rate=cfg.tts_rate,
                strip_emoji=cfg.tts_strip_emoji,
                strip_markdown=cfg.tts_strip_markdown,
                max_chars=cfg.tts_max_chars,
                piper_bin=cfg.tts_piper_bin,
                piper_model=cfg.tts_piper_model,
                cache_dir=cfg.cache_dir,
            )
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n[tts error]\n{e}\n", flush=True)
        return

    utter_q: "queue.Queue[Utterance]" = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    worker = AssistantWorker(
        utter_q=utter_q,
        stt=stt,
        llm=llm,
        tts=tts,
        stop_event=stop_event,
        disable_tts=disable_tts,
        min_text_chars=cfg.min_text_chars,
        duplicate_utt_window_s=cfg.duplicate_utt_window_s,
    )
    worker.start()

    print(
        "\nГотово. Говори в микрофон.\n"
        "- Команда: «ассистент стоп» — прервать озвучку\n"
        "- Команда: «ассистент выход» — завершить\n",
        flush=True,
    )
    if not cfg.allow_barge_in:
        print(
            "[info] Barge-in отключен (VA_ALLOW_BARGE_IN=0): "
            "во время озвучки микрофон игнорируется, чтобы не было рекурсии.",
            flush=True,
        )

    try:
        echo_suppress_s = max(0, cfg.tts_echo_suppress_ms) / 1000.0
        suppress_until = 0.0

        while not stop_event.is_set():
            frame = mic.queue.get()
            now = time.monotonic()
            speaking = tts.is_speaking()

            # Anti-feedback mode: drop mic frames while TTS is active
            # and for a short cooldown after TTS ends.
            if speaking:
                suppress_until = now + echo_suppress_s
                if not cfg.allow_barge_in:
                    vad.reset()
                    continue
            elif now < suppress_until:
                vad.reset()
                continue

            events = vad.process_frame(frame.pcm16)

            for et, payload in events:
                if et == "speech_start":
                    # barge-in: если ты начал говорить — прерываем TTS
                    if tts.is_speaking():
                        if cfg.allow_barge_in:
                            tts.stop()
                            suppress_until = time.monotonic() + echo_suppress_s
                        vad.reset()
                        continue

                elif et == "utterance" and payload:
                    if time.monotonic() < suppress_until:
                        continue
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
