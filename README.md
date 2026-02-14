# VA (локальный голосовой ассистент)

Локальный voice-agent на macOS:
- STT: `whisper.cpp` (через `pywhispercpp`)
- LLM: LM Studio (`OpenAI-compatible` endpoint)
- TTS: локальный `Piper` (по умолчанию) или `macOS say`

## One-click запуск на новом устройстве

1. Клонируй репозиторий.
2. Запусти:

```bash
./run.sh
```

`run.sh` автоматически:
- создаст `.venv`
- установит зависимости
- подхватит `.env` (или создаст из `.env.example`)
- для `Piper` докачает голосовую модель, если ее нет

## Текущие дефолты

- LLM: `qwen/qwen3-4b-2507` (должна быть загружена в LM Studio)
- STT: `whisper.cpp/medium`
- TTS: `piper` + `ru_RU-irina-medium.onnx`

Все запускается локально на машине.

## Перед запуском

1. Подними LM Studio сервер и загрузи нужную модель (например `qwen/qwen3-4b-2507`).
2. Проверь, что endpoint доступен по `http://localhost:1234/v1`.

## Полезные команды

Список микрофонов:
```bash
VA_LIST_AUDIO_DEVICES=1 ./run.sh
```

Если нужно выбрать устройство:
```bash
VA_INPUT_DEVICE=2 ./run.sh
```
или укажи `VA_INPUT_DEVICE` в `.env`.

## Настройка через `.env`

Ключевые параметры:
- `LMSTUDIO_MODEL` — модель LLM
- `VA_WHISPER_MODEL` — модель STT (`small`, `medium`, ...)
- `VA_TTS_BACKEND` — `piper` или `say`
- `VA_TTS_PIPER_MODEL` — путь к `.onnx` модели голоса
- `VA_TTS_PIPER_MODEL_URL` — URL для автодокачки модели
- `VA_STT_NORMALIZE_TECH_TERMS=1` — нормализация тех-терминов (`php`, `c++`, `sql`, ...)

## Fallback на `say` (если нужно)

В `.env`:
```bash
VA_TTS_BACKEND=say
VA_TTS_VOICE=Milena
VA_TTS_RATE=185
```

## Где хранятся модели

- Piper TTS модель: `.cache/voice_assistant/models/piper/`
- Whisper кэш: `~/Library/Application Support/pywhispercpp/models/`
