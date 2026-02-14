# VA (Local Voice Assistant)

Offline-first voice assistant for macOS, Linux, and Windows.

- STT: `whisper.cpp` via `pywhispercpp`
- LLM: local `llama.cpp` by default, optional LM Studio fallback
- TTS: local `Piper` by default (`say` is macOS-only fallback)

## Quick Start (Source)

```bash
./run.sh
```

`run.sh` will automatically:
- create `.venv`
- install/update dependencies
- create `.env` from `.env.example` on first run
- download missing Piper voice assets

## Default Stack

- LLM: `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (auto-downloaded)
- STT: `whisper.cpp/medium`
- TTS: `piper` + `ru_RU-irina-medium.onnx`

## Portable Build For Friends (Recommended)

You should share ZIP artifacts from `dist/`, not raw build folders.

- macOS artifact: `dist/va-assistant-macos.zip`
- Linux artifact: `dist/va-assistant-linux.zip`
- Windows artifact: `dist/va-assistant-windows.zip`

## Build For All OS (GitHub Actions)

Use workflow `.github/workflows/build-binaries.yml`.

1. Commit and push your branch.
2. Open GitHub Actions.
3. Run workflow `Build Binaries`.
4. Download artifacts: `va-macOS`, `va-Linux`, `va-Windows`.

This is the easiest way to generate all three platforms.

## Local Build (venv required)

If you build locally, create and use a virtual environment first.

macOS/Linux:
```bash
cd /path/to/va
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements-build.txt
./.venv/bin/python scripts/package.py --prefetch-assets
```

Windows (PowerShell):
```powershell
cd C:\path\to\va
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements-build.txt
.\.venv\Scripts\python .\scripts\package.py --prefetch-assets
```

## Packaging Options

- `--prefetch-assets`: download missing LLM/TTS assets before build
- `--onefile`: build one-file binary instead of one-dir app
- `--skip-bundle-assets`: do not copy local model files into release ZIP

Example:
```bash
./.venv/bin/python scripts/package.py --prefetch-assets --onefile
```

## What Gets Included In Release

- assistant binary
- `.env.example`
- `README.md`
- launcher scripts: `run.sh`, `run.bat`
- local model assets (if present or prefetched)

## Friend’s First Run

1. Unzip the archive.
2. Run `run.sh` (macOS/Linux) or `run.bat` (Windows).
3. Wait for first-run model download if assets were not bundled.
4. `.env` is created automatically from `.env.example`.

LM Studio is not required in default mode (`VA_LLM_BACKEND=local`).

## Useful Commands

List input devices:
```bash
VA_LIST_AUDIO_DEVICES=1 ./run.sh
```

Select input device:
```bash
VA_INPUT_DEVICE=2 ./run.sh
```

## Environment Configuration

Main parameters:
- `VA_LLM_BACKEND`: `local`, `auto`, `lmstudio`
- `VA_LLM_MODEL_PATH`: local `.gguf` model path
- `VA_LLM_MODEL_URL`: `.gguf` auto-download URL
- `VA_WHISPER_MODEL`: STT model (`base`, `small`, `medium`, ...)
- `VA_TTS_BACKEND`: `piper` or `say` (macOS only)
- `VA_TTS_PIPER_MODEL`: local `.onnx` voice model path
- `VA_TTS_PIPER_MODEL_URL`: voice model URL
- `VA_STT_NORMALIZE_TECH_TERMS=1`: normalize terms like `PHP`, `C++`, `SQL`

## Model Cache Paths

- Piper TTS: `.cache/voice_assistant/models/piper/`
- LLM GGUF: `.cache/voice_assistant/models/llm/`
- Whisper cache: OS user cache used by `pywhispercpp`
