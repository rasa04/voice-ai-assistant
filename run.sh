#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

load_env_file() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0

  while IFS= read -r raw_line || [ -n "$raw_line" ]; do
    local line="$raw_line"
    local key=""
    local value=""

    # trim leading/trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    # skip comments/empty lines
    [ -z "$line" ] && continue
    [[ "${line:0:1}" == "#" ]] && continue

    # allow optional "export KEY=VALUE"
    if [[ "$line" == export[[:space:]]* ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi

    [[ "$line" == *"="* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"

    # trim key/value whitespace around '='
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    # strip one pair of wrapping quotes
    if [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi

    # keep shell/env naming rules
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    # do not override variables already set in the environment
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$env_file"
}

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp ".env.example" ".env"
  echo "[info] Created .env from .env.example"
fi

load_env_file ".env"

hash_requirements() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 requirements.txt | awk '{print $1}'
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum requirements.txt | awk '{print $1}'
    return
  fi
  "$PY" - <<'PY'
import hashlib
with open("requirements.txt", "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest())
PY
}

requirements_installed() {
  local line=""
  local req=""
  local pkg=""

  while IFS= read -r line || [ -n "$line" ]; do
    # trim spaces
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    [[ "${line:0:1}" == "#" ]] && continue

    # strip inline comments and environment markers
    req="${line%%#*}"
    req="${req%%;*}"
    req="${req#"${req%%[![:space:]]*}"}"
    req="${req%"${req##*[![:space:]]}"}"
    [ -z "$req" ] && continue

    pkg="$(printf '%s' "$req" | sed -E 's/[[:space:]]+//g; s/\[.*\]//; s/[<>=!~].*$//')"
    [ -z "$pkg" ] && continue

    if ! python -m pip show "$pkg" >/dev/null 2>&1; then
      return 1
    fi
  done < requirements.txt

  return 0
}

ensure_piper_assets() {
  local backend="${VA_TTS_BACKEND:-piper}"
  backend="$(printf '%s' "$backend" | tr '[:upper:]' '[:lower:]')"
  if [ "$backend" != "piper" ]; then
    return 0
  fi

  local piper_bin="${VA_TTS_PIPER_BIN:-piper}"
  if ! command -v "$piper_bin" >/dev/null 2>&1; then
    echo "[tts] Piper backend selected but binary not found: $piper_bin" >&2
    echo "[tts] Run dependencies sync or set VA_TTS_BACKEND=say." >&2
    return 1
  fi

  local model_path="${VA_TTS_PIPER_MODEL:-.cache/voice_assistant/models/piper/ru_RU-irina-medium.onnx}"
  local model_url="${VA_TTS_PIPER_MODEL_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx}"
  local config_path="${model_path}.json"
  local config_url="${VA_TTS_PIPER_CONFIG_URL:-${model_url}.json}"

  export VA_TTS_PIPER_MODEL="$model_path"

  if ! command -v curl >/dev/null 2>&1; then
    echo "[tts] curl is required to auto-download Piper model files." >&2
    return 1
  fi

  mkdir -p "$(dirname "$model_path")"

  if [ ! -f "$model_path" ]; then
    echo "[tts] Downloading Piper voice model..."
    curl -L --fail --retry 3 -o "$model_path" "$model_url"
  fi

  if [ ! -f "$config_path" ]; then
    echo "[tts] Downloading Piper voice config..."
    curl -L --fail --retry 3 -o "$config_path" "$config_url"
  fi
}

PY="${PYTHON:-python3}"
REQ_STAMP=".cache/requirements.sha256"
FORCE_PIP_SYNC="${VA_FORCE_PIP_SYNC:-0}"
SKIP_PIP_SYNC="${VA_SKIP_PIP_SYNC:-0}"
CREATED_VENV=0

if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
  CREATED_VENV=1
fi

source .venv/bin/activate

if [ "$SKIP_PIP_SYNC" = "1" ]; then
  echo "[deps] VA_SKIP_PIP_SYNC=1, skipping dependency sync."
else
  NEED_PIP_SYNC="$CREATED_VENV"
  REQ_HASH="$(hash_requirements)"
  CACHED_HASH=""
  if [ -f "$REQ_STAMP" ]; then
    CACHED_HASH="$(cat "$REQ_STAMP" 2>/dev/null || true)"
  fi

  if [ "$FORCE_PIP_SYNC" = "1" ] || [ "$REQ_HASH" != "$CACHED_HASH" ]; then
    NEED_PIP_SYNC=1
  fi

  if [ "$NEED_PIP_SYNC" = "1" ] && [ -z "$CACHED_HASH" ] && [ "$FORCE_PIP_SYNC" != "1" ]; then
    if requirements_installed; then
      mkdir -p "$(dirname "$REQ_STAMP")"
      printf "%s" "$REQ_HASH" > "$REQ_STAMP"
      NEED_PIP_SYNC=0
      echo "[deps] requirements already installed, hash cache created."
    fi
  fi

  if [ "$NEED_PIP_SYNC" = "1" ]; then
    echo "[deps] Syncing python dependencies..."
    python -m pip install -U pip wheel
    pip install -r requirements.txt
    mkdir -p "$(dirname "$REQ_STAMP")"
    printf "%s" "$REQ_HASH" > "$REQ_STAMP"
  else
    echo "[deps] requirements unchanged, skipping pip install."
  fi
fi

ensure_piper_assets

if [ "${VA_LIST_AUDIO_DEVICES:-0}" = "1" ]; then
  python - <<'PY'
import sounddevice as sd

print("Input devices:")
found = False
for idx, info in enumerate(sd.query_devices()):
    max_in = int(info.get("max_input_channels", 0))
    if max_in <= 0:
        continue
    found = True
    name = info.get("name", "unknown")
    print(f"  {idx}: {name} (inputs={max_in})")

if not found:
    print("  <none>")
PY
  exit 0
fi

exec python -m modules.voice_assistant
