#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt

python -m modules.voice_assistant
