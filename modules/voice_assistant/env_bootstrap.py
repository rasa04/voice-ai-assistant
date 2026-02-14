from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def resolve_runtime_root() -> Path:
    # PyInstaller: keep config near executable, not in temp extraction dir.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def bootstrap_env(runtime_root: Path) -> None:
    env_path = runtime_root / ".env"
    env_example_path = runtime_root / ".env.example"

    if not env_path.exists() and env_example_path.exists():
        shutil.copyfile(env_example_path, env_path)
        print("[info] Created .env from .env.example", flush=True)

    if env_path.exists():
        _load_env_file(env_path)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not _valid_env_name(key):
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        # Do not override explicit process env.
        if key not in os.environ:
            os.environ[key] = value


def _valid_env_name(key: str) -> bool:
    if not key:
        return False
    if not (key[0].isalpha() or key[0] == "_"):
        return False
    for ch in key[1:]:
        if not (ch.isalnum() or ch == "_"):
            return False
    return True
