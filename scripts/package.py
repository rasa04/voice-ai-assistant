#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portable VA binaries with PyInstaller")
    parser.add_argument("--name", default="va-assistant", help="binary/app name")
    parser.add_argument("--onefile", action="store_true", help="build one-file binary")
    parser.add_argument(
        "--skip-bundle-assets",
        action="store_true",
        help="do not copy pre-downloaded LLM/TTS model files into release folder",
    )
    parser.add_argument(
        "--prefetch-assets",
        action="store_true",
        help="download missing LLM/TTS assets before packaging",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    dist_dir = root / "dist"
    entrypoint = root / "app_entry.py"

    if not entrypoint.exists():
        print(f"[error] entrypoint not found: {entrypoint}")
        return 1

    if args.prefetch_assets:
        _prefetch_optional_assets(root)

    pyinstaller_cmd = _pyinstaller_cmd(
        root=root,
        build_dir=build_dir,
        dist_dir=dist_dir,
        entrypoint=entrypoint,
        name=args.name,
        onefile=args.onefile,
    )

    print("[build] Running PyInstaller:")
    print(" ".join(str(x) for x in pyinstaller_cmd))
    subprocess.run(pyinstaller_cmd, check=True, cwd=root)

    stage_dir, binary_name = _create_release_folder(
        root=root,
        dist_dir=dist_dir,
        name=args.name,
        onefile=args.onefile,
    )
    if not args.skip_bundle_assets:
        _bundle_optional_assets(root, stage_dir)
    _write_launchers(stage_dir, binary_name)

    archive_path = shutil.make_archive(str(stage_dir), "zip", root_dir=stage_dir.parent, base_dir=stage_dir.name)
    print(f"[build] Release folder: {stage_dir}")
    print(f"[build] Zip archive: {archive_path}")
    return 0


def _pyinstaller_cmd(
    *,
    root: Path,
    build_dir: Path,
    dist_dir: Path,
    entrypoint: Path,
    name: str,
    onefile: bool,
) -> list[str]:
    sep = ";" if os.name == "nt" else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--workpath",
        str(build_dir),
        "--distpath",
        str(dist_dir),
        "--paths",
        str(root),
        "--additional-hooks-dir",
        str(root / "packaging_hooks"),
        "--collect-all",
        "pywhispercpp",
        "--collect-all",
        "webrtcvad",
        "--collect-all",
        "sounddevice",
        "--collect-all",
        "numpy",
        "--collect-all",
        "onnxruntime",
        "--collect-all",
        "llama_cpp",
        "--copy-metadata",
        "openai",
        "--copy-metadata",
        "pywhispercpp",
        "--copy-metadata",
        "piper-tts",
        "--copy-metadata",
        "webrtcvad-wheels",
        "--copy-metadata",
        "llama-cpp-python",
        "--add-data",
        f"{root / '.env.example'}{sep}.",
        "--add-data",
        f"{root / 'README.md'}{sep}.",
    ]

    piper_bin = _find_piper_binary()
    if piper_bin is not None:
        cmd += ["--add-binary", f"{piper_bin}{sep}bin"]
        print(f"[build] Including piper binary: {piper_bin}")
    else:
        print("[warn] piper binary not found in PATH. Packaged app may require VA_TTS_BACKEND=say.")

    cmd.append("--onefile" if onefile else "--onedir")
    cmd.append(str(entrypoint))
    return cmd


def _find_piper_binary() -> Path | None:
    candidates: list[str] = ["piper"]
    if os.name == "nt":
        candidates.append("piper.exe")

    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    py_bin_dir = Path(sys.executable).resolve().parent
    project_root = Path(__file__).resolve().parents[1]
    extra_roots = [py_bin_dir, project_root / ".venv" / ("Scripts" if os.name == "nt" else "bin")]
    for root in extra_roots:
        for name in candidates:
            candidate = root / name
            if candidate.exists():
                return candidate

    return None


def _create_release_folder(*, root: Path, dist_dir: Path, name: str, onefile: bool) -> tuple[Path, str]:
    platform_tag = _platform_tag()
    release_dir = dist_dir / f"{name}-{platform_tag}"
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    exe_suffix = ".exe" if os.name == "nt" else ""
    binary_name = f"{name}{exe_suffix}"

    if onefile:
        source_bin = dist_dir / binary_name
        if not source_bin.exists():
            raise RuntimeError(f"Built binary not found: {source_bin}")
        shutil.copy2(source_bin, release_dir / binary_name)
    else:
        source_dir = dist_dir / name
        if not source_dir.exists():
            raise RuntimeError(f"Built app folder not found: {source_dir}")
        shutil.copytree(source_dir, release_dir, dirs_exist_ok=True)

    shutil.copy2(root / ".env.example", release_dir / ".env.example")
    shutil.copy2(root / "README.md", release_dir / "README.md")
    return release_dir, binary_name


def _prefetch_optional_assets(root: Path) -> None:
    env = _load_env_values(root)

    llm_model = env.get(
        "VA_LLM_MODEL_PATH",
        ".cache/voice_assistant/models/llm/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )
    llm_url = env.get(
        "VA_LLM_MODEL_URL",
        "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )
    tts_model = env.get(
        "VA_TTS_PIPER_MODEL",
        ".cache/voice_assistant/models/piper/ru_RU-irina-medium.onnx",
    )
    tts_model_url = env.get(
        "VA_TTS_PIPER_MODEL_URL",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    )
    tts_config_url = env.get("VA_TTS_PIPER_CONFIG_URL", "").strip() or (f"{tts_model_url}.json" if tts_model_url else "")

    _prefetch_asset(root, llm_model, llm_url, "LLM model")
    _prefetch_asset(root, tts_model, tts_model_url, "TTS model")
    _prefetch_asset(root, f"{tts_model}.json", tts_config_url, "TTS model config")


def _prefetch_asset(root: Path, rel_path: str, url: str, label: str) -> None:
    if not rel_path:
        return
    rel = Path(rel_path)
    if rel.is_absolute():
        print(f"[build] Skip prefetch {label}: absolute path is not portable ({rel_path})", flush=True)
        return

    target = root / rel
    if target.exists():
        print(f"[build] {label} already present: {rel}", flush=True)
        return
    if not url:
        print(f"[build] Skip prefetch {label}: URL is empty.", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, target, label)


def _download_file(url: str, dst_path: Path, label: str) -> None:
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    last_err: Exception | None = None
    url_variants = _download_url_variants(url)
    for idx, candidate_url in enumerate(url_variants, start=1):
        if idx > 1:
            print(f"[build] Trying fallback URL for {label}: {candidate_url}", flush=True)

        for attempt in range(1, 4):
            try:
                print(f"[build] Downloading {label} (attempt {attempt}/3)...", flush=True)
                ssl_ctx = _download_ssl_context()
                with urllib.request.urlopen(candidate_url, timeout=180, context=ssl_ctx) as response, open(tmp_path, "wb") as out:
                    shutil.copyfileobj(response, out)
                tmp_path.replace(dst_path)
                return
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 404:
                    # A missing file name on HF is deterministic; switch to next URL variant.
                    break
                if _is_ssl_error(e):
                    curl_ok, curl_err = _download_with_curl(candidate_url, tmp_path)
                    if curl_ok:
                        tmp_path.replace(dst_path)
                        return
                    last_err = RuntimeError(f"{e}; curl fallback failed: {curl_err}")  # noqa: TRY004
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_ssl_error(e):
                    curl_ok, curl_err = _download_with_curl(candidate_url, tmp_path)
                    if curl_ok:
                        tmp_path.replace(dst_path)
                        return
                    last_err = RuntimeError(f"{e}; curl fallback failed: {curl_err}")  # noqa: TRY004
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass

            time.sleep(0.8 * attempt)

    raise RuntimeError(
        f"Failed to download {label} from {url}. "
        f"Tried {len(url_variants)} URL variant(s). Last error: {last_err}"
    ) from last_err


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


def _bundle_optional_assets(root: Path, release_dir: Path) -> None:
    env = _load_env_values(root)

    llm_model = env.get(
        "VA_LLM_MODEL_PATH",
        ".cache/voice_assistant/models/llm/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )
    tts_model = env.get(
        "VA_TTS_PIPER_MODEL",
        ".cache/voice_assistant/models/piper/ru_RU-irina-medium.onnx",
    )

    copied = 0
    copied += _copy_asset_if_exists(root, release_dir, llm_model, label="LLM model")
    copied += _copy_asset_if_exists(root, release_dir, tts_model, label="TTS model")
    copied += _copy_asset_if_exists(root, release_dir, f"{tts_model}.json", label="TTS model config")
    if copied == 0:
        print("[build] No local model assets found to bundle.", flush=True)


def _load_env_values(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        env_path = root / ".env.example"
    if not env_path.exists():
        return {}

    result: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key] = value

    # Explicit process env should win over file values.
    for key, value in os.environ.items():
        if key.startswith("VA_") or key.startswith("LMSTUDIO_"):
            result[key] = value
    return result


def _copy_asset_if_exists(root: Path, release_dir: Path, raw_path: str, *, label: str) -> int:
    if not raw_path:
        return 0
    rel_path = Path(raw_path)
    if rel_path.is_absolute():
        print(f"[build] Skip {label}: absolute path is not portable ({raw_path})", flush=True)
        return 0
    source = root / rel_path
    if not source.exists():
        return 0

    target = release_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"[build] Bundled {label}: {rel_path}", flush=True)
    return 1


def _write_launchers(release_dir: Path, binary_name: str) -> None:
    sh_path = release_dir / "run.sh"
    sh_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "$ROOT_DIR"\n'
        f'"./{binary_name}" "$@"\n',
        encoding="utf-8",
    )
    os.chmod(sh_path, 0o755)

    bat_path = release_dir / "run.bat"
    bat_path.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        "cd /d %~dp0\r\n"
        f"{binary_name} %*\r\n",
        encoding="utf-8",
    )


def _platform_tag() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    if os.name == "nt":
        return "windows"
    return sys.platform


if __name__ == "__main__":
    raise SystemExit(main())
