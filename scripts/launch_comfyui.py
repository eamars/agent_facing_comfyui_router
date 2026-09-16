"""Launch this project's isolated ComfyUI with normal device selection."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY = ROOT / "ComfyUI"
VENV_PY = COMFY / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8191)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not VENV_PY.is_file():
        raise SystemExit(f"Project venv not found at {VENV_PY}; run setup_comfyui.ps1 first")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = [
        str(VENV_PY),
        str(COMFY / "main.py"),
        "--listen",
        args.host,
        "--port",
        str(args.port),
    ] + extra_args
    print(
        f"ComfyUI will use its default device selection; port: {args.port}",
        flush=True,
    )
    return subprocess.call(cmd, cwd=COMFY)


if __name__ == "__main__":
    sys.exit(main())
