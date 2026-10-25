"""Launch this project's isolated ComfyUI on the RTX 4090 only."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY = ROOT / "ComfyUI"
VENV_PY = COMFY / ".venv" / "Scripts" / "python.exe"
GPU_UUID = "GPU-eed52936-813f-8d68-1654-bfb56cb42bc3"


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

    if any(arg.split("=", 1)[0] in {"--cuda-device", "--default-device", "--cpu", "--directml"} for arg in extra_args):
        raise SystemExit("Device overrides are disabled: this deployment requires its RTX 4090.")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    devices = dict(line.strip().split(", ", 1) for line in result.stdout.splitlines() if line.strip())
    if "RTX 4090" not in devices.get(GPU_UUID, ""):
        raise SystemExit(f"Required RTX 4090 is unavailable: {GPU_UUID}")
    # torch's CPU allocator (mimalloc 2.1 in c10.dll) ignores a failed commit on
    # memory it reserved for an arena and then writes into it, so a commit-limit
    # miss kills the server with an access violation instead of raising "not
    # enough memory". Without arenas every segment is reserved and committed in
    # one VirtualAlloc, which fails cleanly. SHOW_ERRORS logs such misses.
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUID,
                   "MIMALLOC_DISALLOW_ARENA_ALLOC": "1", "MIMALLOC_SHOW_ERRORS": "1"}

    cmd = [
        str(VENV_PY),
        str(COMFY / "main.py"),
        "--listen",
        args.host,
        "--port",
        str(args.port),
    ] + extra_args
    print(
        f"ComfyUI device: {devices[GPU_UUID]} ({GPU_UUID}); visible as cuda:0; port: {args.port}",
        flush=True,
    )
    return subprocess.call(cmd, cwd=COMFY, env=environment)


if __name__ == "__main__":
    sys.exit(main())
