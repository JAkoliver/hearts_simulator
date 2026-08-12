"""Fetch the NVIDIA CUDA runtime headers the C++ build includes.

The headers live in the pip wheel nvidia-cuda-runtime-cu12==12.6.77
(the exact version the project vendored during development). They are
NVIDIA-licensed, so they are not tracked in this repository; this
script materializes them at third_party/cuda_include/, which is where
CMakeLists.txt expects them. All CUDA *symbols* still come from
libtorch's own libraries - only headers are needed here.

Usage:  python scripts/fetch_cuda_headers.py
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

WHEEL = "nvidia-cuda-runtime-cu12==12.6.77"
DEST = Path(__file__).resolve().parent.parent / "third_party" / "cuda_include"


def main():
    if (DEST / "cuda_runtime_api.h").exists():
        print(f"already present: {DEST}")
        return 0
    with tempfile.TemporaryDirectory() as td:
        print(f"downloading {WHEEL} ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", WHEEL,
            "--no-deps", "-d", td,
        ])
        wheel = next(Path(td).glob("nvidia_cuda_runtime_cu12-*.whl"))
        DEST.mkdir(parents=True, exist_ok=True)
        n = 0
        with zipfile.ZipFile(wheel) as zf:
            for info in zf.infolist():
                parts = Path(info.filename).parts
                # wheel layout: nvidia/cuda_runtime/include/<headers...>
                if "include" in parts and not info.is_dir():
                    rel = Path(*parts[parts.index("include") + 1:])
                    out = DEST / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(info))
                    n += 1
        print(f"extracted {n} headers to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
