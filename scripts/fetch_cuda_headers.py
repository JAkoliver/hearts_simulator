"""Fetch the NVIDIA CUDA headers the C++ build includes.

The headers span FOUR pip wheels (versions matched to the libtorch
2.12.1+cu126 the project builds against). They are NVIDIA-licensed, so
they are not tracked in this repository; this script materializes them
at third_party/cuda_include/, where CMakeLists.txt expects them. All
CUDA *symbols* still come from libtorch's own libraries - only headers
are needed here.

Wheel set (verified by a clean-clone SelfPlayGen build, 2026-08-12):
  nvidia-cuda-runtime-cu12==12.6.77   cuda_runtime*.h and friends
  nvidia-cuda-nvcc-cu12==12.6.77      crt/ (host_config.h ...)
  nvidia-cuda-cccl-cu12==12.6.77      cub/, thrust/, cuda/, nv/
  nvidia-cublas-cu12==12.6.4.1        cublas*.h

Usage:  python scripts/fetch_cuda_headers.py
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

WHEELS = [
    "nvidia-cuda-runtime-cu12==12.6.77",
    "nvidia-cuda-nvcc-cu12==12.6.77",
    "nvidia-cuda-cccl-cu12==12.6.77",
    "nvidia-cublas-cu12==12.6.4.1",
]
DEST = Path(__file__).resolve().parent.parent / "third_party" / "cuda_include"


def main():
    if (DEST / "cuda_runtime_api.h").exists() and (DEST / "crt" / "host_config.h").exists():
        print(f"already present: {DEST}")
        return 0
    total = 0
    with tempfile.TemporaryDirectory() as td:
        print(f"downloading {len(WHEELS)} wheels ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", *WHEELS,
            "--no-deps", "-d", td,
        ])
        DEST.mkdir(parents=True, exist_ok=True)
        for wheel in sorted(Path(td).glob("*.whl")):
            n = 0
            with zipfile.ZipFile(wheel) as zf:
                for info in zf.infolist():
                    parts = Path(info.filename).parts
                    # wheel layout: nvidia/<pkg>/include/<headers...>
                    if "include" in parts and not info.is_dir():
                        rel = Path(*parts[parts.index("include") + 1:])
                        out = DEST / rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(zf.read(info))
                        n += 1
            print(f"  {wheel.name}: {n} headers")
            total += n
    print(f"extracted {total} headers to {DEST}")
    if not (DEST / "crt" / "host_config.h").exists():
        print("WARNING: crt/host_config.h missing - build will fail; "
              "check the nvcc wheel extraction", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
