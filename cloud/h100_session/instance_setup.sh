#!/bin/bash
# H100 validation session - on-instance setup (cloud/RENTAL_PLAN.md).
# Runs inside a RunPod pod (ubuntu 22.04 base with NVIDIA driver). Mirrors
# cloud/Dockerfile exactly (R6 parity), just executed directly in the pod
# since RunPod pods are themselves containers.
#
# Expects in /workspace: hearts_src.tar (git archive of the repo commit),
# hearts_ai_search.pt, hearts_ai_grandmaster_v3_milestone7.pt
set -euo pipefail
# PEP 668 (ubuntu 24.04 images): allow system-site pip installs; the env var
# form is ignored by older pips, unlike the --break-system-packages flag
export PIP_BREAK_SYSTEM_PACKAGES=1

cd /workspace
apt-get update
apt-get install -y --no-install-recommends g++ cmake make curl \
    ca-certificates unzip python3-pip python3-numpy

LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.12.1%2Bcu126.zip"
if [ ! -d /opt/libtorch ]; then
    curl -fSL -o /tmp/libtorch.zip "$LIBTORCH_URL"
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
    pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
        nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12
    PYNV=$(python3 -c "import glob; print(glob.glob('/usr/local/lib/python3.*/dist-packages/nvidia')[0])")
    cp -a "$PYNV"/*/lib/*.so* /opt/libtorch/lib/
fi

mkdir -p /workspace/hearts && tar -xf hearts_src.tar -C /workspace/hearts
cd /workspace/hearts
ln -sfn /opt/libtorch libtorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEARTS_CLOUD_ONLY=ON
cmake --build build -j"$(nproc)" --target SelfPlayGen SearchEval

export LD_LIBRARY_PATH=/opt/libtorch/lib
echo "=== setup complete; GPU check: ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
./build/SelfPlayGen 2>&1 | head -1 || true   # prints "--model is required"
