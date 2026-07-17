# Hearts cloud generation worker (REQUIREMENTS R6: build/runtime parity).
# Same libtorch 2.12.1+cu126 as local, same engine sources, traces pulled
# sha-verified from the orchestrator at start.
#
# Build FROM THE REPO ROOT:  docker build -f cloud/Dockerfile -t hearts-worker .
# Run:  docker run --gpus all -e HEARTS_QUEUE_URL=... -e HEARTS_QUEUE_TOKEN=... hearts-worker

FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake make curl ca-certificates unzip python3-pip \
 && rm -rf /var/lib/apt/lists/*

# NB: no "cxx11-abi" in the 2.12 file name - cxx11 is the (only) ABI now
ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.12.1%2Bcu126.zip
RUN curl -fSL -o /tmp/libtorch.zip "$LIBTORCH_URL" \
 && unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

# The 2.12 Linux zip no longer bundles the CUDA runtime libs (libtorch_cuda's
# DT_NEEDED wants cudart/cublas/cudnn/cusparse/cufft/cufile/cusparseLt/nccl/
# curand/nvrtc/cupti). Pull the version-matched wheels from the cu126 index
# and fold their .so files into libtorch/lib - one dir serves link + runtime.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
    nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
 && cp -a /usr/local/lib/python3.10/dist-packages/nvidia/*/lib/*.so* /opt/libtorch/lib/

WORKDIR /src
COPY CMakeLists.txt HeartsEnv.hpp SearchPlayer.hpp TreeSearchPlayer.hpp \
     InferenceServer.hpp selfplay_gen.cpp search_eval.cpp ./
COPY third_party/cuda_include ./third_party/cuda_include
RUN ln -s /opt/libtorch /src/libtorch \
 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEARTS_CLOUD_ONLY=ON \
 && cmake --build build -j"$(nproc)" --target SelfPlayGen SearchEval

FROM ubuntu:22.04
# libtorch "shared-with-deps" carries its own CUDA runtime libs; the NVIDIA
# container runtime (--gpus all) injects the driver. Python is for worker.py
# (stdlib) + numpy (shard validation).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-numpy libgomp1 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python
COPY --from=build /opt/libtorch/lib /opt/libtorch/lib
COPY --from=build /src/build/SelfPlayGen /src/build/SearchEval /app/
COPY cloud/worker.py cloud/shard_check.py /app/
ENV LD_LIBRARY_PATH=/opt/libtorch/lib \
    HEARTS_GEN_EXE=/app/SelfPlayGen \
    HEARTS_WORK_DIR=/work
RUN mkdir -p /work
WORKDIR /app
CMD ["python3", "worker.py"]
