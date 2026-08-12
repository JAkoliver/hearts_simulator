# REPRODUCING — builds, gates, data, and the demo

> STATUS: DRAFT - not yet released; project ongoing.

Toolchain pins and procedures for rebuilding the system and re-running
its measurements. On conflict, the build files (CMakeLists.txt,
cloud/Dockerfile) and docs/RELEASE_PLAN.md sec. 4 are canonical.

## 1. Windows build (the development machine)

- **Compiler:** MSVC (C++17; the project develops on Windows 11 with
  Visual Studio's toolchain via CMake ≥ 3.14).
- **libtorch: 2.12.1+cu126**, unzipped to `./libtorch` at the repo
  root. CMake wires Torch manually instead of `find_package(Torch)` —
  the CUDA-enabled TorchConfig.cmake demands a full CUDA Toolkit at
  configure time, but the build only links against the import libs
  libtorch ships (no .cu compilation) (CMakeLists.txt comment).
- **third_party/cuda_include:** NVIDIA CUDA headers (needed by
  ATen/cuda/CUDAGraph.h; all CUDA symbols still come from libtorch's
  DLLs). NVIDIA-licensed, therefore NOT tracked (RELEASE_PLAN
  sec. 4.3): materialize once with `python scripts/fetch_cuda_headers.py`
  (four wheels; see the clean-tree note below).
- FTXUI v5.0.0 is fetched by CMake for the terminal game
  (FetchContent); `-DHEARTS_CLOUD_ONLY=ON` skips it and builds only the
  headless targets SelfPlayGen + SearchEval.
- Configure/build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
  then `cmake --build build --config Release`.
- Python side: the pybind module (hearts_env) builds from the same
  tree; training/eval scripts run under CPython 3.13 with the exact
  pins in **requirements.lock** (pip freeze of the working env; torch
  2.12.1+cu126 matching the libtorch flavour). Scope honestly: the
  lock makes training/eval run on a clean machine — the bit-identical
  data contract lives in the C++/libtorch side and rule #14's caveats.
- **Clean-tree build verified 2026-08-12:** fresh `git clone` +
  `python scripts/fetch_cuda_headers.py` (four NVIDIA wheels: runtime/
  nvcc/cccl 12.6.77 + cublas 12.6.4.1, 1,679 headers) + libtorch drop
  + `cmake ... -DHEARTS_CLOUD_ONLY=ON` built SelfPlayGen.exe with no
  repo-local state. The headers are NOT tracked (NVIDIA license
  boundary); CMake fails at configure time with the exact fetch
  command if they are missing.

## 2. Linux build (cloud/Dockerfile)

`docker build -f cloud/Dockerfile -t hearts-worker .` from the repo
root. Two-stage: Ubuntu 22.04, libtorch 2.12.1+cu126 zip, plus the
version-matched CUDA runtime wheels folded into libtorch/lib (the 2.12
Linux zip no longer bundles them — the Dockerfile lists the exact wheel
set). Runtime image carries only SelfPlayGen, SearchEval, worker.py,
and numpy for shard validation; `--gpus all` injects the driver.
cloud/Dockerfile.aoti adds the AOTInductor compile environment
(cloud/export_aoti.py produces per-arch .pt2 packages; serving is
enabled with HEARTS_SRV_AOTI, Linux only).

## 3. Running a gate

All gates run locally, both arms on the same machine (rule #14,
docs/experiment_rules.md). Launch discipline: unbuffered, output to a
file, via a dedicated script (rule #8).

- **Neutral raw gate:** `python neutral_raw_eval.py` — candidate and
  baseline each seated vs 3x neutral anchors on identical deals,
  n=2,500 paired deals (~2.5 min at 12 workers on the 4090; ledger
  2026-07-21).
- **Match gate:** `python match_eval.py --cand <ckpt> --matches 3200`
  — paired matches-to-100, alternating v3-m7/v4-m10 anchor fields,
  placement t-test + win McNemar (match_eval.py docstring; n=3,200
  ~40 min, ledger 2026-07-28).
- **Search guard:** SearchEval paired per-deal comparison, both arms
  match-aware (556 traces + equity leaves), n=4,800, one-sided 95% UB
  vs +0.3 (ledger 2026-07-28; driver: orchestrator / run_gates.py).
- **Promotion:** only through orchestrator / promote_raw_line.py — they
  bundle gates, milestone copy, optimizer carry-through, and
  hash-verified trace re-export (rule #6).
- Sanity check to trust any paired harness: run an arm against itself;
  every paired delta must be exactly zero (SE exactly 0.000 in a REAL
  comparison means you gated a net against itself — rule #3).

## 4. Regenerating data from seeds

The reproducibility contract: **every shard is reproducible from
(seed, chunk) alone** (cloud/worker.py lease fields; RELEASE_PLAN
sec. 2). SelfPlayGen takes `--seed` and a deal/match quota; multi-chunk
banks use disjoint seed ranges recorded in the ledger per installment
(e.g. the fresh bank's four installments, seeds 20260722 / 30260722 /
40260723 / 50260723 — ledger 2026-07-22/23). Match-mode (v2) files
embed base_seed and thread id in the 32-byte file header
(selfplay_gen.cpp). The N=8000 validation used seed block
20260726 + shard*1e7 with pairs never split across nodes (ledger
2026-07-26); its dataset ships as equity_data/validation_v1/ with an
MD5SUMS manifest (written 2026-08-01 from the shards as collected and
verified on 2026-07-27), so reanalysis needs no regeneration. Verify
shard integrity before any analysis with
`python analyze_validation.py --verify-md5` (exit 0 = all shards OK).

The era-9 defended-game corpus follows the same contract:
ops/run_r2_gen.sh regenerates it from its recorded seed bases (AGG
160M+, SEL 170M+, stride 1M/shard, match index x1000 within a shard —
audited disjoint from every earlier block, script header), attacker
clones identified by md5. validate_r2_corpus.py re-runs the registered
instrument checks and the volume count from the generator outputs
alone. Corpus regeneration is not a paired measurement, so resume after
interruption is match-lossless but not required to be bit-identical to
an uninterrupted run (prereg amendment, 2026-08-08).

Determinism caveats: bit-identical replay holds within a hardware/OS
class (the Linux pilot's A/A run was 20/20 bit-identical; ledger
2026-07-26); across heterogeneous hardware, bf16 argmax flips occur at
near-ties (ledger 2026-07-17) — which is why cross-hardware data is
never mixed inside one comparison (rule #14).

## 5. Running the web app

`python -m uvicorn hearts_web.server:app --host 0.0.0.0 --port 8642`
then open the served page: one human seat vs three AI seats (current
promoted baseline, raw policy) on the exact training match rules
(hearts_web/server.py docstring). Matches append telemetry to
hearts_web/match_logs.jsonl (personal data; not part of the release).

## 6. Hardware notes and expected paces

Development machine: one RTX 4090 (24 GB) under Windows/WDDM. Measured
planning numbers (docs/speed_ledger.md — quote durations only from
there, never mixing the A/B and steady conventions):
- Teacher-trace generation (K=64, pass-k 24, 14 threads, CUDA bf16):
  **6.32 s/deal steady**; 3,500 deals ≈ 6.1 h. v4-m10 teacher: 0.34
  s/deal (ledger header).
- H100 SXM (certified AOTI stack): 2.22 s/deal, $1.84/1k deals at
  2026-07 RunPod pricing (ledger, H100 AOTI session).
- Search-vs-search match comparisons: local optimum 2 concurrent
  shards ≈ 56 pairs/h; a 3090 runs ~38.8 pairs/h (ledger 2026-07-25/26).
- Match gate n=3,200 ≈ 40 min; n=2,400 search gate ≈ 82 min sharded
  (ledger 2026-07-28 / 2026-07-19).
- VRAM: keep InferenceServer's row cap (default HEARTS_SRV_MAX_ROWS
  8192) — it bounds peak activation memory and is the standing fix for
  the Windows driver wedges (ledger 2026-07-30).
- Desktop-sharing profiles: HEARTS_HEADROOM (fractional pacing, also
  honoured natively by SelfPlayGen) and reduced-thread "gentle"
  launches (rule #17).

## 7. Models needed to reproduce headline numbers

See docs/release/model_cards/. Minimum set (RELEASE_PLAN sec. 3):
the v3-m7 and v4-m10 anchors (they define every opponent field), the
frozen match-blind reference hearts_ai_search_ref_matchblind_20260724
(md5 a1a0be31 — the N=8000 comparator), hearts_equity.pt, and the
final champion's .pth + traces.

Cross-references: what the components are — docs/release/
ARCHITECTURE.md; what the gates mean — docs/release/METHODOLOGY.md.
