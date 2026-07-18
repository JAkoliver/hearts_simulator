# Generation speed ledger — measured figures only

Settings for every entry: v5 teacher trace, K=64, pass-k 24, 14 threads,
CUDA bf16, single process. Two measurement conventions, never mixed:
- **A/B**: 50 deals, seed 4242 (startup-inflated; good for controlled diffs)
- **Steady**: long-run average (seed 777 for 400-deal runs; the honest
  planning number)

| Config (chronological) | A/B 50-deal | Steady (long-run avg) |
|---|---|---|
| Power-of-2 buckets, per-launch autocast (commit 3422383) | 693 s = 13.86 s/deal | 12.20 s/deal (200-deal, seed 4242)* |
| + finer buckets + persistent autocast (d790ef8) | 489 s = 9.78 s/deal | 9.56 s/deal (400-deal, seed 777) |
| **+ SDPA fused attention (24bb45d) — current production** | **328 s = 6.56 s/deal** | **6.32 s/deal (400-deal, seed 777, 2026-07-17)** |

Current-production steady detail: 2527 s / 400 deals; per-100-deal bins
7.03 / 6.45 / 6.02 / 5.77 (usual startup skew, no decay); 76,122 launches;
record count 24,424 — identical to the pre-SDPA seed-777 run, i.e. the
teacher's play is unchanged at this seed.

*Directly measured (2440 s / 200 deals, the pre-change characterization
run), but 200 deals at seed 4242 rather than 400 at seed 777 like the other
two steady entries - the original config was superseded before a
same-length/same-seed run existed. Treat ratios against it as ~±5% soft.

**Measured steady speedups (same-convention ratios):**
- vs previous steady 9.56: **1.51×** (clean: same seed, length, method)
- vs original-config steady 12.20: **1.93×** (soft: see * above)
- The often-quoted 2.11× (693→328 s) is the A/B convention vs the 13.86
  s/deal mark; dividing steady 6.32 into A/B 13.86 would mix conventions
  and overstate the gain — do not do it.

**Projection retired:** the ~6.4 s/deal projection (9.56 × 328/489) measured
in at 6.32 — confirmed within 1.5%. From here on, only these measured
figures are quoted.

**Planning numbers (steady 6.32 s/deal):**
- 3,500-deal generation: ~6.1 h — 2,500 deals: ~4.4 h
- v4m10 teacher for reference: 0.34 s/deal (100-deal run, 2026-07-16)

This ledger is the local baseline for all H100 cost/speed comparisons
(cloud/REQUIREMENTS.md R5).

## H100 validation session (2026-07-17, RunPod Secure, H100 SXM $2.99/hr)

| Metric | Measured |
|---|---|
| Steady generation (300-deal, seed 777, threads 52) | **3.24 s/deal** = 1.95× local 6.32 |
| 50-deal A/B seed 4242 (threads 14/26/52/96) | 184/174/169/170 s — GPU-bound, threads ~flat |
| $ per 1,000 deals (on-demand) | **$2.69** |
| 3,500-deal generation | ~3.15 h ≈ $9.42 (vs ~6.1 h local, $0 cloud) |
| Queue-over-tunnel chunk (250 deals, sha-verified model) | PASS, 3.23 s/deal, validated both sides |
| GPU util (trace, 5 s samples) | generation 85% mean / 525 W; gate workload 52% / 263 W |
| Session actual spend | 2.50 h × $2.99 = **$7.48** |

## H100 AOTI session (2026-07-18, RunPod Secure, H100 SXM $2.99/hr, pod 2)

| Metric | Measured |
|---|---|
| AOTI forward, 2048 rows | 10.79 ms (JIT H100: 15.6; local JIT: 45.5) |
| 50-deal A/B seed 4242 | JIT 166 s → **AOTI 119 s = 1.39×** (threads 52≈96, still flat) |
| **Steady (300-deal, seed 777, AOTI)** | **2.22 s/deal = 1.46× H100-JIT, 2.85× local 6.32** |
| $ per 1,000 deals | **$1.84** — 3,500-deal generation ≈ 2.2 h ≈ $6.45 |
| GPU during AOTI generation | 92% SM, 59% mem-BW, **667 W of 700 W cap** (power-limited; JIT was 535 W) |
| Gate wall rate (8 shards) | 0.47 s/deal (vs 0.95 single-process JIT) |
| **R1 gate (AOTI stack vs local reference)** | **PASS**: n=8,000, seed 20260719, mean **−0.139**, SE 0.123, UB **+0.064** < +0.30 |
| Session spend | ~2.55 h ≈ **$7.60** |

The AOTI stack is certified for real generation. Remaining H100 headroom is
small: the card runs at its power cap during generation; server-overlap
(L2) may recover the 19-35 ms queue waits (~10-20%), nothing structural.
Iteration economics now: 3,500-deal generation = 2.2 h/$6.45 on one H100
(or ~35 min across four), distill+gates local ≈ 1 h → **~3 h iterations
vs ~7 h all-local**.

**R1 cross-hardware gate, session 1 (JIT stack): FAIL (on power, not effect).** n=3,000 paired
deals seed 20260718: mean **−0.010** (dead parity, H100 nominally better),
SE 0.2072, one-sided 95% UB +0.331 vs +0.30. Cross-hardware bf16 flips
decorrelate the pairing (per-deal delta std 11.35 vs the ~7.6 the criterion
assumed), so n=3,000 cannot certify the +0.30 margin. Per R1: cloud data
feeds no real iteration yet. Proposed fresh re-attempt (needs approval):
n=8,000, new seed, ~2.2 h ≈ $6.50 → SE ≈ 0.127, passes if true parity holds.
Session learnings committed: SearchEval thread-pool pin (e973243) - the
default libtorch pool stalls entirely on a 224-core box.

## Queue-wait investigation (2026-07-18, local 4090, commit b3f1a55)

Verdict: the 19-35 ms server queue waits are INTRINSIC queueing on a
saturated GPU (requests wait behind in-flight forwards), not overhead.
- P1 pinned staging + async copies: neutral (319 s == 319 s, waits flat).
- P2 two-slot pipeline: NEGATIVE locally (355 s, waits 23 -> 51 ms) -
  two-deep queueing costs more latency than the recovered CPU gaps.
Both env-gated off (HEARTS_SRV_STAGE / HEARTS_SRV_PIPE). Optional cheap
follow-up: 15-min HEARTS_SRV_PIPE=1 re-test on H100+AOTI in a future paid
session (faster forwards there = proportionally larger CPU gaps).
