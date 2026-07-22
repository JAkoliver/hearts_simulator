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

## First cloud-generated expert iteration (2026-07-18/19, pod qbc4d63x46myg7)

Operationally clean end to end; strength verdict: gate FAIL.
- Generate (H100+AOTI, queue over tunnel): 3,500 deals / 14 chunks, seed
  20260720, 2.30 s/deal sustained, zero retries, every shard validated
  twice. Pod 2.35 h = $7.02.
- Distill (local 4090): 210,966 records, 3 epochs, teacher match 58.4%,
  holdout 59.6%.
- Gates (local, measured v5 durations for the record): raw guard 2,500
  deals = 551 s (+0.003, PASS); search gate 600 deals = 1,274 s
  (+0.450, p 0.90, FAIL). Baseline unchanged.
- Iteration wall-clock ~3.5 h (vs ~7 h all-local) at $7.02 cloud.
Ops incident: a scratchpad gate driver without the __main__ guard
crash-respawned pool workers for 2.3 h; driver now committed
(cloud/run_gates.py, 38eb8b3) and watchdog discipline adopted.

## Gate re-powering + candidate re-verdict (2026-07-19, commit 7f484f9)

search_gate_deals 600 -> 2400, sharded 4-way in orchestrator (stride-1M
pair seeds, SE + n in every verdict). Measured: n=2400 gate = 4,927 s
(~82 min) on the 4090 with 8 concurrent SearchEval processes; promotion
bar moves from -0.51 to ~-0.26 at alpha=0.05.

First full-power verdict - cloud-iter-0 candidate re-gate:
**+0.679 (SE 0.169, n=2400, t=4.0): definitively WORSE, not noise.**
The n=600 FAIL is confirmed and sharpened. Diagnostic value: raw play
was dead-even (+0.003) while searched play degraded ~0.7 - the
"distillation erodes search-carrying properties" pattern visible within
a single 3,500-deal same-teacher iteration. Rejected candidate archived
as hearts_model_last_rejected.pth.

## PPO-on-v5 round 2 under the powered gate (2026-07-19/20, 4 trials)

run_loop campaign on the v5-M baseline, all gates n=2400 / K=32 /
alpha=0.05 (bar ~ -0.26). Trial wall-clock ~2.1-2.25 h each, matching the
ledger phase estimates (train 35-45 min, raw guard ~9 min, gate ~82 min).

| Trial | Mutation (vs base)              | Raw guard (head-to-head)  | Search gate (n=2400)  | p     |
|---|---|---|---|---|
| 1 | lr 7e-6, clip .166, lam .917, aux .582 | -0.659 (t=-4.86, sig) | -0.078 (SE 0.165) | 0.318 |
| 2 | lam .891                        | -0.626 (t=-4.57, sig)     | -0.146 (SE 0.163)     | 0.184 |
| 3 | lr 7e-6, gamma .993             | -0.775 (t=-5.81, sig)     | -0.062 (SE 0.157)     | 0.348 |
| 4 | lr 1.2e-5                       | -0.187 (t=-1.32, ns)      | -0.026 (SE 0.164)     | 0.437 |

**Pooled search delta: -0.078, SE 0.081 (9,600 paired deals); 95% CI
[-0.24, +0.08].** All four FAIL individually; pooled effect
indistinguishable from zero and the CI excludes every historically
promotion-worthy effect size (-0.5..-1.2). Raw-gain size does not
predict search delta (strongest raw trial had the weakest search lean).
Verdict: this recipe (PPO from the current baseline, single-trial
mutations) does not move SEARCHED strength; its consistent -0.6..-0.8
raw gains vs the training opponent do not carry through search.
Full analysis + next-step options: docs/ppo_v5_round2_findings.md.

## Diagnostics A + B (2026-07-21)

- **B, promotion re-gate** (pre-PPO milestone vs post-PPO baseline,
  n=2400): pre-PPO **+1.025 (SE 0.180, t=5.68)** worse - the 07-15 PPO
  promotion was real and the n=600 gate UNDERestimated it (-0.712).
  Wall: 4,942 s, right on the 82-min ledger figure.
- **A, neutral raw eval** (new neutral_raw_eval.py, n=2500 paired deals
  vs 3x v3-m7 anchors, ~131 s at 12 workers): trial-3-repro candidate
  **-0.654 (SE 0.143)**, trial-4 rejected candidate **-0.636 (SE
  0.144)** - PPO raw gains are genuine strength, not opponent
  exploitation. Trial-3 repro train run: 73.4 min (config lr 7e-06 /
  gamma 0.993).
- Verdict + raw-line promotion recommendation (pending user decision):
  docs/ppo_v5_round2_findings.md.

## Raw-line promotion adopted + first promotion (2026-07-21)

User approved the raw-line design. orchestrator.main now promotes on the
neutral raw gate (n=2500, alpha=0.05, ~2.5 min at 12 workers) with the
n=2400 search gate demoted to a non-regression guard (one-sided 95% UB
vs +0.3 margin); promote_raw_line.py added as the manual driver.

**First raw-line promotion - cand_A_trial3repro.pth (trial-3 config
repro):**
- Neutral raw gate: **-0.619 (SE 0.141, n=2500, p=0.00001)** PASS -
  third independent replication of the ~-0.63 effect (fresh seed).
- Search non-regression guard: -0.070 (SE 0.156, n=2400), UB +0.187
  vs +0.3 -> PASS (searched play unchanged, leaning better).
- Milestone: hearts_model_milestone_1784674184.pth; traces re-exported
  and hash-verified. Known caveat: the candidate's Adam optimizer state
  was not preserved (diag-A restore) - the first post-promotion PPO
  trial starts from the old baseline's moments.
- Raw-line trial economics: raw-FAIL trials cost ~1.3 h (train ~74 min
  + 2.5 min gate); only raw-PASS trials pay the 82-min guard (~2.7 h).

## v5-L distill throughput (2026-07-21, measured on the 4090)

fp32 forward_train+backward, batch 1024, synthetic batches:
- d=384 L=8 (14.4M params): 197 ms/step = 5,199 rec/s -> **9.4 min per
  2.93M-record epoch** (11.8 GB reserved)
- d=448 L=8 (19.6M params): 247 ms/step = 4,152 rec/s -> **11.8 min per
  epoch** (13.8 GB reserved)

**Batch 2048 does NOT fit v5-L in fp32** (~21+ GB activations for d=384
L=8): Windows CUDA sysmem fallback silently oversubscribes instead of
OOMing - 24 GB "allocated", 0% GPU util, idle wattage, ~1000x slowdown.
distill.py defaults to batch 2048 (fine for v5-M d=320 L=6 at ~13 GB);
pass --batch 1024 for v5-L-size nets, or add bf16 autocast.

## v5-L from-bank distills: data is the constraint (2026-07-22 overnight)

Both from-scratch distills from the 2.1M-record bank (107 files, all
decision dirs; leaf/oracle dirs auto-excluded by record size) failed the
neutral raw gate by ~4 points vs the twice-promoted baseline:

| Candidate | Holdout match | Neutral raw vs baseline (n=2500) |
|---|---|---|
| v5-L d=448 L=8, 6 epochs (52 min) | 51.9% | **+3.988 (SE 0.175)** |
| + one PPO step (2.35 h, minibatch 1024) | - | **+3.102 (SE 0.175)** |
| control d=384 L=8, 10 epochs (52 min) | 53.8% | **+3.776 (SE 0.174)** |

Size/epochs changed nothing (both plateau ~52-54% teacher match; value
head overfits from ~epoch 4). The bank's teacher predates two promotions;
its records cannot produce a net that beats today's baseline. Notable:
the PPO step on the weak fresh net gained **-0.89** - the largest
single-step PPO gain measured (vs -0.6 near the ceiling), consistent
with PPO gains scaling with distance from the ceiling. Baseline
hash-verified untouched after every phase.

**Fresh-teacher bank started 2026-07-22 06:30am:** 3,250 deals local
(selfplay_data/0722_fresh_iter0, seed 20260722, current teacher trace,
K=64/pass-k 24/14 threads) - first installment of the >=12k cumulative
fresh bank; ~5.7 h at the 6.32 s/deal ledger rate.

## Raw-line round 1 vs the NEW baseline (2026-07-21 evening, 3 trials)

| Trial | Neutral raw delta (n=2500) | p |
|---|---|---|
| 1 | -0.159 (SE 0.129) | 0.109 |
| 2 | +0.009 (SE 0.128) | 0.529 |
| 3 | -0.128 (SE 0.129) | 0.162 |

**Pooled: -0.093, SE 0.074 (7,500 paired deals)** - vs the -0.62..-0.65
per-trial gains every candidate showed against the PREVIOUS baseline.
The one-step PPO raw gain is substantially ONE-SHOT: once promoted into
the baseline, further PPO steps recover ~5x less. Caveat: all three
trials warmed up from stale Adam moments (promotion did not carry the
candidate's optimizer state). Loop stopped after trial 3 by user
request; state verified clean (config restore needed - the kill raced
the loop's 5s sleep and caught a mutated config pre-launch).
