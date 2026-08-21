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

**Fresh-teacher bank installment 1 COMPLETE (2026-07-22, 06:30-12:16):**
3,250 deals local (selfplay_data/0722_fresh_iter0, seed 20260722,
current teacher trace, K=64/pass-k 24/14 threads): 20,749 s = **6.38
s/deal** (ledger steady 6.32 confirmed again), 182 files, **199,264
records** (61.3/deal, matching the cloud iteration's 60.3). Fresh bank
now 3,250 of the >=12k target.

**Installment 2 (2026-07-22 pm, selfplay_data/0722_fresh_iter0_pm, seed
30260722):** 3,000 deals / 183,936 records / 168 files at 6.28 s/deal,
stopped cleanly at the chunk-12 boundary on user request (partial
chunk-13 files deleted). Sizing lesson: the 3,500-deal target assumed a
12:25pm launch but the run started ~1:57pm - size end-time-capped runs
from the ACTUAL launch timestamp, not the planning timestamp. Fresh
bank total: **6,250 deals / 383,200 records** of the >=12k target.

**Installment 3 (2026-07-23 02:05-08:38am, selfplay_data/
0723_fresh_iter0_am, seed 40260723):** 3,750 deals / 229,920 records /
210 files in 23,591 s = **6.29 s/deal**, zero retries, finished 21 min
inside the 9:00am cutoff (sized from the verified launch clock; 8:55
hard-stop guard armed but not needed). Fresh bank total: **10,000 deals
/ 613,120 records** - one short session from the >=12k target.

**Installment 4 - FINAL (2026-07-23 day, selfplay_data/
0723_fresh_iter0_day, seed 50260723):** 2,500 deals / 153,280 records /
140 files in 15,801 s = 6.32 s/deal, zero retries. **FRESH BANK
COMPLETE: 12,500 deals / 766,400 records across 4 installments (all
current-teacher, disjoint seed ranges), 100% local, $0.** Exceeds the
>=12k prescription from the expert-iteration post-mortem.

## Step 1: powered expert iteration - FAIL (2026-07-23)

Warm-start distill of the current baseline on the full fresh bank
(766,400 records / 12,500 deals, 3 epochs, **4.5 min** on the 4090 -
v5-M at batch 2048 runs ~9,500 rec/s), then the raw-line gate:

**Neutral raw +0.479 (SE 0.144, n=2500, p=0.9996) - significantly
WORSE. Same-lineage distillation degrades the net even with a fresh
teacher and 3.6x the data of the failed 3.5k recipe.** Train teacher
match 56.6%. (Holdout metrics this run are late-trick-biased and not
comparable: 2% tail cuts on 700 small ~1,100-record files keep only
~20 late-trick records each - split by DEAL for honest metrics on
many-small-file banks.)

Pattern now measured three ways: cloud-iter-0 (3.5k deals: raw +0.003,
search +0.68), fresh 12.5k (raw +0.48). Distilling a search teacher
back into its own PPO-sharpened policy net makes it worse - plausibly
because soft value-derived targets UN-sharpen an RL-sharpened policy.
Candidate parked as cand_fresh_iter1.pth. Baseline hash-verified.

## Step 2: three distill variants, all quick-gate FAIL (2026-07-23)

| Variant | Neutral raw vs baseline (n=2500) |
|---|---|
| A: warm-start on fresh bank + **--sharpen 2.0** | **+0.248 (SE 0.139)** |
| B: fresh v5-L d=448, fresh data only (766k) | +5.276 (SE 0.183) |
| C: fresh v5-L d=448, fresh+old mixed (2.87M) | +3.292 (SE 0.168) |

Reads: (1) **sharpening halved the warm-start degradation** (+0.479
unsharpened -> +0.248 at sharpen 2.0) - supports the "soft search
targets un-sharpen an RL-sharpened policy" hypothesis; higher sharpen
untested. (2) v5-L from static distillation cannot reach the baseline
on ANY data mix - the baseline's strength includes two RL steps that
imitation targets do not encode. From-scratch v5-L needs its own
PPO steps, not more imitation. Candidates parked: cand_fresh_sharp.pth,
cand_v5L_fresh.pth, cand_v5L_mixed.pth. Baseline hash-verified.

## Phase 1 match-to-100 infrastructure (2026-07-23 night)

Built + verified (commit follows): HeartsNetV5 match-context extension
(6 appended dims, zero-init projection - extended net BIT-IDENTICAL to
the baseline on 550-dim AND 556-dim inputs, trace path unchanged vs the
deployed .pt), hearts_match_env.py (score carry, 100-termination,
tie-aware placements, pairing-deterministic deal sequences), and
match_eval.py (paired match gate + telemetry rider).

Measured:
- Null calibration (baseline vs itself, 24 matches): ALL paired deltas
  exactly 0, discordant 0:0 - pairing airtight.
- Match gate cost: 60 paired matches = 45 s at 12 workers ->
  **n=800 gate ~= 10 min** (placement SE ~0.04, score-diff SE ~0.9,
  ~110 discordant pairs at n=800).
- Headroom teaser: the score-blind baseline (~2 pts/deal above the v3
  anchors raw) wins only ~30-37% of 4-seat matches (chance 25%) -
  per-deal strength translates weakly to match wins, as the
  score-conditioning thesis predicts.

## Phase 1 COMPLETE: match-mode training + gates (2026-07-24, 77d38b6)

- MatchVecEnv batch wrapper: equivalence-tested vs single MatchEnv over
  14,640 decisions including match resets - bit-identical obs+ctx,
  dones, placements.
- train.py match_mode smoke (SMOKE_TEST, 256 envs): 2,589 deals / 256
  matches in one cycle+drain; learner avg placement 2.006, win 37.8%
  vs pool; belief BCE normal; critic EV -0.755 as EXPECTED (value head
  has never seen match returns - warmup exists for this).
- Orchestrator: match gate promotes when config match_mode=true
  (placement paired t-test, alpha=0.05, n=800 ~= 10 min); neutral raw
  demoted to a telemetry line; search non-regression guard unchanged.
- config keys: match_mode (false until launch), match_reward_scale 4.0,
  match_gate_matches 800, match_gate_alpha 0.05.
Ready for the first score-aware PPO run: set match_mode=true, launch
run_loop per launcher discipline.

## FIRST MATCH-AWARE PROMOTION (2026-07-24 ~03:15, trial 1 of the match era)

Score-conditioned PPO from the zero-init extended baseline, first trial,
25% headroom pace (~3.4 h wall):
- Training: critic EV climbed -0.755 (cold) -> **0.92** - the value head
  fully learned match returns in one 250k-deal run.
- Match gate (promoter): placement **-0.085 (SE 0.041, n=800, p=0.018)**;
  win rate 28.5% vs 26.8% (discordant 109:95, p=0.18).
- Neutral raw telemetry: **-0.233 (SE 0.136, p=0.043)** - match training
  ADDED per-deal strength rather than trading it away.
- Search guard: +0.033 (SE 0.158), UB +0.292 vs +0.3 -> PASS (narrow).
- Milestone 1784888158 (hash 25db6f93...), traces re-exported+verified.
The score-conditioning thesis paid out on trial 1. Loop continues from
the new baseline (match-era compounding question now open).

**Match-era night 1 complete (trials 1-4, ~23:50-12:16): THREE
promotions.** Trial 3: -0.059 (SE 0.038, p=0.059) near-miss FAIL, raw
flat. Trial 4: **-0.087 (SE 0.036, p=0.008), win 39.8% vs 35.9%
(p=0.025)**, raw flat (-0.068 ns), guard UB +0.261 PASS -> milestone
1784920549 (10abe622). Cumulative vs the fixed anchor field: win rate
26.8% -> 39.8%, placement 2.44 -> 2.05, summed placement delta ~-0.31.
Later gains are pure match-strategy (raw telemetry flat after trial 2).
Guard margins all narrow-positive (+0.292/+0.218/+0.261): searched
strength holding. WATCHPOINT: gate anchors are a FIXED v3-m7 field -
anchor-overfit risk compounds with each promotion; diversify anchor
family soon (same caveat as the raw-line era).

## Validation ops: driver wedge, root cause, hardening, concurrency curve
## (2026-07-25/26)

The first N=8000 launch (8 shards x K=256 search-vs-search) WEDGED the
NVIDIA driver: unkillable processes, hung nvidia-smi, zero output from
minute one; required a reboot. Root cause (diagnosed, then confirmed by
fix-and-retest): DirectBackend lacked BOTH server-path cures - its
AutocastGuard cleared the bf16 cast-cache EVERY forward (re-casting
~15MB of weights per call, millions of times) and it never bucketed
batch shapes (K=256 rollouts emit hundreds of distinct multi-MB shapes
per decision as active sims shrink 3328->1). At 8 processes x 2 CUDA
modules each (16 contexts - search-vs-search doubles the count), the
allocation storm livelocked WDDM's serialized kernel path. Fixed in
b929c3d (persistent cast-cache + BucketRowsDirect); selftest equivalence
PASS; 8-way re-test at original conditions = STABLE, no wedge.

Measured search-vs-search concurrency curve (K=64/256-endgame,
10-15 pairs/shard): 2 shards 56 pairs/h, 3 -> 41, 4 -> 41, 8 -> ~15-25
(clipped) - **local optimum = 2 shards ~= 56 pairs/h; the GPU saturates
and extra shards thrash**. True pair cost ~= 2 min (the earlier ~21
s/pair figure was search-vs-RAW and never applied). N=8000 local ~= 6
days; N=2000 ~= 36 h; cloud fan-out (pairs never split across nodes,
1-2 procs/pod) is the only true multiplication.

## Match-aware search: equity pipeline + SPINE GATE HALT (2026-07-25)

Equity data: 30k seeded (105,156 states) + 5k natural holdout (54,360
states; S3/S1/S2 = 652/2,348/2,000 matches). Equity net: ECE BELOW its
clustered noise floor everywhere (agg 0.0037 vs floor 0.0575; strata
0.019-0.023 vs floors 0.054-0.063) - indistinguishable from perfectly
calibrated; Brier 0.614 agg / 0.336 near-terminal; SELECTED over both
frozen lookups (0.645) aggregate and per-stratum.

Probe collection: 200 matches K=64, 11,245 decisions with match context.
**Flip/SNR gate: HALT.** Tension-band raw flip rate 36.8% (floor 5%
passed hugely) BUT SNR 0.41 vs threshold 1.0 - and the deal-point
REFERENCE is itself 0.57: the working system also lives below 1.0, so
the threshold was mis-set; the reference comparison did its job.
Decisive diagnostic - CONFIDENT flips (equity gap > 2xSE at K=64):
**tension 1.59%, runaway 3.13%, early 1.32%; median flip z ~= 0.2-0.35.**
Nearly all raw flips are noise-flips between statistically tied actions.
At deployment K=64, equity scoring would confidently change only ~1-1.5
decisions per match. The halt-default held exactly as designed.

Options surfaced to user (no unilateral proceed): (a) accept halt, park
match-aware search, pivot to Phase 2; (b) adaptive-K probe (K=256 in
tension states, ~6% of decisions, ~2.5h collection) to test whether
confident-flip rate rises toward the floor; (c) tie-break-only
integration (score by equity ONLY where deal-points are within their
own noise - free in expectation, validated by the N=8000 design).

## Match bridge measurement: search vs match-aware raw AT MATCH PLAY (07-24)

SearchEval --match-pair (new C++ mode, 8c934be): deployed search player
(K=64, pass-search, match-BLIND) vs the match-aware raw net (556 trace),
same seat, paired match seeds, 3x v3-m7 anchors, n=200 matches, run
concurrently with trial-7 training at below-normal priority (37 s/match).

**Search still wins matches: win rate 50.5% vs 38.0% (discordant 51:26,
McNemar p=0.006); placement diff -0.175 (SE 0.095, two-sided p=0.068).**
Stats clarifications (2026-07-24 review): the two win rates are per-table
vs the ANCHOR field (not shares of one contest - anchors won the rest:
98/200 at search's table, 122/200 at raw's); tied-firsts (1 search, 2
raw) counted as non-wins symmetrically, moving the 12.5-pt gap <=1 pt
either way. The p=0.006-vs-p=0.068 split is real, not contradictory:
the effect concentrates NEAR the win boundary (placement floor-dist
search 102/35/37/26 vs raw 78/50/43/29): of the +24 firsts, ~15 come
from P2 conversion and ~9 from the bottom half (P3+P4 63 vs 72 - a
12.5% relative reduction, not "nearly unchanged"). The win-indicator
statistic captures this at full power while mean placement dilutes it
against SD 1.35 per-match noise. Verdicts: (1) the search guard's protection of the
search player is JUSTIFIED with data - guard stays; (2) match-aware
search is the queued ceiling configuration - NO additivity assumed
between search's edge and raw's score-awareness (overlap expected: a
stronger per-deal player reaches fewer desperate positions); success
gate = significantly beats match-blind search, nothing more specific.
Trial 7 is treated as UNINFORMATIVE (GPU contention), not evidence
against PPO - the case for pivoting is trial 6's guard veto.

**Trials 5-6 (07-24 afternoon/evening):** T5 FAIL at match gate (-0.043,
SE 0.036, p=0.11 - variance, not saturation). T6: match gate STRONG PASS
(-0.121, SE 0.037, p=0.0006; win 40.2% vs 36.5%, p=0.026) but **FIRST
EVER search-guard rejection: searched per-deal +0.477 (SE 0.159), UB
+0.739 vs +0.3.** Match training has begun trading away search-substrate
quality (the narrow guard margins on promotions 1-3 foreshadowed this).
Guard design question surfaced to user: protect searched per-deal
strength (status quo) vs raw per-deal (T6 raw was FINE at -0.119) vs
match-level search. Status quo stands pending user decision.

**Trial 2 (~08:40): SECOND promotion - match PPO COMPOUNDS.** vs the
trial-1 baseline: placement **-0.133 (SE 0.040, n=800, p=0.00044)**,
win rate **36.0% vs 31.5% (discordant 136:100, p=0.011 - significant)**,
neutral raw telemetry -0.358 (p=0.003), search guard UB +0.218 -> PASS
(more room than trial 1's +0.292). Milestone 1784900322 (9cb0ba9f),
traces verified. Win rate tracking placement = no placement-bias
signature; the win-bonus reward change stays unneeded on current
evidence. Unlike the per-deal axis (second step always died), the match
axis is compounding.

## Step 3: sharpen sweep + staged PPO-finish - FAIL (2026-07-23 evening)

Sweep (warm-start fresh-bank distills, quick gates n=2500 + policy
entropy over 8,192 bank observations):

| Sharpen | Neutral raw | Entropy (nats) |
|---|---|---|
| 2.0 | +0.248 (SE 0.139) | 0.9736 |
| 3.0 | +0.254 (SE 0.142) | 0.9651 |
| 4.0 | +0.470 (SE 0.138) | 0.9534 |

Sharpening saturates at 2.0; 4.0 is actively worse WITHOUT meaningful
entropy collapse (0.95 vs 0.97) - over-sharpening degrades the policy
directly, not via exploration loss. Winner (entropy tie-break): 2.0.

Staged PPO-finish on the 2.0 candidate (3 x 85k games, warmup
20k/0/0, per-stage neutral-raw gates vs the snapshotted baseline):
start +0.248 -> s1 +0.159 (re-measured +0.326 on a second seed; avg
~+0.24) -> s2 +0.337 -> s3 +0.428 (SE ~0.14 each). **255k games of PPO
produced zero-to-negative movement.** The -0.6..-0.9 PPO gain does NOT
transfer to a candidate starting near the baseline's level -
re-confirming PPO-near-ceiling is one-shot; the sharpened distill
apparently consumes the same improvement PPO would have made.

Measurement bug for the record: the driver's in-loop stage gates
compared checkpoints against hearts_model_final.pth, which DURING PPO
stages is the working candidate itself -> +0.000 (SE 0.000)
self-comparisons. Correct out-of-band gates vs model_step3_base.pth are
the numbers above. Symptom to remember: SE exactly 0.000 = you gated a
net against itself.

## Deployed player vs the 2026-07-14 calibration opponent (2026-07-23)

eval_search_pair.py, both sides K=64 search vs neutral anchors, 1,200
paired deals: current hearts_ai_search.pt vs hearts_ai_search_v4m10.pt
(the deployed player the user beat by ~3.3 pts/round on 07-14):
**-1.016 (SE 0.234, t=-4.34, p=0.00002) - the current best is a full
point per deal stronger than the 07-14 opponent.** Matches the
predicted chain (v5~=v4 searched tie, +1.025 PPO gap, raw-line lean).

Verdict: the distill-refresh + PPO-finish recipe CANNOT beat the
current baseline from same-lineage data. All candidates parked
(cand_fresh_sharp{,3,4}.pth, cand_step3_s{1,2,3}.pth + .optim stashes).
Baseline hash-verified untouched. Per user rule: stopped here; fallback
directions (visit-count targets, exploiter league, v5-L RL ladder,
bigger data) await discussion.

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

## 2026-07-26: Cloud pilot pod (validation pre-flight) - secure 4090

Community 4090 pool ($0.34/h) rejected 5 placement attempts ("machine
does not have the resources", disk 20-40GB all tried) - RunPod-side
matcher issue, stock listed. User-approved fallback: SECURE 4090
$0.69/h (pod 8c2qt0eb0snrgj, 96-thread host, driver 570.195).
Total pilot cost ~48 min = ~$0.55.

- On-pod build: cloud/Dockerfile recipe (libtorch 2.12.1+cu126 +
  folded CUDA wheels) on runpod/pytorch ubuntu2404 image. BUILD OK
  ~11 min. Ref model md5 verified on-pod (a1a0be31...).
- A/A sanity (pre-registered): ref-vs-ref, 20 pairs, K=64/256,
  pass-search, CRN. 20/20 pairs BIT-IDENTICAL across arms (11 wins
  each, 0 discordant rows). Harness pairing + Linux determinism
  confirmed. 1,150s = 62.6 pairs/h.
- Pace run (real config: match-aware + equity leaves vs frozen ref):
  20 pairs in 1,181s = 61.0 pairs/h single process, GPU 84% util,
  3.1GB VRAM. Equity leaf scoring adds ~3% over A/A (batched).
  Outcome direction n=20 (NOT part of pre-registered analysis):
  match-aware 8 wins vs ref 13, mean place diff +0.775 vs
  match-aware, 10/20 discordant. Tiny sample; N=8000 decides.
- Fleet math at 61 pairs/h/pod: N=8000 = ~131 pod-h. Community
  $0.34/h = ~$45 (~$52 w/ 15% pace margin); secure $0.69/h = ~$90
  (over $60 budget). Headroom OFF on pods (env unset by design).

## 2026-07-26: Community 3090 pace check + fleet access mechanism

4090 community stock: zero across ~10 placement attempts over ~1h
(3090s place instantly with identical calls -> stock, not config).
Community hosts split public-IP vs not; no API filter. Access solved:
proxy SSH username = machine.podHostId via GraphQL (MCP lacks it);
proxy is PTY-only -> bulk transfer via private GitHub release assets
(tag pilot-bundle-v1), pod pulls with token. Bundle md5-verified.

3090 pace (pod q3l51dolyeraxc, $0.22/h): BUILD OK ~13 min; real
A-vs-B config 10 pairs in 928s = 38.8 pairs/h = 63.6% of 4090 pace.
Cost/pair: 3090 $0.00567 vs 4090 $0.00557 - equal within 2%.
N=8000: ~206 pod-h, ~$45 (+15% margin ~$52), 8 pods ~26h wall.

## 2026-07-26 09:30: N=8000 VALIDATION FLEET LAUNCHED (user-approved)

8x community RTX 3090 @ $0.22/h, projected ~26h, ~$46. 1000 pairs/pod,
seed block 20260726 + shard*1e7, pairs intra-pod (CRN: shared deal +
search seeds, both arms K=64/256, pass-search, bf16, headroom off).
Arm A hearts_ai_search_match.pt + hearts_equity.pt; arm B frozen
hearts_ai_search_ref_matchblind_20260724.pt (md5 a1a0be31); anchors
v3-m7. Bootstrap: private release pilot-bundle-v1, scoped read-only
7-day PAT on pods (broad git token never left local). Pods:
s0 q3l51dolyeraxc s1 g7gqdk4u2ceu8n s2 svwgy2bugvf07n s3 td0t4h4ym8cjza
s4 i0pxminu4ozq2h s5 zm920a1rb7f436 s6 cte4rf0c4emkyz s7 t2wkxs9tyzyunm
Analysis (pre-registered, SINGLE look at N=8000): one-sided McNemar
alpha=0.05 on match wins; S1/S2 strata + dose-response secondaries.

## 2026-07-27: N=8000 VALIDATION COMPLETE - MATCH-AWARE SEARCH WINS

All 8 shards collected (md5-verified), all pods terminated.
PRIMARY (pre-registered, single analysis): match-aware search
48.91% match wins vs frozen match-blind ref 44.47% = +4.44 win-pts
(SE 0.68). McNemar one-sided: discordant 1668 vs 1313 (q=0.373),
p ~ 5e-11 << 0.05. SIGNIFICANT. All 8 shards positive (+2.4..+6.6).
CRN: paired placement SD 1.351 (bridge ref 1.35 - dead on).

Placement structure (the interesting part): A converts P2->P1 AND
P2->P4: P1 3913v3558, P2 1028v1838, P4 1780v969; mean place WORSE
(2.098 v 1.980). Match-aware search trades expected placement for win
probability - exactly the win-equity objective, not generic strength.

EXPLORATORY (deals terciles; deals endogenous, not the registered
dose measure): +13.3 pts short matches, +0.9 mid, -5.9 long.

LIMITATION (owned): fleet binary logged only final outcomes; the
pre-registered S1/S2 strata + flipped-decision dose-response
secondaries are NOT computable from this dataset (needed per-deal
boundary logging). Primary + CRN reporting unaffected.

COST: actual cloud spend 26th-28th = $62.72 total ($26.95 + $34.47 +
$1.30) vs ~$46-48 projected. Overrun ~$14: realized host pace 28-44
pairs/h (pace-check host was fast at 38.8), slow-pipe download idle,
slowest shard 38h wall. Lesson: re-forecast cost mid-run from realized
per-shard pace, alert on drift.

DECISION (roadmap): match-aware search validated as the new search
standard -> Phase 2 teacher = match-aware search; search guard
benchmark should move to match-aware search per guard-evolution step.

## 2026-07-28: Guard evolution + instrumentation + loop relaunch

Post-validation lock-in, all local, $0:
- Rules #16 added (match-aware search = ceiling config); ROADMAP queued
  entry closed as DONE; next-sequence recorded (match PPO -> on plateau,
  ONE gated match-aware expert-iteration experiment).
- SearchEval match CSVs now carry stratum columns (tens/max85/tdeals
  per arm, computed at deal starts) - the S1/S2 gap of the N=8000 run
  cannot recur. Rebuilt; smoke shows sensible S1/S3 flags.
- SEARCH GUARD EVOLVED: when hearts_equity.pt + hearts_ai_search_match.pt
  exist, both guard arms run match-aware (candidate traced at 556,
  baseline = match trace, --equity-model both sides; single-deal ctx so
  K stays 64). Promotion path now also re-runs export_match.py so the
  guard baseline tracks the champion. Null calibration: +0.000 exact
  (SE 0.000, n=8, intended self-comparison).
- MATCH GATE ANCHORS DIVERSIFIED: matches alternate v3-m7 / v4-m10
  anchor fields by match index (_V4Seat 550-prefix adapter). Null
  calibration exact-zero, pairing intact.
- run_loop RELAUNCHED match-mode under evolved guard, headroom 0.25,
  log logs/run_loop_20260728_match.log; baseline verified 10abe622
  (3rd match-era promotion) pre-launch.

## 2026-07-28: MATCH GATE RE-POWERED n=800 -> n=3200 (user-approved)

Trials 1-4 under the evolved guard: +0.007 / -0.053(p=.060) /
-0.011 / -0.051(p=.069) placement; T4 win rate SIGNIFICANT
(56.5 v 52.9, discordant 133:104, p=.034). Pooled placement
-0.027 +/- 0.017 (p~.06, n=3200) - real sub-bar effect, gate was
coin-flipping (43% power vs true -0.05). At n=3200: SE 0.017,
bar -0.028, 90% power vs -0.05, ~55% vs -0.03; gate ~40 min vs
~2h training. Same lesson as the 07-19 search-gate re-power (600->
2400): don't half-power. config + config_backup both updated (trial
5 already in flight gates at 800; trial 6+ at 3200).

## 2026-07-28: 4TH MATCH-ERA PROMOTION - first under the evolved regime

First n=3200 gate-passing trial promoted end-to-end: match gate
-0.029 (SE 0.017, p=0.0456; n=800 would have coin-flipped this),
win 52.1 v 50.9; EVOLVED match-aware search guard first production
run: +0.029 (SE 0.160, n=2400) UB +0.292 vs +0.3 - PASS by 0.008
(second consecutive knife-edge; false-veto analysis says n=2400
passes a dead-neutral candidate only ~61% - re-power to 4800
recommended, awaiting user). Milestone 1785273667; new baseline
cbfde942 (supersedes 10abe622). export_match.py auto-ran on
promotion (traces 14:21) - guard baseline tracks the new champion.
Loop continues.

## 2026-07-28: SEARCH GUARD RE-POWERED n=2400 -> n=4800 (user-approved)

False-veto fix: at n=2400 (SE 0.160) a dead-neutral candidate passes
the +0.3-UB guard only ~61%; both evolved-guard-era passes cleared by
0.008. At n=4800: SE ~0.11, neutral passes ~86%. Margin unchanged
(+0.3 is a tolerance judgment; the noise was the problem). Cost lands
only on match-gate passers (~1 in 5 trials), ~+1.5h those trials.
Applies from the next trial to reach the guard.

## 2026-07-29: 5TH MATCH-ERA PROMOTION - re-power directly vindicated

Best-shaped candidate of the campaign, all three gate metrics aligned:
placement -0.031 (p=.0247), WIN 53.3 v 50.5 (disc 499:407, p=.0012),
score -0.98 (p=.0078). Guard n=4800: +0.072 (SE 0.113) UB +0.258
PASS with margin 0.042 - at the old n=2400 (SE~0.16) UB would have
been ~+0.335 = FALSE VETO. The n=4800 re-power saved this promotion.
Milestone 1785322724. Since re-powers: 3 gate-passes in 3 trials,
2 promotions + 1 correct substrate veto.

## 2026-07-29: Match-aware expert-iteration TOOLING BUILT + VERIFIED

SelfPlayGen --match (agent-built, reviewed, smoked): 4x match-aware
SearchPlayer seats (556 trace + equity leaves + k-endgame), score
carry, per-deal SetMatchContext all seats, ctx layout verified ==
SearchPlayer::WriteCtx (rotated totals/100 x4, deals/20, (100-mx)/100);
824B records (obs u8[556] ... reward f4 = (2.5-place)*4 tie-aware,
assigned post-match); --start-totals seeded states (behave-style
rotation, implied deals from sum/26). distill.py --match dtype+loader.
Smokes (CPU, K=4, under training contention): natural 552 rec/match
2297s; seeded 90,88,40,30 180 rec 1132s. Verified: sizes %824=0,
rewards exactly {+-6,+-2}, ctx tails zero-then-evolving (natural) /
seeded-from-start, masks 1-13, pi rowsum 255, loader 659+73 by-match
tails, reward mean -0.003 (zero-sum check). READY: generation run
awaits plateau call + GPU window.

## 2026-07-30: WEDGE RECURRENCE during match-aware generation - ROOT CAUSE

Chunk knife_b wedged ~10:00 (last record write 61 min before detection;
nvidia-smi hangs; PID 778208 unkillable - taskkill returns "no running
instance" while Get-Process still lists it). Same signature as the
2026-07-25 8-shard wedge, but with a SINGLE process this time.

ROOT CAUSE - FIRST DIAGNOSIS WAS WRONG, CORRECTED 2026-07-30 19:30.
WRONG: "the equity model bypasses the b929c3d hardening". It does not
matter: the equity module is never .to(device) and its input tensor is
built on CPU, so ScoreEquity runs entirely on the CPU and cannot touch
VRAM. (Lesson: verify device placement before blaming a code path.)

ACTUAL ROOT CAUSE (code-confirmed): UNBOUNDED BATCH COALESCING in
InferenceServer. The server loop did `batch.swap(queue_)` and passed
the WHOLE queue to RunGroup as one forward - no row cap. Match-aware
generation changed two things at once: --k-endgame 256 makes an
endgame request ~4x a K=64 request (~3.3k rows), and 14 worker threads
can be in endgames simultaneously => single forwards of tens of
thousands of rows. Peak ACTIVATION memory scales with threads x K, and
the caching allocator retains those peak blocks, so VRAM ratchets up
run-over-run (12.8 -> 23.9 GB over 4.5h, twice) until WDDM wedges near
the ceiling. Fits the Linux contrast: the N=8000 fleet ran search_eval
one match at a time per process, so requests never coalesced at that
scale - 38h clean.

FIX (APPLIED + BUILT 2026-07-30): InferenceServer chunks each queue
into groups bounded by max_group_rows_ (default 8192 = ~2 concurrent
K=256 endgame requests; HEARTS_SRV_MAX_ROWS overrides). Applied to the
active simple loop AND LoopPipelined. Peak activation memory is now
bounded regardless of thread count and K. Smoke (1 match, K=8/16, 2
threads, CUDA): RC=0, 732 records, 15,996 launches, mean batch 67.9
rows, VRAM back to 2.0 GB idle after exit.

Data safe (per-match flush): bank 110,109 records (96,632 night-1
natural + 13,477 today: knife_a 120 matches, knife_b partial 4,389).
Recovery needs a reboot (unkillable process holds the GPU).

CONFIRMED (nvidia-smi finally returned after ~8 min): GPU 0% util,
210 MHz idle clocks, 11.5W, but 23,918 MiB VRAM still held by the
zombie. Ties the two symptoms into ONE cause: unbucketed equity
shapes -> CUDA shape-cache growth -> VRAM climb (12.8->23.9 GB over
4.5h on night 1, same ~24GB today) -> WDDM wedge near the ceiling.
Identical mechanism to the pre-b929c3d main-model leak; the equity
path simply was never covered by that fix.

## 2026-07-30 evening: gentle-profile generation + throughput surprise

Gentle profile live (user confirms desktop usable): --threads 5,
BelowNormal, HEARTS_HEADROOM=0.45, HEARTS_SRV_MAX_ROWS=2048, chunked.
VRAM 5.75 GB (vs 23.9 GB pre-fix) - the row cap is the load-bearing
control; GPU util stays ~96% because per-thread pacing does NOT create
global GPU gaps (staggered threads keep it busy). Server-level pacing
would be the correct knob if true GPU headroom is ever needed; not
implemented since the desktop is usable at current VRAM.

THROUGHPUT SURPRISE: 44 s/match at 5 paced threads vs 37 s/match at 14
unpaced threads pre-fix (~19% slower with 1/3 the threads) => the old
unbounded coalesced batches were themselves hurting throughput. The
row cap likely made generation FASTER as well as wedge-proof; worth a
clean A/B at some point.

LAUNCHER BUG (fixed, cost ~10 min): backgrounding `$GEN ... | tail -2 &`
then `wait $!` waits on TAIL, not the generator - chunks reported RC=0
after ~34s and the script stacked overlapping generators. Generators
must run in the FOREGROUND with a direct log redirect. This is rule #2
of launcher-discipline resurfacing in a new disguise (third time).

## 2026-07-31 06:35: expert-iteration GENERATION COMPLETE (bank 235,951)

Window closed itself per rule #17 (nat_d skipped past the 06:30 cutoff,
FAST_ALL_DONE 06:35, GPU free at 2.9 GB). Composition:
  natural  156,244 (66.2%) | leader 29,304 | knife 25,955 | trail 24,448
  seeded/tension share 33.8% (vs ~6% in natural play = ~5.6x enriched)
Paces measured (s/match): knife 25.5 fast / 36.4 gentle; leader 44 fast
/ 61 gentle; trail 37 fast; natural 104 fast / 149 gentle. Full-throttle
+ Normal priority ~30% faster than the gentle profile. NOTE: Windows
Task Scheduler starts tasks at BelowNormal - a priority keeper had to
re-raise each new chunk process (fold into future scheduled launchers).
No wedge, no VRAM growth: peak 12.9 GB at 14 threads (vs 23.9 GB
pre-fix) - the row cap held across ~4.5h of full-throttle generation.

## 2026-07-31 14:27: EXPERT-ITERATION BANK COMPLETE - 333,415 records

Target 330k reached and auto-stopped (user-approved daytime full-speed
window; stopped 2.5h before the 17:00 line). Final composition:
  natural 242,700 (72.8%) | trail 31,040 | knife 30,371 | leader 29,304
  seeded/tension share 27.2% (~4.5x enriched over natural play)
All files trimmed to 824-byte boundaries. GPU released.
Ops bugs this cycle, both recorded for the pattern file: (1) session
restart killed all session-spawned watchers (day2 driver died; its
orphaned generator chunk COMPLETED alone - per-match flush is the
resilience backbone); (2) the 330k stop watcher killed processes by
command-line pattern "gen_fast_day" and SELF-MATCHED - it stopped
generation correctly but died before logging/notifying. Kill patterns
must exclude the killer (match on exe name or exact PID list, never a
substring the watcher itself carries).

## 2026-07-31: MATCH-AWARE EXPERT ITERATION - ONE-SHOT GATE: FAIL (decisive)

Candidate cand_expert_iter1_hard (hard-policy distill, holdout teacher
match 60.1%, tension +10.2 over baseline) LOST the match gate
catastrophically: win 39.9% v 50.3% (discordant 377:710), placement
+0.292 (SE 0.019, ~17 SE worse). Guard skipped as moot; baseline
8a89da90 untouched. Per pre-registration this experiment is CLOSED.

Recipe post-mortem (three distill variants measured before the gate):
- Soft targets (sharpen 2, 8): equity-scored teacher policies are
  near-UNIFORM (P(win) gaps of a few pct) - pow-sharpening a uniform is
  a no-op; both variants UN-SHARPENED the champion (entropy 0.32 ->
  1.04-1.08) and dropped teacher-match BELOW the baseline.
- Hard argmax targets: fixed imitation metrics (60.1% holdout,
  +10.2 tension) but the played strength collapsed anyway. Best
  hypothesis: in the ~73% of states where equity is flat, the teacher's
  ARGMAX IS NOISE - a coin flip between near-equal actions. Hard
  training copies those coin flips and overwrites the baseline's real
  per-deal knowledge. Imitating a teacher whose choices are mostly
  arbitrary ties destroys more than the ~6% of genuinely-informative
  tension decisions add.
LESSON (new closed direction, generalize carefully): distilling a
search teacher requires TARGETS THAT ENCODE PREFERENCE STRENGTH.
Equity-argmax carries none in flat states; equity-soft carries almost
none anywhere. A viable future variant must filter to decisions where
the teacher's equity spread is significant (e.g. flip-confident states
only, ~4-6%) and/or mix a per-deal anchor loss - QUEUED to roadmap as a
NEW experiment (fresh pre-registration; this one-shot is consumed).
FALLBACK ENGAGED: match-mode PPO resumes under the evolved regime.

## 2026-08-01: Correction to the 07-30 wedge entry (documentation audit)

The 07-30 entry's final "CONFIRMED" paragraph re-attributed the wedge
to "unbucketed equity shapes" after the same entry had already
established that diagnosis as WRONG (the equity module runs on CPU).
The correct, code-confirmed cause stands as written in the middle of
that entry: UNBOUNDED BATCH COALESCING in InferenceServer (fixed by
the max_group_rows_ cap, 45821a6). Commit 592078e's title ("root-caused
to the equity path") carries the superseded diagnosis; the fix commit
45821a6 carries the correct one. Found by the release-docs audit.

## 2026-08-12: v6 Stage-2 bank generation COMPLETE (cloud fleet + local tail)

24/24 chunks x 96 matches. Full-bank battery (validate_v3_records.py,
all 768 per-thread files): 1,524,821 records, 2,304 matches, 288
shooter-clone matches (EXACT 1/8 schedule held; attacker seat never
recorded), matches-with-moons shooter/natural 154/1170. ALL V3 CHECKS
PASS. Bank at expert_data/v6_bank/ (per-thread files kept separate).

Measured rates (steady, teacher = deployed flat searcher K=64/256,
pass-k 24, threads 32, bf16 CUDA):
- H100 SXM pod (4x fleet, AP-IN-1): 82-92 min per 96-match chunk
  (~53-57 s/match), 0 retries over 20 cloud chunks.
- RTX 4090 local: 93-98 s/match (chunks 20-23; webapp co-resident for
  20-22, dedicated for 23 - no measurable difference).

Cloud cost: $96.31 FINAL ($96.07 GPU + $0.24 disk) vs the $110 hard
cap; fleet + template deleted same day. Ops incident on the record:
the local orchestrator died ~03:50 (collateral of a webapp restart);
restart requeued 4 in-flight chunks, creating a dup-compute hazard
that would have blown the cap (~$116-120 projected). Resolved by
one-time queue-state surgery (in-flight chunks marked awaiting_upload,
final 4 chunks held for local generation); seeds/geometry/teacher
untouched - data is prereg-exact. The escape hatch (finish locally,
chunk-resumable queue) worked as designed.

NEXT (prereg): Stage 3 - from-scratch HMR3 distillation trainer,
three arms, recipe freeze on holdout only.

## 2026-08-13: v6 Stage 3 COMPLETE - freeze + screens; Stage 4 unlocked

Grid: 3 arms x 2 lrs x 4 epochs (epoch snapshots cover the registered
epochs {2,3,4}; batch amendment 2048->512 documented in v6_distill.py -
WDDM VRAM spill at 2048 gave 7.5 s/step and bit-identical fp32/TF32
epochs; at 512 the same epoch fell 89 min -> 5.1 min). By-match holdout
123 matches / 80,220 records.

Freeze (declared criterion: entropy band [0.22,0.87] then lowest
holdout CE): arm a = lr1e-4 ep3 (CE 0.8413, match 64.2%, entropy
0.832); arm b = lr3e-4 ep4 (CE 0.8381, 64.2%, 0.854); arm c = NO
in-band snapshot (0.877-0.955) -> REGISTERED AMENDMENT (user-approved
pre-unblinding, docs/v6_prereg.md): eligibility waived for the control
arm only, lr3e-4 ep2 frozen (CE 0.8681) with out-of-band flag.
Holdout findings: a-vs-b TIED on imitation (scale bought ~nothing
pre-PPO); b-vs-c separates cleanly (structure helps: CE 0.838 vs
0.868, match 64.2 vs 63.0).

Screens (registered n=2500, instrument SE 0.14-0.16, +1.5 band > 6 SE
wide; harness extended: play_round feeds v6 nets obs-v2 with zero
match ctx - the same start-of-match footing the baseline gets):
  arm a +0.828 (SE 0.155) UB +1.083 -> BAND MET, Stage 4 opens
  arm b +0.628 (SE 0.153) UB +0.880
  arm c +0.594 (SE 0.143) UB +0.829
Registered expectation confirmed: from-scratch imitation lands behind
the RL-sharpened baseline (v5-M's scratch-distill had landed AHEAD -
this generation's gap is real). Fresh data alone did not reproduce the
lineage (arm c +0.59 behind). Scale's pre-PPO contribution: none
measurable (a slightly WORSE than b per-deal, within noise).

## 2026-08-13: v6 Stage 4 trial 1 - FAIL (gap real, ladder continues)

Trial 1 (chain-base recipe, minibatch 512, 75k deals ~2h, from
freeze_arm_a): pool telemetry improved through training (avg place
~2.5 -> 2.141, win 27 -> 33.3% vs mixed pool; critic EV 0.91 from the
distilled value head). Registered gate (scripts/run_match_gate.py,
n=3200, 51 min): placement +0.137 vs champion (SE 0.019) - candidate
significantly BEHIND; consistent with the stage-3 screen (+0.83/deal
raw). NOT a structural null (|d| >> 0.02): the ladder continues, trial
2 trains on from trial-1 weights with carried optimizer moments.

Ops incident on the record (cost ~6h + a reboot): an UNGUARDED stdin
pre-flight script around match_eval fork-bombed the machine (Windows
spawn re-imported __main__ recursively; launcher-discipline rules 4
and 10 both recurred). Permanent fix: scripts/run_match_gate.py is the
only sanctioned gate entry. The "v6 gate is 10x slower on CPU" claim
made during the thrash was WRONG - clean timing: 8 matches/24s at 4
workers; full gate ~51 min at 12.

## 2026-08-13: v6 Stage 4 trial 2 - FAIL, no movement; two problems found

Trial 2 (continued from trial-1 weights, optimizer moments carried,
75k deals): gate n=3200 placement **+0.1694 (SE 0.0195)** vs champion
8a89da90 - slightly WORSE than trial 1's +0.1367. Trial-over-trial
movement -0.033 +/- 0.028 = within noise. Neither trial is a
"structural null" by the registered definition (|delta| < 0.02
placement), so the prereg's 3-null halt is NOT met and the ladder
formally continues.

PROCEDURAL DEVIATION (owned): the prereg registers the ladder as
`run_loop` match-mode, which MUTATES the config between trials
(propose -> train -> gate -> promote/rollback). Both trials so far ran
train.py directly on a FIXED config, so trial 2 was the same recipe
continued - which plausibly explains the flat result. Subsequent trials
must go through run_loop (or vary hyperparameters deliberately) for the
ladder to be the registered instrument.

BLOCKER FOUND for Stage 5 (independent of the ladder): the C++ search
path CANNOT LOAD A V6 NET. SearchPlayer.hpp ProbeObsDim probes
{550, 556, 238, 181} only - 882 is absent, and nothing assembles the
obs-v2 vector engine-side. The registered battery is conjunctive
(match gate AND search guard n=4800 K=32), so no v6 candidate can be
promoted until the engine speaks obs v2. This must be built and
verified (A/A determinism + a v5 agreement check) before Stage 5,
regardless of how the ladder ends.

## 2026-08-13: C++ obs-v2 search path BUILT + verified (Stage-5 blocker cleared)

The flat search now speaks obs v2, so a v6 candidate can face the
registered search guard. Changes: ProbeObsDim gained 882 (LAST in the
probe order - every narrower trace would silently accept a wider row);
FillObsRow rebuilt to take the ENV and assemble
[ObserveFor(seat) 550 | WriteCtx 6 | ObserveExtFor(seat) 326], byte-
identical to what selfplay_gen writes into HMR3 records and what
v6_distill concatenates for training; match_aware widened to accept
882; orchestrator._trace_for_search now takes the width from the NET
(a v6 net traced at 556 raises).

TWO LATENT BUGS found by the refactor and fixed:
- FillObsRow memcpy'd 550 floats unconditionally, overrunning a
  narrow (238/181) row into the next row of the batch tensor.
- RawPolicy::ChooseAction wrapped the 550-float Observe() with
  from_blob at obs_dim_, reading PAST THE END for any 556/882 trace.
TreeSearchPlayer now THROWS on 882 rather than silently feeding a v6
net a row with no extension block (the tree search is closed for
strength; the flat player is the only supported obs-v2 path).

VERIFICATION (all pass):
- Static: WriteCtx == hearts_match_env.match_ctx_row term for term;
  record ext == ObserveExtFor(acting seat).
- A/A determinism: same seed twice, byte-identical CSVs (also
  identical across a full rebuild) - no uninitialized reads.
- Sanity: v6 arm -2.33 vs the validated v5 match arm -1.83 on the
  identical seed/deals (n=6, K=8) - the v6 net plays sensibly, which a
  mis-assembled observation could not produce.
- REGRESSION: the v5 arm's result is BYTE-IDENTICAL between the
  pre-change and post-change builds (stash, rebuild, rerun, compare).
- Pipeline: orchestrator.evaluate_candidate_search runs a v6 candidate
  end to end - auto-traced at 882, MATCH-AWARE mode engaged, both arms
  sharded, paired delta returned.

## 2026-08-13: v6 Stage 4 HALTED - three consecutive failed trials

The registered stop rule fired. Three trials ran under `run_loop`
match-mode with `candidate_lineage: true` (the corrected instrument -
trials 1/2 of the earlier entries had run train.py on a fixed config,
the procedural deviation owned above). Each candidate was gated at
n=3200 against ITS OWN v6 lineage baseline, not the v5 champion. All
three FAILED; `scripts/ladder_stop_watch.py` counted the third
consecutive failure and killed the loop while trial 4 was launching.

Positive delta = candidate WORSE (placement, lower is better):

| trial | placement Δ (SE) | final score Δ (SE) | win% A/B | neutral raw (SE, p) | gate |
|---|---|---|---|---|---|
| 1 | +0.296 (0.022) | +7.61 (0.55) | 35.2 / 46.4 | +0.630 (0.170, 1.000) | 5144s |
| 2 | +0.167 (0.022) | +4.37 (0.54) | 40.8 / 48.4 | -0.181 (0.159, 0.127) | 4977s |
| 3 | +0.080 (0.021) | +2.25 (0.52) | 44.1 / 48.1 | +0.187 (0.165, 0.872) | 5188s |

THE FINDING is the trend, not any single verdict: +0.296 -> +0.167 ->
+0.080, monotone convergence toward the baseline that never crosses it.
Trial 3 sits ~3.8 SE behind where trial 1 was ~13.5 SE behind. Because
a failed trial rolls back, each trial is a FRESH run_loop hyperparameter
proposal trained from the same lineage baseline - so the trend reflects
the mutation search finding better recipes, not accumulated training.
The round ran out of trials while still improving.

None of the three is a "structural null" by the registered definition
(|delta| < 0.02 placement) - all are 4-13 SE from zero - so the
prereg's original 3-null halt never applied. What halted the round is
the house PPO stop rule (three consecutive trials without improvement),
applied per the 2026-08-13 clarification. That clarification stops
EARLIER than the registered condition and so cannot inflate a false
positive: **the Stage-4 amendment budget remains untouched.**

TELEMETRY (informs, never gates): the candidate shot FEWER moons than
its baseline in every trial (492/571, 404/543, 528/556 test-seat moons)
while deals/match ran slightly long (A 10.86-10.91 vs B 10.78-10.79) -
consistent with a candidate playing a flatter, less aggressive game
rather than one that is simply weaker everywhere. Moons conceded at
table were mixed (484/478, 557/496, 475/506).

INTEGRITY after the hard kill (launcher-discipline rule 6, verified
BEFORE anything else touched the weights):
- `Hall_of_Fame/hearts_model_milestone_1785322724.pth` md5 **8a89da90** - intact
- `hearts_web_model.pth` (served) md5 **8a89da90** - identical to champion
- `hearts_ai_search_match.pt` 3a2abd36 / `hearts_equity.pt` efdfee07 -
  unchanged since 2026-08-07
- `hearts_model_final.pth` a9653255 holds an UNPROMOTED candidate -
  must never be served (CLAUDE.md MODEL_PATH rule)
- no orphaned processes; GPU released to 2.2GB

CONSEQUENCE: Stage 5 is NOT entered (it requires a gate pass). The
champion is unchanged and v6 has not beaten v5 through the PPO ladder.
What the halt does NOT license is an improvised continuation: the trend
argues the recipe search was still working, but resuming requires a
registered decision (a new round, or the one Stage-4 amendment), not
just relaunching the loop.

## 2026-08-15: v6 DATA-SCALING PROBE — verdict DATA-BOUND (registered reopening test)

Prereg docs/v6_data_scaling_prereg.md (signed 21:44 PDT; instruments
md5-frozen before any counted run). Frozen arm-a recipe (v6_distill.py
UNCHANGED: lr 1e-4, 3 ep, batch 512) on nested, generation-stratified
subsets of the frozen 1.52M bank, 2 seeds per size, all six nets scored
on the SAME full-bank by-match holdout (80,220 rec / 123 matches).

MEASURED durations (RTX 4090, gentle profile, desktop co-resident):
S1 456,150 rec = 288 s; S2 970,749 rec = 608 s; S3 1,444,601 rec =
906 s per 3-epoch training (95/202/301 s per epoch — linear in records,
5.0 min/epoch on the full bank vs the 08-13 5.1). Six trainings 60 min
wall (derived quote was 62). Fixed-holdout eval ~20 s/net. Screens
n=2500 ~3.9 min each incl. worker spin-up.

Fixed-holdout results (seed 20260812 / 20260813):
  S1  CE 0.9333/0.9427  match 60.57/60.06%  entropy 0.926/0.927 (OUT of band)
  S2  CE 0.8586/0.8748  match 63.49/62.84%  entropy 0.864/0.877
  S3  CE 0.8413/0.8488  match 64.17/63.86%  entropy 0.832/0.854 (IN band)
S3 seed 20260812 REPRODUCED the frozen Stage-3 arm-a run bit-for-bit
(0.8413 / 64.17% / 0.832) — recipe deterministic on this box.

Registered paired by-match inference (seed-averaged, 123 clusters):
  control  S1->S2  dCE -0.0703 [-0.0732,-0.0674]  dMatch +2.75pp [+2.51,+3.00]  q<1e-4
  decision S2->S3  dCE -0.0217 [-0.0240,-0.0194]  dMatch +0.89pp [+0.70,+1.09]  q<1e-4
  (Wilcoxon exact agrees everywhere.)  VERDICT: **DATA-BOUND.**
Slope is real but DIMINISHING: per doubling, CE -0.065 (control) ->
-0.038 (decision); match +2.5pp -> +1.6pp. Seed spread 0.008-0.016 CE.

Screens (registered ESTIMATION ONLY, one net per size, n=2500 vs
8a89da90, each its own seed => UNPAIRED across sizes, SE ~0.14-0.15
each, cross-size SE ~0.2):
  S1 +1.888 (0.154)   S2 +0.439 (0.140)   S3 +0.766 (0.149)
  [S3 = the frozen arm-a net; Stage-3 screen of the same net +0.828
  (0.155) — re-measurement consistent.]
The 0.5->1.0M step is huge in strength (-1.45/deal, ~7 SE). The
1.0->1.5M step is NOT resolved by this instrument (+0.33 +/- 0.20, wrong
sign, ~1.6 SE) — the strength axis does not confirm the decision-step
imitation gain at n=2500 single-seed. TENSION ON THE RECORD: imitation
says data-bound; strength at 1.0->1.5M is flat-or-noise. Consequence:
the Stage-2b go/no-go band must be set in STRENGTH units with PAIRED
(CRN) screens and >1 seed, not in imitation units.

CONSEQUENCE (per prereg §7): the CONDITIONAL scale-without-data closure
is reopened by its registered route; Stage-2b becomes PROPOSABLE (own
prereg + approval, ~60h local). Nothing else is licensed: no Path C, no
promotion claim, no PPO statement. Nets in v6_probe/ are probe artifacts
— never serve, never gate. Champion 8a89da90 untouched; hearts_model_
final.pth not written. Verdict JSON: equity_data/verdicts/v6_data_probe.json.

## 2026-08-16: v6 DATA PROBE ADDENDUM A — paired strength: Stage-2b NOT PROPOSED, v6 SHELVED

Registered before running (docs/v6_data_scaling_prereg.md §10). Same
six probe nets; neutral_raw_eval.py with BOTH arms probe nets, n=5000
paired deals per pairing (SE 0.10-0.11), 4 pairings = 45 min wall
(03:01-03:46; ~11 min each incl. spin-up — MEASURED n=5000 row).
  S3-S2: +0.198 (0.105) / -0.158 (0.104)  pooled +0.018 (0.074) UB95 +0.163
  S2-S1: -1.049 (0.112) / -1.345 (0.111)  pooled -1.198 (0.079) UB95 -1.044
Control sane; decision UB95 > -0.10 => registered band 3: STAGE-2B NOT
PROPOSED; v6 SHELVED PENDING A NEW SIGNAL SOURCE. Path C on the existing
distill stays the only v6 continuation (own prereg). Seed finding: same-
size nets differ by 0.30-0.36/deal in strength (~2 SE) — training-seed
variance exceeds the +50%-data effect; >=2 seeds per arm for any future
distill strength comparison. Register updated. Champion untouched.

## 2026-08-16 BACK-FILL: measured rows for 2026-08-01 → 08-11 (recovered from logs/)

Documentation audit 2026-08-16 found the ledger had NO rows between the
08-01 wedge correction and the 08-12 v6 Stage-2 entry, although four
programs ran in that window (expert-iter v2, exploiter league r1-r3 +
side probe, Phase 2 visit-count). Rule 7 quotes durations from this
file only, so the rows are reconstructed here from the driver logs'
own timestamps (PHASEA_/PHASEB_/DEFGATE_/R2GEN_/R3TRIAL/P2* markers).
Where a start had to be inferred from the preceding job's end, it says
so. All local RTX 4090 / 7800X3D, v5 nets unless stated. Order of
programs = chronological.

### Expert-iter v2 (2026-08-01 → 08-05; docs/expert_iter_v2_results.md)
- Match-aware generation, mixed chunk types (nat/knife/mid/leader/asym/
  trail/early), 08-01 → 08-04 02:32 ALL_RESERVES_MET. Per-chunk seconds
  are logged (e.g. nat_70: 40 matches 4,435 s = 111 s/match; knife_148:
  60 matches from totals 90/88/86/84 = 1,630 s = 27 s/match — short
  matches). Bank build 02:45, freeze 03:13.
- v2mix pipeline (6 compositions x 2 reps, 08-04 07:31 → 08-05 02:33 =
  19.0 h): confidence-filtered distill ~7 min per candidate; TWO match
  gates n=3200 (blocks b5/b6) per candidate at **32-34 min each** (v5
  vs v5, 12 workers). Analysis 02:33.

### Exploiter league round 1 (2026-08-05 → 08-08; docs/exploiter_league_prereg.md; results in r2_results.md §r1)
- Phase A (search-shooter vs 3x baseline, 402 matches per combo, 3
  shards x 134, K=64 flat, pass search): agg_base **4 h 38 m**
  (13:56→18:34), sel_base **11 h 20 m** (→05:54), sel_v4 **9 h 53 m**
  (→15:47). Total 25.8 h. (An 11:30 start at 6 shards was aborted and
  restarted at 3 shards at 13:56.)
- Phase B generation (510 matches per mode = 3 x 170): agg **8 h 58 m**
  (16:38→01:36), sel **11 h 20 m** (→12:56). Clone distills + quality
  verify 12:59-13:37 (~40 min).
- Phase C trials (match-mode PPO from 8a89da90, shooter shares
  0.15/0.15, Adam moments carried): t1 ≈1.6 h (start inferred ~13:40,
  end 15:15), t2 **1 h 29 m** (17:57→19:26), t3 **1 h 34 m**
  (20:58→22:32).
- Defense gate (64 CRN-paired seed-matches, 2 shards x 32, BelowNormal,
  frozen SEL search-shooter probe): t1 base arm **78.8 min** + cand arm
  **79.9 min**; t2 cand arm 92 min; t3 cand arm 106 min (concurrent
  desktop load; the base arm is CRN-reusable and was NOT rerun for
  t2/t3).
- Gates 2+3 on t3 (00:18→04:33 = 4 h 15 m): NI match gate n=3200
  **2,066 s = 34.4 min**; search guard n=4800 K=32 (8 concurrent runs)
  **≈3.6 h**.

### Exploiter league round 2 (2026-08-08 → 08-09; docs/exploiter_league_r2_results.md)
- Search-defender smoke 08:56-12:29. A2 corpus generation (3 search
  defenders vs clone attacker, 180 matches per mode, 3 x 60): agg
  **6 h 19 m** (13:26→19:45); sel 1 h 52 m + 4 h 47 m = **6 h 39 m**
  (paused 21:37, resumed 02:53 — the pause/resume machinery's first
  production use).
- b2 supervised grid (12 checkpoints on ~60k moon-alive decisions):
  minutes per checkpoint (09:19-09:56 for grid + exploration + final).
- 16-seed defense pre-probes: **20-23 min per arm** (5 arms 09:58→11:45).
- Defense gates (cand arm only, base arm reused): kl8ep1 **86.6 min**,
  kl4ep1 **83.4 min**.

### Exploiter league round 3 + Phase-2 side probe (2026-08-09 → 08-10)
- Anchored-PPO trials at HEADROOM 25%: c0 (λ=1.0) **2 h 08 m**
  (15:14→17:22); c0b (λ=0.4) **2 h 17 m** (17:24→19:41); probe005
  (λ=0.05) **2 h 13 m** (22:37→00:51). Each includes candidate archive
  + baseline restore + drift measurement (seconds).
- probe005 search guard n=4800 K=32: **3 h 39 m** (02:24→06:03).
- probe005 defense gate cand arm ≈80 min (end 08:53, start inferred).

### Phase 2 visit-count distillation (2026-08-09 → 08-11; docs/phase2_visitcount_results.md)
- Pace probe (tree teacher) budgets 200/400/800: **14.4 / 23.3 / 54.0
  min**. Stage B validity: 37 min (06:05→06:43). Stage B-2 chain incl.
  rebuild: 1 h 47 m (07:06→08:53).
- Stage C strength screen (n=800 deals, 2 shards, tree vs flat):
  **2 h 41 m** (09:12→11:53). Generation 2 x 170 matches at budget 200,
  BelowNormal: **22 h 22 m** (11:53 08-10 → 10:14 08-11).
- Stage D distills: ~8 min for the six candidates (10:14→10:22).
- Stage E gates per candidate (match n=3200 + guard n=4800): ep3
  **4 h 15 m** (10:29→14:44), ep2 **4 h 29 m** (→19:13).

### Reconciliation notes written 2026-08-16
- **Match gate n=3200 durations by net generation (same 12-worker
  convention):** v5 vs v5 **32-34 min** (this window, 3 measurements);
  v6 vs v5 champion **51 min** (08-13); v6 vs v6 **83-86 min** (08-13
  ladder). The 07-28 "~40 min" was the first evolved-guard-era figure.
- **Search guard n=4800 K=32:** measured **3.6-3.7 h** in this window
  (two runs) vs the 2.7 h projected on 07-28 from the n=2400 82-min row.
  Quote 3.6 h from here on.
- **Defense gate (64 CRN seed-matches, search-speed SEL probe):**
  **80-107 min per arm** depending on co-resident load; base arm
  reusable. Pre-probe (16 seeds): ~21 min per arm.
- No results doc existed for round 1; docs/exploiter_league_r1_results.md
  now carries the round-1 record with verdict-file pointers.

## 2026-08-16: LEAGUE R4 STAGE 0 — base arm reproduces; fast defense probe VALIDATED; r1-t3 replicates

Prereg docs/exploiter_league_r4_prereg.md (signed 08-16, defense gate
n=320). MEASURED:
- Base-arm reproduction (SEL search-shooter shard 0, 32 matches, seed
  720260806, single shard, K=64 flat, GPU): 33.4 min (11:07:49->11:41:11)
  = 63 s/match single-shard. Output BYTE-IDENTICAL to r1's base_0.csv
  under probe trace 3a2abd36 + the 08-13-rebuilt SearchEval.
- Fast defense probe (defense_probe_fast.py, 3 raw defenders vs the
  shooter_sel_v1 clone, CRN-paired vs baseline defenders): 8 arms x
  1,000 matches = 8,000 raw matches in 73.2 min at 12 workers,
  BelowNormal (~9 min per arm-thousand; the desktop stayed usable).
Results (delta moons conceded/match, negative = better; SE ~0.04):
  A/A 0.000 | c0 -0.011 | c0b +0.065 | probe005 -0.060 (p=.066) |
  r1t1 -0.074 (p=.030) | r1t2 -0.089 (p=.015) | r1t3 -0.235 (p<1e-4)
=> §3.3 VALIDATION PASS; ordering reproduces the search-SEL gate's;
r1-t3's -0.250 (n=64) REPLICATES at -0.235 +/- 0.041 under a different
attacker. Clone concedes 1.48 moons/match to the champion vs the search
shooter's 2.48. Nothing else launched; trials await go.

## 2026-08-16: LEAGUE R4 base-arm EXTENSION COMPLETE (n=64 -> 320)

8 shards x 32 CRN seed-matches (seeds 724260806..731260806), baseline
defenders vs the frozen SEL search-shooter, K=64 flat, pass search,
FULL SPEED: two waves of 4 concurrent shards, Normal priority, machine
otherwise idle. MEASURED: wave 1 2 h 51 m (13:40->16:31), wave 2
2 h 52 m (16:31->19:23) => **5 h 42 m for 256 matches = 80 s/match
aggregate at 4-wide** (single-shard was 63 s/match; 4-wide buys ~3.2x
throughput, GPU pinned 100% / 11 GB - wider would not help). All 8
shards rc=0, 32 matches each, 1,763 deals.
Base rate on the extension: 2.559 moons/match (var 0.475) vs r1's 64:
2.484 (var 0.444) - consistent. Combined base arm n=320: 2.544/match.
Files: equity_data/exploiter_r4/base_ext/base_{4..11}.csv (+ .tricks).
Quote for a CANDIDATE arm at n=320 (10 shards): ~7.1 h at 4-wide
(80 s/match), not the 6.6 h projected from 2-wide r1 timing.

## 2026-08-17: LEAGUE R4 round-1 factorial COMPLETE — no gate-eligible cell; reserve arm is the program

Four anchored-PPO cells from 8a89da90 (r3 recipe, fresh Adam,
config_r4_base.json + {anchor_kl_coef, shooter_share}), each followed by
drift measurement + fast defense probe on mid and end snapshots
(docs/exploiter_league_r4_results.md). MEASURED:
- Cell A (lambda 0.05, share 0.15) at HEADROOM 0.25: train 2 h 18 m
  (02:48->05:06), probes+drift 31 min, cell 2 h 49 m.
- Cells D/B/C at FULL SPEED (HEADROOM 0): train 1 h 34 m / 1 h 33 m /
  1 h 35 m; cells 2 h 01 m / 2 h 01 m / 2 h 03 m incl. ~27-min probes.
  Full speed = ~1.5x the headroom pace. Whole chain 02:48->11:42.
Fast-probe END deltas (moons conceded/match, n=1000 CRN, SE ~0.038):
  A -0.082 (p=.014, UB -0.009) | D +0.006 | B -0.019 | C +0.033
Drift: A 7.9%, B 7.9%, D 10.4%, C 10.2% (predictions 7.8% / ~9.5%).
None meets the registered eligibility (UB<0 AND point <= -0.10);
D,B,C = three consecutive non-eligible => round-1 closed. Cell A +
probe005 = two seeds at ~-0.07 (2.6 SE pooled): a real but sub-bar
signal at lambda=0.05. Looser anchor and denser threats both WORSE.
Nothing gated; champion md5 8a89da90 verified after cell C.

## 2026-08-17: LEAGUE R4 Addendum R trial R1 (block credit b=2.0) — NOT eligible; R2 = b=4.0 by rule

Anchored PPO from 8a89da90, r4 cell-A recipe + block_credit_b 2.0,
HEADROOM 0.25: train 2 h 04 m (13:47->15:52), drift+probes 30 min,
COMPLETE 16:22. Drift 7.7% (agreement 0.923). Fast probe n=1000:
END -0.063 (SE 0.037, UB95 +0.010, p=.046); MID -0.114 (SE 0.036,
UB -0.043, p=.0009). Registered END reading fails both eligibility
criteria. Block credits 70,565 (+11,619 reward, ~+0.165 each);
defender placement vs probe field -0.02/-0.03 (no Goodhart sign).
Champion restored/verified 8a89da90. Per §9.3 -> R2 at b=4.0.

## 2026-08-17: LEAGUE R4 Addendum R trial R2 (b=4.0) — NOT eligible; ROUND 4 CONCLUDED, no promotion

HEADROOM 0.25: train 2 h 02 m (16:22->18:25), drift+probes 34 min,
COMPLETE 18:59. Drift 8.2%. Fast probe END -0.056 (SE 0.037, UB95
+0.017, p=.066); MID +0.010. Credits 71,362 (+23,655 reward, ~+0.33
each); defender placement -0.018 (no Goodhart sign at b=4).
Both reserve trials non-eligible => §9.6 closure: block-credit reward
shaping at b<=4 does not move defense inside the anchor. Across the
lambda=0.05/share=0.15 family, four seeds at b=0/0/2/4 read
-0.060/-0.082/-0.063/-0.056: a stable ~-0.065 +/- 0.019 in-band ceiling
(real, sub-bar). Champion 8a89da90 verified. Round-4 total machine time:
Stage 0 ~2.5 h + extension 5.7 h + 6 trials ~14 h = ~22 h, $0.

## 2026-08-17/18: LEAGUE R5 STAGE 0 — instruments frozen, T0 no-halt, T1 positive

Prereg docs/exploiter_league_r5_prereg.md (signed 08-17). MEASURED:
- T0 information audit (audit_t0_info.py, 80,220 holdout records, linear
  probes on champion activations, GPU L-BFGS): ~6 min. Per-seat points
  R^2 0.62 (raw 0.39); moon-alive AUC 0.975 given points taken (raw
  0.911); QS holder acc 0.93; per-card taken-by from the card token 0.784
  = raw channels 0.787 (majority 0.61) -> capture attribution ABSENT.
  Registered halt NOT triggered.
- T1: fast probe on arm b (obs v2) / arm c (obs v1), 2 nets + base x
  1,000 = 3,000 matches: 30 min at 12 workers; paired strength n=5000
  4.7 min. arm b -0.117 (SE 0.041) vs champion, arm c +0.126 (0.041);
  b-c strength -0.246/deal (SE 0.103).
- validate_v5ext 4,096 states ~1 min; drift_screen_v6holdout ~1 min/net;
  ext-learner trainer smoke (SMOKE_TEST) ~2 min.
Trials E1,C1,E2,C2 launched 00:21 at HEADROOM 0.25.

## 2026-08-18: LEAGUE R5 trials COMPLETE — mechanism NULL (adapters never became a pathway); no promotion

Four cells at HEADROOM 0.25 (E1 00:21->03:19, C1 ->06:08, E2 ->08:56,
C2 ->11:44): train 2 h 17-19 m each; drift x2 + fast probes + paired
strength ~41 min per cell (fast probe 3 arms ~28 min, neutral_raw
n=5000 ~4.7 min, drift instruments ~2 min). MEASURED per-cell 2 h 48-
59 m; four cells 11 h 23 m. Fast probe END: E1 -0.015, C1 -0.029, E2
-0.015, C2 -0.046 (SE 0.038 each); pooled E-C +0.022 (SE 0.038) => NULL;
no cell eligible. Adapters after 250k deals: mean |w| ~0.006 (~30x below
the trunk's card_proj) - the zero-init pathway did not grow into use at
lr 9e-6 under the anchor. Paired strength vs champion: E1 -0.003, E2
-0.001, C2 -0.001, C1 -0.198 (SE ~0.08). Champion 8a89da90 verified.
Round-5 total: Stage 0 ~1 h + trials 11.4 h, $0.

## 2026-08-18: LEAGUE R5 Addendum W — warm-start HALTED at acceptance (no trials)

warmstart_v5ext.py: adapters-only supervised imitation on the 1.44M
training records, trunk frozen, 1 epoch = 108 s (batch 512, GPU);
acceptance (teacher-match on the 80k holdout + shared drift) ~2 min.
lr 1e-3: tm 54.27% (champ 53.47%) PASS, threat-dead drift 34.1% FAIL.
Tune lr 1e-4: tm 49.9% FAIL, drift 36.0% FAIL. HALT per the registered
one-tune rule. Both rejected checkpoints kept untracked. Champion
untouched (only read).

## 2026-08-18: HYBRID "defense specialist" probe (docs/hybrid_specialist_probe.md)

HeartsHybrid (champion + arm b switching on moon-alive threat states).
MEASURED: fast probe 2 hybrids + base x 1,000 = 3,000 matches in 45.7 min
at 12 workers (the hybrid runs TWO nets per decision, one at 882 - ~1.6x
a plain-net probe); neutral_raw n=5000 vs champion 6.6 min each.
threat gate: defense -0.107 (SE 0.036), strength +0.104 (SE 0.055);
any_alive gate: -0.147 (0.041), +0.246 (0.093). Defense lives in the
threat-state play (19% of decisions).

## 2026-08-18: OPS INCIDENT — desktop starved by an 8-arm hybrid probe (hard power-off); NI gate result preserved

MEASURED before the incident: match NI n=3200, threat hybrid vs champion,
12 workers, 2,675 s = 44.6 min (two nets per decision on 19% of steps):
placement +0.001 (SE 0.010), UB95 +0.017 -> NI PASS; win 52.4/52.6.
Then the gate LADDER (7 hybrids + base = 8 arms/decision, ~240 ms per
decision per worker, 12 workers, NORMAL priority - headroom's BelowNormal
was a no-op with HEARTS_HEADROOM unset) started 15:05; by ~15:20 shells
timed out, the display would not wake, user hard-powered-off at 16:19
(EventLog 6008/41; no 2004 resource-exhaustion, no bugcheck). CPU
starvation, not a wedge and not memory (~1.1 GB/worker measured).
FIXES: headroom.apply_process_priority now lowers priority for every pool
(HEARTS_NO_LOWPRI=1 to opt out); hybrid checkpoints load in-memory;
RULE: daytime CPU pools <= 8 workers, never stack > 3 arms per probe.
Champion/working/served weights verified 8a89da90 after reboot.

## 2026-08-18: HYBRID gate ladder (gentle rerun) — moon-head router wins: -0.247 defense at ~0 cost

8 workers BelowNormal (post-incident rules): fast probe 4 arms x 1000 =
4,000 hybrid matches 88 min (A), 4 arms 135 min (B, moon-head gates run
arm b's aux forward), 2 arms 40 min (C); neutral_raw n=5000 8-13 min each
at 8 workers. Total 5 h 40 m; desktop usable throughout. Results in
docs/hybrid_specialist_probe.md: moonhead:0.1 -0.247 (SE 0.032) defense /
+0.014 (0.031) strength; threat:3/6/10 -0.17/-0.17/-0.15 at ~-0.01;
uncertain +0.03 / +0.05 (dead).

## 2026-08-19: LEAGUE R6 gate 1 — NI PASS for the tau=0.1 ensemble; defense gate candidate arm launched 00:59

NI n=3200 (8 workers BelowNormal, ensemble = 2 nets on gated ~10% of
decisions): 5,121 s = 85 min. placement +0.003 (SE 0.006), UB95 +0.013
-> PASS; win 51.4/51.4; score +0.12; moons conceded at table 509 vs 521.
Defense gate candidate arm (n=320, shards 0,1,4..11, 4-wide) started
00:59 (equity_data/exploiter_r4/r6_gate/cand_*.csv; analyzer -> r6_defense_gate_n320.json).

## 2026-08-19: LEAGUE R6 defense gate — HALT (null vs the search shooter); attacker-transfer failure on record

Candidate arm n=320 at 4-wide (waves of 4,4,2): 00:59 -> 07:48 = 6 h 49 m
(shards 0/1/4/5 2h49m; 6/7/8/9 2h40m; 10/11 1h20m at 2-wide) — 77 s/match
aggregate 4-wide, matching the 08-16 extension rate. Analyzer: base 2.544
vs cand 2.525, delta -0.019 (SE 0.043), p=0.33 -> HALT. vs the fast probe's
-0.247 (SE 0.032) for the same ensemble: the clone attacker overstates
defense for this clone-distilled specialist ~10x. Guard not run. r6 total
~9 h, $0. Champion untouched.

## 2026-08-19 evening: r6 DEFENSE GATE RETRACTED — instrument artifact (882 defenders zero-filled); engine fixed

Byte-identical candidate CSVs across THREE different ensemble runs
(r6 n=320 + two n=64 checks; md5 b1e7b7d4) exposed it: in shooter mode
an 882 defender trace fell through to RawPolicy's zero-fill (dims
550..882 = 0), the router never fired, every "candidate" arm replayed
the champion at zero match ctx. r6 gate verdict VOID (it was an A/A);
the "attacker-transfer failure ~10x" reading WITHDRAWN; register entry
corrected. NI (+0.003, Python harness) and all clone-probe numbers
stand. FIX: MatchRawPolicy(obs_dim) assembles [550 obs | ctx 6 | ext
326] for 882 defenders (mirrors FillObsRow); search_eval accepts 882
for mdef. Rebuilt; 556 regression byte-identical vs r1 base rows.
MEASURED same day: steps 1-2 of the specialist work (all Python, valid):
AGG diagnostic - every candidate defends the AGG clone hugely (ensemble
-0.440, arm b -0.468, r1-t3 -0.473 vs base 2.018); specialist ladder
SEL/AGG/strength - champion+arm a (own moon head, tau .1): -0.360/-0.647
at -0.010 strength; champion+r1-t3 (arm-b head): -0.252/-0.364 at +0.005.
NEW RULE (register): C++ ensemble measurements need a gate-fires check
(candidate CSV must DIFFER from the default's on same seeds) first.

## 2026-08-19 night: FIXED-ENGINE search-SEL n=64 — the arm-a ensemble DEFENDS THE SEARCH ATTACKER

Reruns on the repaired engine (gate-fires check passed; base arm r1
shards 0,1 reused; 2-wide, ~92 min/candidate):
- champion + r1-t3 (arm-b moon head, tau .1): -0.078 (SE 0.125), p=.27
  - inconclusive at n=64.
- champion + ARM A (own moon head, tau .1): baseline 2.484 -> 1.969,
  **delta -0.516 (SE 0.151), p=0.0003, PASS at n=64** - the largest
  defense effect ever measured against the search shooter (r1-t3 whole
  net: -0.250). Defender placement vs the probe field 2.969 -> 2.589
  (better). Verdicts: equity_data/verdicts/searchsel_{r1t3mb,arma}.json.
Next decision (user): full battery for the arm-a ensemble under a fresh
registration (defense gate n=320 ~7h + guard ~4.5h; NI already
measured for the sibling tau=.1 ensemble but must be re-run for THIS
candidate).

## 2026-08-20: r7 battery — NI PASS, defense gate PASS (primary -0.742); guard wedged at 4-wide, relaunched 2-wide

NI n=3200 (6 headroom workers; moonhead router runs the 19M aux EVERY
decision): 3 h 03 m — dplace -0.011 (SE 0.007), UB95 +0.0004 PASS (point
estimate favors the ensemble). Defense gate candidate arm (10 shards,
waves 4/4/2): 02:27->10:11 = 7 h 44 m. POOLED n=320: 2.544 -> 1.847,
delta -0.697 (SE 0.065), t=-10.8. REGISTERED PRIMARY (fresh shards 4..11,
n=256): 2.559 -> 1.816, **delta -0.742 (SE 0.071), t=-10.4, PASS** —
29% fewer concessions to the search shooter; defender placement vs probe
field 2.951 -> 2.576 (better). Chain's inline primary script crashed
(numpy int into statistics.stdev) AFTER the analyzer's PASS print — halt-
default worked; primary recomputed with numpy, identical data.
GUARD: ensemble-as-search-policy measured ~21.4 s/deal (~4x champion —
champion + 19M per rollout row): true guard cost ~14 h, not the prereg's
~4.5 h note. At 4 concurrent shards the run WEDGED (base shards zero-CPU
2 h, cand shards stopped writing; GPU 7.1 GB — NOT the 07-30 VRAM
ratchet; 4-process contention class). Killed cleanly; relaunched
shards=2 (1 pair, 2 concurrent — an evaluate_candidate_search parameter,
not an instrument change; same seeds/pairing). ETA ~14 h.

## 2026-08-20 CORRECTION: the guard was NOT wedged — killed in error at deal 600/2400

The 13:30 "wedge" call was WRONG: logs/r7_guard.log shows steady progress
(deal 600/2400, running mean diff -0.363, elapsed 11,212 s = 18.7
s/paired-deal) at the moment of the kill. The per-shard CSV mtimes and
per-process CPU snapshots I diagnosed from do not track this run mode's
progress; the guard log's own deal lines do. ~3.1 h lost; relaunched
fresh with the ORIGINAL instrument settings (shards=4). RULE: judge a
guard/gate run by ITS OWN progress log, never by shard-file mtimes; a
kill decision needs the run's primary log read FIRST (rule 10's watchdog
patterns already say this - re-learned). Measured guard pace for the
ensemble candidate: 18.7 s/paired-deal => n=4800 ~ 12.5 h.

## 2026-08-20 evening: r7 AMENDMENT 1 (registered) — gate 3 re-pointed at the actual deployment; telemetry chunked+resumable

User decision "(A)": ensemble promotes RAW-ONLY; the served search
substrate keeps the champion traces (3a2abd36 / efdfee07, verified) and
traces are NOT re-exported — so gate 3's subject (served searched play)
is unchanged by construction and is satisfied by substrate verification.
The amendment budget is SPENT; recorded before any valid gate-3 data
existed (two 4-wide runs wedged, one staged run stopped by user — all
VOID, no statistic computed). Ensemble-as-rollout number demoted to
TELEMETRY at n=2400 via the new chunked resumable driver
(scripts/run_r7_guard_telemetry.py: 12x200-deal chunks/arm, fixed chunk
seeds 745260820+stride, <=2 concurrent, resume-by-complete-chunk;
launched 20:0x). Root-cause note for the wedges: candidate trace =
champion + arm a with the specialist forwarded TWICE per row (gate +
action) ~ 6x a normal candidate's compute; 4-way concurrency with the
60MB trace stalls progressively (07-25 class; base shards froze at
~deal 630 both runs). Fix queued for round 8/serving: single-aux-forward
hybrid (bit-equal null contract) + a cheap pre-gate; guard-class runs at
<=2 concurrent for big traces.

## 2026-08-21: r7 COMPLETE — ALL GATES PASS (Amendment 1); telemetry neutral; candidate eligible for promotion

Telemetry finished via resume (16 complete chunks detected, 8 re-run;
the two FAIL lines in the log are last night's user-requested stop —
chunks re-ran cleanly): base chunks 8.8-8.9 min each, cand chunks ~50
min each at 2-wide headroom; assembled n=2400: +0.016 (SE 0.167), UB95
+0.291 — ensemble-as-rollout ~neutral for searched play (does not serve;
informs only). Battery: NI PASS (-0.011 +/- 0.007), defense primary
PASS (-0.742 +/- 0.071), guard satisfied by substrate verification
(Amendment 1). Results doc: docs/exploiter_league_r7_results.md.
PROMOTION DECISION -> user.

## 2026-08-21: 6TH MATCH-ERA PROMOTION — the gated ensemble (first ensemble promotion)

hybrid_champ_arma_moonhead_0p1.pth (8d7816d1) -> Hall_of_Fame/
hearts_model_milestone_1787333162.pth, md5-verified. Battery: NI PASS
(-0.011 +/- 0.007), SEL defense primary n=256 PASS (-0.742 +/- 0.071,
29% fewer concessions), guard = substrate verification (Amendment 1;
3a2abd36/efdfee07 re-verified at promotion). Rollout telemetry +0.016
+/- 0.167 (neutral). NOT scp'd to the VPS: serving awaits the obs-v2
raw path in perilune-site; hearts_web_model.pth stays 8a89da90 until
then (verified). Round-8 bar = the ensemble's SEL numbers (1.816 vs
base-arm 2.544/320). No optimizer state exists (composition of frozen
nets); traces NOT re-exported (Amendment 1).
