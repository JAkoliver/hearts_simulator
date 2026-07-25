# Match-aware search — design (2026-07-24, revised per stats/design review)

Goal: make the deployed search player optimize MATCH equity instead of
deal points. Success gate (review point 7): **match-aware search beats
match-blind search significantly** (paired --match-pair, n>=200, K=64,
McNemar p<0.05 on wins, placement trend consistent). No additive
prediction is part of the gate: search's +12.5-pt win edge and raw's
score-awareness are expected to OVERLAP substantially (a stronger
per-deal player reaches fewer desperate positions).

## Components

### 1. Match-equity model (small supervised net)
- **Output (review 6): a 4-vector P(place 1..4) for the self seat**, not
  a scalar. Runtime objective is then a flag: `win` (maximize P1) or
  `place` (minimize E[place]) - switchable without retraining. FLAG: the
  headline metric (match wins) and the match gate's promoter (mean
  placement) imply DIFFERENT endgame risk appetites; default objective
  for v1 = `win`, revisit against gate design later. Tie outcomes train
  as soft targets (rank 1.5 -> half mass on P1 and P2).
- Input: rotated totals/100 (self, left, across, right), deals/20,
  leader-distance/100, **plus pass direction of the upcoming deal
  (one-hot 4: deal index mod 4 drives the rotation and therefore
  future play - third review point 4)**. Verified: HeartsEnv advances
  the rotation on every Reset(), so seeded states align direction via
  deals_played mod 4 extra resets.
- **Training data (review 5): seeded, not natural.** Matches start from
  SAMPLED score vectors: 50% uniform on [0,99]^4 (max<100), 30% one seat
  in 85-99 (threshold tails), 10% two seats >=85 (near-threshold ties /
  moon-decisive territory), 10% natural-trajectory states (calibration
  anchor). deals covariate = round(sum/26) + small noise. Play out with
  the current raw nets; record (seeded state -> final placements).
- **Policy dependence acknowledged**: equity(state) depends on how
  players play from that state. Plan 1-2 REGENERATION rounds after the
  ecosystem goes match-aware; version the model (equity_v1, v2...).

### 2. C++ integration (review 4 - the top silent-null risk)
- SearchPlayer carries the live match totals + deals played.
- **Every net evaluation inside search - root candidate scoring AND all
  four rollout seats - receives 556-dim obs with the match context
  rotated to the ACTING seat.** Rollout opponents are therefore
  match-aware (to raw-net quality) and can represent equity-rational
  moves like feeding the leader; scoring any seat's decision with
  another seat's perspective is the failure mode this rule exists to
  prevent.
- Leaf scoring: rollout completes the deal -> totals' = totals + deal
  scores -> equity net evaluated PER SEAT on totals' -> root's action
  value = ROOT seat's objective (P_win or E[place]). Terminal matches
  (max>=100) bypass the net: exact placements.
- Equity net traced; ~microseconds per call, negligible vs rollouts.

### 3. SNR probe BEFORE full integration (review 8)
Equity compresses scores nonlinearly; per-action differentials may be
noise-dominated at K=64 even if the design is right. Instrumented probe
at stratified decision states (early-match / one-seat-near-100 /
self-trailing-badly): per-action equity means over K=64 with COMMON
determinizations across actions; report median |action gap| vs SE of
the K-mean, side by side with the same ratio for deal-point scoring
(the known-working reference). Go/no-go before C++ work proceeds.
Mitigations if poor: adaptive K at high-leverage states, common-random
variance pairing, or lambda-blend of deal-points + equity.

### 4. Behavioral diagnostics BEFORE any win-rate run (review 9)
Scenario suite with constructed score states; match-aware vs match-blind
search at identical states:
- trailing badly (self ~80, leader ~95): moon-attempt rate must RISE;
- leader at 97, self safe mid: point-dumping onto the leader must
  APPEAR when available.
Absent behaviors = wrong plumbing regardless of aggregate results; the
win-rate validation runs only after these move in the right direction.

### 5. Validation
--match-pair: match-aware search vs match-blind search (same net, same
K, same anchors, paired seeds), n>=200. Gate per the top of this doc.

## Validation pre-registration (added 2026-07-24, second review)

- **Design: anchor-field, identical to the bridge measurement** - each
  arm plays the test seat vs three v3-m7 anchors on paired seeds. NOT
  2v2 (which changes game dynamics and breaks comparability).
- **N = 400 with one interim look at 200** (Haybittle-Peto: interim
  alpha 0.005, final ~0.048). POWER CAVEAT (third review): N=400's MDE
  of 3.9-5.6 win-pts holds only if CRN takes discordance to q<=0.16 -
  at q=0.28 (inside the range the n=200 arithmetic itself treated as
  plausible) MDE is 7.4 pts, ABOVE the 6.25-pt half-bridge benchmark.
  The bridge run realized q=0.385 WITH common deals, so CRN delivering
  q~0.10 is currently an assumption, not a measurement. Therefore:
  **pre-registered BLINDED sample-size re-estimation** - at the interim
  look, read realized q WITHOUT unblinding the split direction,
  recompute required N for MDE<=6.25 at 80% power, extend accordingly.
  q is a nuisance parameter; blinded re-estimation does not inflate
  type I error and needs no alpha adjustment. **Max funded N = 800
  pairs** (the rule terminates there; ~5 h local single-process or
  ~1.2 h sharded at measured rates).
- **Common random numbers beyond deals**: both arms share the search
  stack, so pair identical determinization sets and rollout seeds
  wherever the action sets agree. Report the realized paired placement
  SD vs the bridge run's 1.35 and the realized discordance rate q.
- **Primary endpoint: match-win McNemar (one-sided).** Pre-registered
  as the continuation of the metric tracked across all seven match-era
  trials - NOT derived from the bridge dataset.
- **Pre-registered strata** (assigned per match-pair by events in
  EITHER arm, hierarchical): S2 = top-two seats within 10 pts with max
  >= 85 at any deal boundary; else S1 = any seat >= 85 reached; else
  S3 = neither. Report McNemar + placement per stratum (nominal p,
  labeled secondary). Rationale: match-awareness can only pay in S1/S2;
  aggregate-only reporting would repeat the mean-placement dilution
  mistake documented in the bridge stats.
- **Frozen reference**: hearts_ai_search_ref_matchblind_20260724.pt (+
  matching .pth), snapshotted BEFORE any match-aware work; every
  regeneration round validates against THIS artifact, never a moving
  baseline.

## Equity model addenda (second review)

- **Calibration decoupled from training**: training mix stays
  50/30/10/10 seeded; calibration is measured on a DEDICATED large
  natural-play holdout (>=5k matches of natural self-play states) -
  reliability diagrams + Brier per placement. The 10% natural slice in
  training is not the calibration denominator.
- **Correlated labels (fourth review)**: ~3.3 boundary states per match
  share ONE outcome - effective sample size ~= match count, not state
  count. Train/val split is BY MATCH_ID (the generator records it per
  row for exactly this); calibration CIs are computed against MATCH
  counts. Per-stratum sufficiency: after the natural holdout lands,
  count S2 matches (the thinnest, most important stratum); if <500,
  EXTEND natural generation until >=500 S2 matches before any
  calibration claim.
- **Terminal states analytic**: max>=100 states excluded from training
  (exact placements known); near-terminal net outputs checked against
  analytic placements as a correctness probe.
- **Canonicalization test before adoption**: candidate input (my score,
  sorted opponent scores) is ~6x data-efficient IF seats are
  exchangeable - but next-deal pass direction breaks neighbor symmetry
  at deal boundaries, so exchangeability is tested on held-out natural
  data (full-input vs canonical log-loss; permutation-spread of the
  full model) and adopted only if the cost is negligible.

## SNR probe implementation (resolves the ordering question)

The probe does NOT require the match-aware C++ integration. It requires
one small LOGGING-ONLY addition to SearchEval (--probe-log): at sampled
decision points, dump per-action x per-determinization completed-rollout
DEAL SCORE 4-vectors plus current match totals. Equity arithmetic is
then applied OFFLINE in Python over the logged rollout outcomes. No such
logs exist today (SearchEval writes only per-deal outcome diffs), so the
logging mode is a prerequisite; it changes no scoring behavior, and the
gate ordering stands: full scoring/ctx integration only after the probe
passes. Caveat stated: probe rollouts use the match-BLIND rollout
policy, so measured SNR is an approximation of the final system's -
acceptable for go/no-go.

**Action-flip rate (third review, part of the same offline analysis):**
over logged decisions, report how often argmax under
equity(totals + deal scores) differs from argmax under raw deal points,
BROKEN OUT BY the pre-registered strata S1/S2/S3. This bounds the
achievable effect before any C++ scoring work or 400-match validation:
low flip rate outside S2 => the edge is structurally small regardless
of equity-model quality (change the plan); high flip rate concentrated
in S2 => power the validation on that stratum rather than the diluted
aggregate. Upgrades the SNR gate from one bit to a magnitude estimate.

## Behavioral suite addendum: desperation pre-registration

Under the `win` objective, P(place 1) ~= 0 makes variance free:
near-hopeless positions SHOULD produce erratic-looking, variance-seeking
play. Pre-registered as correct behavior. The suite therefore separates
"hopeless" states (P1 below ~2%) - where erratic play is expected and
not a plumbing failure - from "trailing but alive" states, which carry
the moon-attempt diagnostic.

## Cloud policy (REVISED fourth review - supersedes "4090s not H100s")

**TRAIN IN CLOUD, EVALUATE LOCALLY.** Numerical equivalence only
matters where two arms are compared and a difference is tested.
Training needs no bit-equivalence - it emits a checkpoint, which comes
home and is evaluated on the 4090 against a locally-run baseline with
both arms on shared hardware. This DISSOLVES the 2026-07-17 H100
equivalence failure (that run made measurement portable; measurement is
the one thing that must not be).

- Corollaries: never split a paired comparison across heterogeneous
  hardware; never split one dataset generation across heterogeneous
  hardware; any arm-vs-arm run stays on one machine class.
- H100 rental IS justified for: Phase 2 teacher distillation
  (visit-count targets, large-batch supervised), the v5-L RL ladder
  (VRAM + bandwidth bind - strongest case), exploiter league
  (parallel concurrent runs), hyperparameter sweeps. NOT for search
  evaluation (latency-bound, CPU-heavy, would idle an H100).
- Search-based validation stays LOCAL at funded N (measured: ~21
  s/pair clean, ~700/h sharded -> N=800 ~= 70 min).
- CPU dataset generation is MORE portable than GPU work (fp32 x86):
  regeneration rounds may shard across cheap CPU instances AFTER a
  determinism audit (reproduce one cloud shard locally, bit-compare);
  shards must be reproducible from (seed, shard_index) alone.
- Standing rule: every cloud-trained checkpoint is evaluated locally
  against a locally-run baseline before any promotion decision.

## Gate thresholds (pre-registered 2026-07-25, BEFORE data; fifth review)

**HALT IS THE DEFAULT.** Proceeding past any gate requires an explicit
numeric PASS; ambiguous, missing, or errored results halt and report.
Every gate emits a machine-readable verdict JSON
(equity_data/verdicts/<gate>.json): {gate, metrics, thresholds, pass,
branch (if any), git_sha, data_sha256, timestamp}. All CIs and ECE/Brier
denominators are MATCH-level (cluster bootstrap by match_id, 1000
resamples), never state-level.

1. **Calibration** (natural holdout): ECE (10 equal-mass bins, averaged
   over P(place 1..4)) <= 0.03 aggregate AND <= 0.05 per stratum
   (S1/S2/S3); Brier reported alongside. S2 denominator >= 500 matches
   (extension rule) before this gate can pass.
2. **Beat-the-baseline** (binned lookup over (my score, max opponent,
   deals) + small logistic, built from the SAME training matches):
   net Brier <= baseline Brier - 0.005 aggregate, and net no worse
   than baseline + 0.002 in ANY stratum. Fail => the net is not
   load-bearing: use the lookup in the rollout scorer or halt.
3. **Near-terminal correctness**: on holdout states whose match ended
   within the NEXT deal: ECE <= 0.06 and net Brier <= baseline Brier
   on the same slice.
4. **Canonicalization (a BRANCH, not pass/fail)**: adopt (my score,
   sorted opponents) iff holdout log-loss degrades by < 0.005
   nats/state vs the full-input model; automation picks the branch and
   records which in the verdict artifact.
5. **Flip-rate floor** (from --probe-log offline analysis): S2
   action-flip rate >= 5% required to proceed to the C++ scoring
   integration; below => HALT and reconsider (the edge is structurally
   small regardless of equity-model quality).
6. **SNR**: median |per-action equity gap| / SE(K=64 mean) >= 1.0
   within S2 decisions to proceed; report alongside the same ratio for
   deal-point scoring as the known-good reference.

## Regeneration-round default (fifth review, point 10)

Label noise, not state coverage, is the binding constraint (~5-dim
smooth input, one-hot outcome labels). Regeneration rounds re-run 4-8
matches from each IDENTICAL seeded state and train on the AVERAGED soft
target - same match budget, substantially lower label variance. (v1,
already in flight, uses one outcome per state; acceptable for the first
model, superseded at regeneration.)

## Build order
freeze match-blind reference -> equity data gen (CPU, seeded) -> equity
model + natural-holdout calibration + canonicalization test ->
--probe-log C++ addition + offline SNR probe (go/no-go) -> full C++
integration -> behavioral diagnostics (with desperation carve-out) ->
paired validation per the pre-registration -> ledger + guard-evolution
decision.
