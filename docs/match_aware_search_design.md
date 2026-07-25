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
  leader-distance/100 - identical layout to the net's match ctx.
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
- **N = 400 with one interim look at 200** (interim alpha 0.005, final
  ~0.048, O'Brien-Fleming-flavored). MDE at N=400: 3.9-5.6 win-rate
  points across plausible CRN discordance (q=0.10-0.20), below the
  6.25-pt "half of bridge effect" benchmark; N=200 alone cannot detect
  a half-sized effect (MDE 5.6-10.4) and is pre-declared insufficient.
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

## Behavioral suite addendum: desperation pre-registration

Under the `win` objective, P(place 1) ~= 0 makes variance free:
near-hopeless positions SHOULD produce erratic-looking, variance-seeking
play. Pre-registered as correct behavior. The suite therefore separates
"hopeless" states (P1 below ~2%) - where erratic play is expected and
not a plumbing failure - from "trailing but alive" states, which carry
the moon-attempt diagnostic.

## Build order
freeze match-blind reference -> equity data gen (CPU, seeded) -> equity
model + natural-holdout calibration + canonicalization test ->
--probe-log C++ addition + offline SNR probe (go/no-go) -> full C++
integration -> behavioral diagnostics (with desperation carve-out) ->
paired validation per the pre-registration -> ledger + guard-evolution
decision.
