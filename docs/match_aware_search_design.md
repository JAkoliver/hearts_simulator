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

## Build order
equity data gen (CPU, seeded) -> equity model + calibration check ->
SNR probe (go/no-go) -> C++ integration -> behavioral diagnostics ->
paired validation -> ledger + guard-evolution decision.
