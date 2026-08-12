# Model card — hearts_equity.pt / equity_v1.pth (equity model)

> STATUS: DRAFT - not yet released; project ongoing.

## Identity
- **Files:** hearts_equity.pt (trace, md5 efdfee07) + equity_v1.pth
  (checkpoint, md5 fa059524). Small CPU-friendly net (~32 KB trace).
- **Inputs/outputs:** 10-dim post-deal match score state (rotated
  totals/100, deals/20, leader distance, pass direction) ->
  P(place 1..4) for the acting seat
  (docs/release/ARCHITECTURE.md sec. 3).

## Provenance
Trained on 30k seeded coverage-mixture matches; calibrated on a
dedicated 5k-match natural holdout (2,000 S2 matches in the clustered
bootstrap; ledger 2026-07-25; verdicts/diagnostics.json).

## Role in the record
The leaf scorer that makes search match-aware: rollout leaves are
scored in P(win the match) instead of deal points; exact placements
substitute at terminal states. Also the source of "match win here"
numbers surfaced in the web app's reviews.

## Measured quality
- Calibration: aggregate ECE 0.0037, below its clustered noise floor
  0.0575; Brier 0.614 aggregate / 0.336 near-terminal
  (verdicts/diagnostics.json, 2026-07-25).
- Selected over both frozen lookup baselines: Brier 0.614 vs 0.645,
  aggregate and per-stratum (verdicts/selection.json).
- Downstream validation: the N=8000 result (see
  hearts_ai_search_match card) is the end-to-end evidence that its
  equity signal buys match wins.

## Known weaknesses (stated honestly)
- At flat K=64, equity-vs-deal-point decision flips are ~97% noise
  (confident flips ~1.6% of tension decisions); the K=256-endgame
  schedule exists precisely to compensate (ledger 2026-07-25;
  rules #15).
- Trained/calibrated on this project's match distribution; no claim of
  calibration on other Hearts populations.

## Intended use
Leaf scoring inside the match-aware search player, and score-state win
probability displays. Not a standalone strength model.
