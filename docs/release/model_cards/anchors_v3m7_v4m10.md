# Model card — the anchor field (v3-m7 and v4-m10)

> STATUS: DRAFT - not yet released; project ongoing.

## Identity
- **v3-m7:** hearts_ai_grandmaster_v3_milestone7.pt (md5 2e1f46e2);
  legacy v3-generation MLP lineage. The eval scripts bind the path
  `legacy_v3_pass238/hearts_ai_grandmaster_v3_milestone7.pt`.
- **v4-m10:** hearts_ai_grandmaster_v4m10.pt (raw, md5 36bba8fa) /
  hearts_ai_search_v4m10.pt (search, md5 7e28477b); v4-generation
  lineage (~7.9 MB traces).
- Both produced by the automated loop during the early-July eras
  (docs/release/JOURNEY.md eras 1-2 interlude).

## Role in the record
The **fixed neutral opponent field**. Rule: strength is measured
against frozen neutral anchors, never against the training opponent
(docs/experiment_rules.md). Neutral-raw gates seat candidates vs 3x
v3-m7; match gates alternate v3-m7 and v4-m10 fields so gains stay
general rather than anchor-shaped (match_eval.py). Without these exact
files, none of the project's comparative numbers can be reproduced.

## Measured properties
- v4-m10 vs the current champion (2,000 deals, 16,000 games/side):
  7.35 avg pts/deal, 27.3% deal wins vs champion's 6.11 / 36.1%
  (analyzer_history.csv 2026-07-28) — weaker overall, BUT:
- v4-m10 defends moons better than the champion lineage: 61.1% vs
  51.5% defense rate; and in-instrument it holds the SEL attacker to
  0.237 moons/deal vs 0.367 (CIs disjoint;
  docs/exploiter_league_phaseA.md) — the ordering check that validated
  the moon-defense hole as real.
- v3-m7 numbers exist relative to every gate that used it (see the
  ledger); it defines the "neutral raw" scale.

## Known weaknesses (stated honestly)
- Both are frozen historical artifacts of earlier generations; they
  are calibration instruments, not competitive players.

## Intended use
Seat them as the opponent field when re-running gates
(neutral_raw_eval.py, match_eval.py, analyzer). Do not retrain or
fine-tune them; their value is that they never change.
