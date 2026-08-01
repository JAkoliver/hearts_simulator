# Model card — hearts_ai_search_ref_matchblind_20260724

> STATUS: DRAFT - not yet released; project ongoing.

## Identity
- **Files:** hearts_ai_search_ref_matchblind_20260724.pt (traced search
  net) + matching .pth checkpoint.
- **Hash:** md5 a1a0be31… (verified on-pod before the pilot A/A run,
  ledger 2026-07-26; RELEASE_PLAN sec. 3).
- **Architecture:** HeartsNetV5 card-token transformer, 550-dim
  observation (match-BLIND — no match-context dims in this trace); the
  v5-M production shape is d=320, L=6, 7.6M params
  (docs/release/ARCHITECTURE.md sec. 2; a4136b5).
- **Date frozen:** 2026-07-24, snapshotted BEFORE any match-aware
  search work began (docs/match_aware_search_design.md, "Frozen
  reference").

## Provenance
Snapshot of the deployed search trace at the end of match-era night 1 —
the lineage that had absorbed the v5 architecture promotion (293c0fd),
the first raw-line PPO promotion (fff75a2), and the first three
match-era PPO promotions (ledger 2026-07-24; the trace itself carries
no match conditioning). Frozen specifically so that every match-aware
regeneration round validates against THIS artifact, never a moving
baseline (match_aware_search_design.md).

## Role in the record
This is the **comparator of the project's headline result**. In the
N=8000 cloud validation it was arm B: the match-blind search player
(K=64/256 schedule, pass search) against arm A, the identical search
stack plus match context and equity leaf scoring (ledger 2026-07-26/27).
Its release is described as non-negotiable for reproducibility of that
result (RELEASE_PLAN sec. 3).

## Measured strength
- **N=8000 validation (as the reference arm):** won 44.47% of matches
  vs the match-aware arm's 48.91% — a deficit of 4.44 win-points
  (SE 0.68), McNemar one-sided p≈5e-11, discordant 1668:1313
  (q=0.373), all 8 shards against it; its MEAN placement was better
  (1.980 vs 2.098) — the match-aware arm trades placement for wins
  (ledger 2026-07-27; docs/release/RESULTS.md).
- Field/config: three v3-m7 anchor seats, paired CRN seeds, K=64 base /
  K=256 endgame, bf16, community RTX 3090 fleet; pairs never split
  across nodes (ledger 2026-07-26).
- Lineage per-deal strength context: the deployed search player of this
  lineage measured -1.016 pts/deal (SE 0.234, n=1,200) stronger than
  the 07-14 calibration opponent v4-m10 (ledger 2026-07-23).
- A/A determinism: ref-vs-ref on the pilot pod produced 20/20
  bit-identical pairs (ledger 2026-07-26) — the artifact double-checks
  the harness as well as the science.

## Known weaknesses (stated honestly)
- **Match-blind by construction:** rollouts score deal points; it
  cannot represent end-of-match reasoning. That is its purpose as a
  control, and the measured cost is the +4.44 win-point deficit above.
- Inherited lineage weakness: moon defense ~51.5% vs v4-m10's 61.1%
  (moons conceded 272 vs 151 per 2,000 deals) — telemetry, informs
  never gates (analyzer_history.csv 2026-07-28; rule #5).

## Intended use
- Reference arm for reproducing/reanalyzing the N=8000 validation
  (equity_data/validation_v1/ + analyze_validation.py).
- Baseline for any future match-aware regeneration round (per the
  design doc's frozen-reference rule).
- Not a current champion: superseded for play by the match-aware
  lineage (promotions #4–#5, milestones cbfde942 / 8a89da90).
