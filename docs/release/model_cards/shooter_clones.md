# Model card — shooter_agg_v1b.pt / shooter_sel_v1.pt (moon-shooter clones)

> STATUS: DRAFT - not yet released; project ongoing.

## Identity
- **Files:** shooter_agg_v1b.pt (md5 ecc0fa03; d256) and
  shooter_sel_v1.pt (md5 54a55f71; d192). Small nets distilled from
  the search-shooter teachers.
- **Modes:** AGG always attempts the moon (dense threat); SEL commits
  only when moon equity beats normal play (realistic threat).

## Provenance
Exploiter league Phase B (docs/exploiter_league_phaseB.md): distilled
from SearchPlayer's shooter mode (moon-probability scoring over shared
determinizations, moon-line rollout continuation, pass-phase shooting
via rewound pass search; instrument frozen at K=64 flat,
docs/exploiter_league_phaseA.md). Certification bar: retain >= 50% of
the teacher's completion rate. SEL passed first try; the AGG d192
attempt HALTED at 46% retention (confirmed at n=500, a real miss) and
the d256 retrain passed on its single registered shot.

## Role in the record
Frozen attack instruments of the exploiter league and the v6 bank:
- League: generate defended games at trace speed and stress moon
  defense (the defense gate seats candidates vs the frozen SEL probe).
- v6 Stage-2 bank: exactly 1/8 of teacher matches include a shooter
  clone in one seat (attacker seat never recorded), so the successor
  sees defended-moon distributions (docs/v6_prereg.md).

## Measured properties
- Teachers (Phase A, 402 matches/combo, vs baseline defenders):
  AGG completes **0.515 moons/deal** (77x the 0.0067 background);
  SEL 0.367 moons/deal at 71.2% attempt rate.
- Clones (certification): AGG d256 **0.2906** moons/deal
  [0.276, 0.306] (n=500 matches); SEL **0.2242** [0.196, 0.252]
  (n=120) (docs/exploiter_league_phaseB.md).
- Ordering check the league rests on: v4-m10 defenders hold SEL to
  0.237 vs baseline's 0.367 (CIs disjoint; phaseA doc).

## Known weaknesses (stated honestly)
- Clones retain roughly half the teachers' completion rate by design
  bar; they are threat generators, not maximal shooters.
- Certified against this project's defender population only.

## Intended use
Opponent seats for defended-game generation and the frozen defense
gate. Identified by md5; do not retrain in place — a changed attacker
invalidates every measurement that used it.
