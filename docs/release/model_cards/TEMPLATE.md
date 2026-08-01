# Model card — <artifact name>

> STATUS: DRAFT - not yet released; project ongoing.

One card per released checkpoint/trace (docs/RELEASE_PLAN.md sec. 3).
Every number cited to a primary source (ledger entry date, verdict
JSON, or commit hash); measured claims only, with N and CI — no
"superhuman" language; user calibration matches are n=1 anecdotes.

## Identity
- **Files:** <.pth checkpoint / .pt trace(s)>
- **Hashes:** <md5/sha as recorded in ledger or experiment_ledger>
- **Architecture:** <e.g. HeartsNetV5, d=…, L=…, heads, param count>
  (see docs/release/ARCHITECTURE.md sec. 2)
- **Observation width:** <550 / 556 / legacy prefix>
- **Date frozen / promoted:** <date + commit>

## Provenance
- Lineage: <parent milestone(s), training recipe: distill / PPO /
  match-mode PPO, data bank(s) with seeds>
- Promotion path: <gates passed, with numbers + citations>

## Role in the record
- <What measurements this artifact anchors; why it must not drift.>

## Measured strength
- <Each claim: number, N, CI/p, opponent field, citation.>
- Opponent-field caveat: all strength numbers are vs the stated anchor
  field and configuration; nothing generalizes past them without new
  measurement (docs/expert_iter_v2_prereg.md, external validity note).

## Known weaknesses (stated honestly)
- <e.g. moon-defense rate vs v4-m10; telemetry citations.>

## Intended use
- <Play configuration: raw argmax / search player with K schedule /
  gate anchor. What it should NOT be used for.>
