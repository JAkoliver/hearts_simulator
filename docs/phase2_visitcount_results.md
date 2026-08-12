# Phase 2 — Visit-Count Distillation: Results (CLOSED, both picks HALT)

**Prereg:** `docs/phase2_visitcount_prereg.md` (signed 2026-08-09; Stage C
rider and the Stage D entropy re-anchor amendment both signed BEFORE
their data). **Baseline throughout:** 5th match-era promotion
`Hall_of_Fame/hearts_model_milestone_1785322724.pth` (md5 8a89da90).
**Verdicts:** `equity_data/verdicts/p2_*.json`. Concluded 2026-08-11
19:12 (`P2E COMPLETE`).

## One-paragraph summary

The hypothesis was that tree-search **visit counts** — unlike the
equity-ordering targets that failed in expert-iter v1/v2 — encode
preference *strength* and could distill searched knowledge back into
the raw net. The pipeline worked at every stage: recording verified
bit-identical, the strength screen passed, generation delivered 188k
clean records, and distillation achieved exactly its objective (5.6×
reduction in KL to the teacher's visit distributions). The product is
a **weaker player**. Both frozen picks failed the match gate with high
confidence and the search guard band. Faithful imitation of where the
search *looked* softens the policy — entropy rises from 0.43 toward
the teacher's 0.85+ — spreading probability mass onto moves the search
explored but did not endorse. Visit-count targets are **closed for
this net**, joining equity-ordering targets (expert-iter v1/v2), PPO
fine-tuning past its harvest point, and learned leaf evaluation.

## Stage record

- **Stage A (instrument):** TreeSearchPlayer match-aware self-play
  recorder; A/A byte-identical; forced-move stale-visits bug caught
  pre-data. Signal preview at 64 iters: median top-1 share 0.672, 9.2%
  one-hot (bar 80%), entropy 0.917 — graded preferences, as required.
- **Stage B-2 (validity):** same-pair Spearman vs flat-K256 landed in
  the registered judgment zone (reliability datum, not a halt);
  user-approved proceed. `p2_stageb2.json`.
- **Stage C (strength screen):** tree-vs-flat screen PASS (band
  UB ≤ +1.0; tree slightly better). Generation: 2×170 matches, seeds
  210,000,000 / 210,500,000, budget 200 local — 188k records; visit
  profile matched the probes. `p2_strength.json`.
- **Stage D (distillation):** soft-target CE = KL(teacher‖student) on
  play decisions with visits, policy head only, holdout split by match
  (≥153). Teacher-KL reduced 5.6×. Entropy diagnostic initially
  mis-anchored to the baseline (0.434); re-anchored to the teacher
  mean (0.847) by signed amendment → picks `lr3e-05_ep3` (primary) and
  `lr3e-05_ep2`. `p2_stageD_freeze.json`. (Masked-logits NaN trap —
  0 × −inf — fixed with `masked_fill` before any kept run.)
- **Stage E (promotion battery, per pick: match superiority n=3200
  α=0.05 vs the frozen milestone + search guard n=4800 K=32 one-sided
  95% UB ≤ +0.3):**

| pick | match Δplace (SE) | p | search Δ (SE) | UB95 | verdict |
|---|---|---|---|---|---|
| ep3 | **+0.083** (0.015) | ≈1.0 | +0.236 (0.116) | **+0.427** | HALT (both gates) |
| ep2 | **+0.042** (0.015) | 0.997 | +0.132 (0.113) | **+0.318** | HALT (both gates) |

  ep2 telemetry (informs, never gates): win 50.5% vs 51.7%, final
  score +1.46/match worse, moons shot 134 vs 120. The ep2 guard miss
  is marginal (+0.318 vs +0.300) but the gates are conjunctive and its
  match gate had already failed decisively.

## Interpretation (registered)

- **Dose-response consistent with the mechanism:** ep3 (more
  distillation) is worse than ep2 on both gates. The harm scales with
  how faithfully the visit distribution is learned.
- **Exploration is not endorsement.** A search's visit distribution
  includes where it had to look to *reject* moves. Distilling it is
  distilling the question, not the answer. Combined with expert-iter
  v2's finding (equity-ordering targets inert), both available
  encodings of the searched teacher's preferences — value ordering and
  attention allocation — now measure as non-improving for this net.
- **The capacity conversation is next** (per prereg): the network, not
  the teacher signal, is the binding constraint — consistent with the
  earlier finding that ALL search amplification plateaus at the same
  ceiling. v6 architecture/capacity work proceeds under the same match
  gates; no new distillation recipe without a fresh prereg naming a
  genuinely new signal source (e.g. exploiter-league games).

## Ops notes

- Gate workers reduced 12 → 6 after a WinError 1455 commit-charge
  crash (CUDA workers ~2.3GB commit each; documented in
  `run_p2_gates.py`).
- Chain: `ops/run_p2_gates_chain.sh`, log `logs/p2_gates_chain.log`
  (markers P2E). Search-guard shards write CSVs continuously; the
  chain log is silent mid-gate — check shard mtimes before calling it
  stalled.
