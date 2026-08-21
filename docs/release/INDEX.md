# Release documentation (DRAFTS - project not yet open-sourced)

Contract for everything in this directory:
- DERIVED documents. Primary sources (docs/speed_ledger.md,
  docs/experiment_rules.md, docs/ROADMAP.md, preregs, verdict JSONs,
  git history + notes) remain canonical; on conflict, the primary
  source wins and the release doc gets fixed.
- Every quantitative claim carries a citation: (ledger entry date),
  (verdict JSON), or (commit hash). No untraceable claims.
- Claims discipline (RELEASE_PLAN sec. 4.5): measured claims only,
  with N and CI; user calibration matches are n=1 anecdotes; no
  "superhuman" language.
- Updated as eras complete, not reconstructed at release time.

Reading order for an outside observer:
1. The repository root README.md - front door: what, headline results,
   build, play, models (added at the 2026-08-12 release-prep pass).
2. JOURNEY.md - the narrative: eras, experiments, failures, pivots.
3. ARCHITECTURE.md - the system: engine, net, search, training, gates,
   data formats, cloud, web app.
4. METHODOLOGY.md - the measurement discipline (paired deals, CRN,
   neutral anchors, pre-registration, halt-is-default, re-powering).
5. RESULTS.md - claim table: every major result, N, CI, citation.
6. REPRODUCING.md - toolchain pins, builds, rerunning gates,
   regenerating data from seeds.
7. model_cards/ - per released checkpoint.

Status: drafts 2026-08-01; update pass 2026-08-21 (6th promotion — the
gated ensemble; ARCHITECTURE §6b, JOURNEY coda, model card
hearts_ensemble_8d7816d1.md); update pass 2026-08-09 (era 8 closed —
expert-iter v2 decisive negative; era 9 exploiter league round 1 +
round 2 in flight); update pass 2026-08-12 (era 9 closed — league
rounds 2-3 halted; era 10 closed — visit-count distillation both
picks HALT, docs/phase2_visitcount_results.md; era 11 v6 opened —
prereg signed, stages 0-2 verified, stage-2 bank generation in
flight). Scope rule from the 08-09 pass: these docs narrate
the AI's development; the web app appears only as a user tool with its
use cases (ARCHITECTURE sec. 8) — its own development history is not
release-doc material.

## Glossary (project-specific terms, used across all docs here)

| Term | Meaning |
|---|---|
| raw net / raw player | The policy network alone — one forward pass per decision, no search. |
| search player | Flat Monte Carlo: sample K hidden-hand completions, roll each legal action out to deal end with the raw net playing all seats, pick the best mean. |
| determinization / K | One sampled assignment of the unseen cards consistent with voids and known cards; K = how many are sampled per decision (64 base, 256 endgame). |
| belief head | Net output predicting which opponent holds each unseen card; weights the determinization sampling. |
| trace | TorchScript export of a net (`.pt`), the form the C++ engine consumes; re-exported on every promotion. |
| anchor (v3-m7, v4-m10) | Frozen older-generation nets forming the fixed neutral opponent field every comparison seats against. |
| CRN | Common random numbers: both arms play identical deal/seat/rollout seeds so per-deal deltas pair and variance cancels. |
| gate | A promotion criterion the candidate must PASS significantly (e.g. match gate n=3,200, alpha=0.05 placement). |
| guard | A non-regression bound: candidate is rejected only if the one-sided 95% upper bound of its regression exceeds a margin (+0.3 pts/deal). |
| UB | One-sided 95% upper confidence bound (the guard's decision statistic). |
| match era / match-to-100 | The objective since 2026-07-24: placement in games played to 100 points, not per-deal score. |
| deal-point vs equity scoring | Two rollout leaf scorers: round points (per-deal play) vs the equity net's P(win the match) (match-aware play). |
| equity net | Small CPU net mapping post-deal match score-state (10 dims) to each seat's P(place 1); see ARCHITECTURE sec. 3 lifecycle note. |
| selection vs confirmation | Anything chosen by looking at results is a selection (estimate only); claims come from a fresh one-shot pre-registered confirmation. |
| prereg | Pre-registration: criterion and action per outcome branch fixed in a signed doc before results exist; HALT is the default on gate failure. |
| SDPA | `F.scaled_dot_product_attention` — PyTorch's fused attention kernel (adopted via its own pre-registered gate). |
| AOTI | AOTInductor: ahead-of-time compiled serving path (Linux/cloud; certified on H100). |
| ISMCTS | Information-set Monte Carlo tree search — built, measured, closed for strength (kept as a possible target generator). |
| shooter probe | SearchPlayer in shooter mode: a moon-seeking search attacker used as a frozen measurement instrument. AGG always shoots; SEL commits only when moon equity beats normal play. |
| clone attacker | Small net distilled from a search-shooter, certified to retain >=50% of the teacher's moon rate, so defended-game generation runs at trace speed. |
| defense gate | 64 CRN-paired seed-matches vs the frozen SEL probe; a candidate passes by conceding significantly fewer moons (one-sided alpha=0.05). |
| moon-alive | Harness label on a recorded decision: no non-attacker seat has taken a point yet this deal, i.e. the moon threat is live. Selects defense-stream training samples; never enters the observation. |
| anchored distillation | Round-2 recipe: hard-CE imitation of search-defender decisions in moon-alive states + self-distillation to the baseline's own argmax in ordinary states, making drift an offline-measurable quantity (the >=97% agreement drift screen). |
| Perilune | The project's public web app (hearts_web/, play.perilune.ai) — named for the lowest point of a lunar orbit: moon + lowest-score-wins. |

## Known documentation debts (from the 2026-08-01 drafting audit)
- [NEEDS CITATION] v5 -1.233 gate sample size (commit a4136b5 lacks n).
- [NEEDS CITATION] behavioral spine-gate pass/fail was never given a
  ledger entry / verdict JSON (tooling ea2d33e, behave_*.csv exist).
- flip_snr numbers differ between ledger 07-25 (36.8%/0.41/0.57) and
  equity_data/verdicts/flip_snr.json (38.9%/0.62/0.86); same HALT
  verdict both places - resolve against the analysis script pre-release.
- ~~REPRODUCING.md needs a clean-machine build verification + Python
  pins.~~ DONE 2026-08-12: requirements.lock committed; clean-clone
  SelfPlayGen build verified (REPRODUCING.md sec. 1).
- Stale figures in match_aware_search_design.md (N=8000 "local ~5.8h")
  are superseded by the ledger - noted in JOURNEY, doc left as-is
  (historical record).

Added at the 2026-08-09 update pass:
- Exploiter-league round 1 defense-gate trial deltas are cited from
  the per-trial verdict JSONs + the r2 prereg summary; the ledger
  entry for the round-1 PPO trials should be confirmed present
  pre-release.
- Era 9 / round 2 sections describe an IN-FLIGHT experiment - rewrite
  to past tense with final verdicts when round 2 concludes.
