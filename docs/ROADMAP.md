# Roadmap to better-than-human (adopted 2026-07-23)

Goal: the strongest Hearts AI we can build — ultimately a RAW net, fast
on any hardware, above best-human MATCH play (games to 100).

Diagnosis behind the ordering: the network is the measured binding
constraint AND cannot currently be pushed past the existing teacher by
any signal we possess (imitation fails same-lineage and cross-size; PPO
one-shots at the ceiling). Therefore: define the real objective first,
solve the self-improvement loop cheaply, and only then spend on scale.

## Phase 1 — Match-to-100 objective (days; local, cheap)
- Match env wrapper: deal sequences with carried scores, 100-point
  termination, terminal reward = placement/win.
- Observation extension by APPENDING match-context dims (running
  scores, deals elapsed, distance-to-100) — preserves the
  prefix-stability contract (SearchPlayer {550,238,181} probe, legacy
  adapters).
- Zero-init conditioning on the current baseline: extended net is
  behavior-identical at step 0; training learns only score-awareness.
- Match-level paired gate (same deal sequences both sides, win rate +
  mean placement vs the score-blind baseline, neutral anchors) BEFORE
  any training. Telemetry rider (experiment_rules.md #5) with
  score-conditional splits ships with this tooling.

## Phase 2 — Prove the self-improvement loop at 7.6M (the make-or-break)
One demonstrated generation of: search with current net -> richer
targets -> train -> measurably stronger at MATCH play. Key untested
variable: TreeSearchPlayer visit-count targets (+ match-outcome value
targets) instead of flat-search soft pi. Everything is cheap at this
size (fast search, ~4-min distills, ~2-min raw gates). If no target
formulation compounds, learn it here, not after scaling.

## Phase 3 — Scale INSIDE the working loop
v5-L/XL initialized from a distill of the current baseline, pushed past
it by the proven loop. Cloud (certified H100+AOTI pipeline, ~3h
iterations) becomes worth spending on exactly here, with user approval
per rental.

## Continuous (all phases)
- Telemetry diff on every candidate; history CSV; flags in reports.
- Periodic user calibration matches — the only ground-truth metric.
- Exploiter-style opponents folded in IF match training leaves blind
  spots (e.g. the measured 2x moon-concession hole of the per-deal
  lineage).

## DONE: match-aware search (built 2026-07-25/26, VALIDATED 2026-07-27)
Both queued steps completed and the build validated at N=8000 (cloud
fleet, pre-registered single analysis): match-aware search +4.44
win-pts vs the frozen match-blind reference (McNemar p~5e-11, all 8
shards positive; P2->P1 AND P2->P4 conversion = win-equity behavior).
Rules #15 (K=64/256) and #16 (ceiling config + evolved guard) govern.
Guard now runs both arms match-aware. Phase 2 teacher = match-aware
search. NEXT TRAINING SEQUENCE (adopted 2026-07-28): (1) resume
match-mode PPO under the evolved guard with diversified gate anchors
(v3-m7 + v4-m10); (2) ON PPO PLATEAU (pooled null over >=3-4 trials):
run match-aware EXPERT ITERATION once, properly gated - generate match
states with the match-aware search as actor, distill its decisions
(556 ctx, sharpen ~2.0, split holdout by deal) into the raw net. This
is NOT the closed same-lineage recipe: the teacher demonstrably makes
different, better decisions in score context than the student's own
knowledge (+4.4 win-pts of equity signal).

## CLOSED 2026-07-31: match-aware expert iteration, recipe v1
One-shot gate: FAIL, decisively (win 39.9 v 50.3, placement +0.292 at
~17 SE). Mechanism documented in the ledger: equity-scored teachers
emit near-uniform policies; soft distills un-sharpen, hard-argmax
distills copy COIN FLIPS in the ~73% of flat states and overwrite real
knowledge. The +4.4-win-pt search edge exists at decision time but is
NOT extractable by whole-distribution imitation.

## Queued: expert iteration v2 - FILTERED targets (new pre-registration required)
Only if attempted again: train ONLY on decisions where the teacher's
equity spread is significant (flip-confident states, ~4-6% of
decisions, the probe machinery already identifies them), and/or mix a
per-deal anchor loss to protect existing knowledge. Smaller data need
(the informative slice), cheap to generate with the fixed pipeline.
Requires its own pre-registered gate battery.

## Queued: K-endgame threshold optimization (added 2026-07-29, low priority)
The >=85 trigger for K=256 (rules #15) is a DESIGN HEURISTIC ("one deal
from elimination"), never tested against 75/80/90 - the probe measured
K=256's value inside the >=85 band, and the N=8000 validation validated
the package, not the threshold. Plan, in order:
1. FREE reanalysis (CPU, hours): re-bin confident-flip rate in
   probe_decisions_v2_k256.csv by max-total bands (70-75/75-80/80-85/
   85-90/90+); same data gives each threshold's K=256 deal fraction
   (cost). Near-zero flips below 85 => threshold already right, close
   this item.
2. Only if (1) nominates a candidate: ONE paired match run (current 85
   vs candidate, CRN, both arms otherwise identical, n~3200, ~2 days
   local). No sweeps - expected gain is a fraction of a win-point.
Run during a quiet stretch; never ahead of expert iteration.

## Queued: equity-net drift check (added 2026-08-01 - trigger-based, cheap-first)
hearts_equity.pt is FROZEN: trained once 2026-07-25 on 30k seeded + 5k
natural matches from the 3rd match-era baseline; promotions re-export
traces but never retrain it (ARCHITECTURE sec. 3 lifecycle note). Its
P(win | score-state) map is conditional on the play population that
generated that data, so each promotion adds a little distribution
drift. Discipline, in order:
1. CHEAP DIAGNOSTIC FIRST, no retrain: generate a small fresh natural
   holdout with the current champion (gen_equity_data.py) and score the
   frozen net's Brier/ECE against the 2026-07-25 calibration noise
   floor (train_equity.py holdout machinery - all tooling exists).
   Run during a quiet stretch, or BEFORE the next campaign that leans
   on equity calibration.
2. RETRAIN ONLY on measured degradation. A retrained equity net
   changes the N=8000-validated package (rules #16): the search
   package must be re-validated (powered paired run) before the new
   net becomes ceiling config or Phase 2 teacher.
3. NEVER swap the equity net mid-generation: v2 banks embed the frozen
   net's per-decision equity stats; a mid-bank swap silently mixes two
   scoring functions in one dataset.
Mitigating context: expert-iter v2's confidence filter consumes
within-decision equity ORDERING (gaps), which is more drift-robust
than absolute levels - drift concern is real but not urgent.

## CURRENT MAIN LINE (2026-08-18): the gated-ensemble program
The best raw player may be several raw nets that switch: a DEFAULT net
(the champion) + SPECIALISTS + a public-information ROUTER, exported as
one 882-input traced module. Measured 2026-08-18: champion + the obs-v2
student arm b routed by arm b's own moon head concedes -0.247
moons/match vs the champion (twice arm b alone, ~r1-t3) at no
measurable strength cost; NI passed on the cruder variant. Program doc
(design space, currency, sequencing, limits): docs/gated_ensemble_
program.md. First battery: docs/exploiter_league_r6_prereg.md. Next:
specialist ladder -> router refinements -> round 7 (train the specialist
with the default frozen) -> round 8 (search-judged router) -> more
domains behind audits.

## Queued: v6 network (added 2026-07-29 - trigger condition, not a date)
**STATUS 2026-08-16: RUN AND SHELVED.** The v6 that actually ran
(docs/v6_prereg.md, signed 08-11) departed from this queue entry: it
bundled scale (2.55x) + structure (seat tokens, aux heads) + obs v2 in
one from-scratch distill rather than the pure-scale-first, warm-started,
A/B-isolated sequence below. Outcome: structure helped imitation, scale
bought nothing, match-PPO from the fresh distill damaged it (Stage 4
halted 08-13), and the data probe found strength saturated at 1.0-1.5M
records (08-16). Post-mortem and the v6(2) design that follows THIS
entry's sequencing: docs/v6_postmortem.md. Main compute program now:
exploiter league round 4 (docs/exploiter_league_r4_prereg.md).

TRIGGER: the improvement loop (PPO alternation + match-aware expert
iteration) demonstrably compounding AND the 7.6M v5 stops responding
across several cycles with powered gates. Rationale: the v5-L lesson -
you cannot imitate your way to a bigger/newer net; scale only inside a
working loop (Phase 3). Sequence when triggered:
1. v5-L-shaped scale-up FIRST (d=448 L=8, init from a distill of the
   then-champion, pushed by the loop; zero retooling). SPEC LOCK
   (2026-07-29 review): heads chosen for integer, tensor-core-aligned
   head_dim - d=448 => 7 heads x head_dim 64 (SDPA sweet spot on Ada).
   Current v5-M is 10 heads x 32 at d=320 (verified; NOT 6 heads).
   PURE SCALE ONLY: no bundled architecture deltas - the run answers
   exactly one question (does scale compound inside the loop?).
1b. Factorized card embeddings (identity_52 + additive suit_4/rank_13,
   keep the identity term = strict superset) as a SEPARATE ~5-min
   distill A/B after the pure scale-up - zero C++ retooling (internal
   to card_embed), but never bundled with the scale variable.
   Expectation: small/nil (52 identities are not data-starved);
   cheap to check, attribution kept clean.
2. Architectural v6 second, as small-scale controlled A/Bs: native
   match conditioning (score state as tokens/FiLM, not a bolt-on
   projection), distributional value head (placement/points
   distribution - match equity is ill-served by a scalar), belief-
   policy fusion (belief BCE 0.270 bounds determinization quality).
   Belief-fusion A/B arm (2026-07-29 review): per-card CATEGORICAL
   belief over opponents instead of 3 independent BCE logits -
   requires a 4th "not-with-an-opponent" class or masking to the
   unseen set (cards can be in-hand/played/on-table); enforces only
   per-card sum-to-1 (hand-COUNT consistency still lives in the
   determinizer), and changes trace belief semantics = small C++
   consumer change. Judge on determinization quality + downstream
   search strength, not elegance.
3. Observation-revision rider (pay only when the v6 obs surgery is
   already open for match conditioning): add a captured-by-seat
   channel - who-played-what + timing gives which trick a card fell
   in but NOT the trick winner (lead order is implicit). NOT the moon
   fix: running deal scores (obs 156-159) already expose
   moon-in-progress; the 51.5% defense rate is curriculum (exploiter
   league is the treatment). Pre-registered diagnostic first: bin
   moon-defense failures by capture-attribution ambiguity; if
   failures don't concentrate there, the channel is a rider only.
   NOT an architecture fix: the moon-defense hole (data/objective gap -
   exploiter league).
   METADATA RIDER (2026-08-01): any obs surgery also attaches explicit
   obs-dim/version metadata to exported traces and retires
   ProbeObsDim's length-probe list (the accumulating-fragility note in
   docs/release/ARCHITECTURE.md sec. 1) - paid together with the
   surgery, never as a standalone retooling.
Obs-format changes ripple through C++/traces/exports - pay that cost
only for measured headroom.

## Superseded queue entry (kept for history)
Queued: match-aware search (added 2026-07-24 after the first guard veto)
The search player is match-BLIND (rollouts score deal points), while the
raw net is now match-aware - their advantages are incomparable and the
guard currently protects the possibly-wrong crown jewel (trial 6:
best-ever match candidate vetoed for search-substrate regression +0.48).
Sequenced plan:
1. PREREQUISITE (cheap, do soon): C++ bridge to seat the search player
   in the match harness -> measure match-blind search vs match-aware raw
   AT MATCH PLAY. If raw already wins matches, re-point the guard at raw
   per-deal strength; if search still wins, its protection stays earned.
2. BUILD match-aware search when EITHER (a) guard rejections become the
   norm (recurring match-vs-substrate trade, not a one-off) OR (b)
   Phase 2 starts in earnest - the self-improvement loop's teacher must
   be match-aware or its targets pull the student back toward per-deal
   play. Scope: match context into SearchPlayer rollout evaluation +
   556-dim traced value path + guard swap to match-level search.
   This is also the ceiling configuration (search strength + score
   awareness) for "best AI possible."

## Deliberately deferred / skipped
- (superseded by the queued item above) Search-side match-awareness was
  deferred outright while the end-goal was framed as raw-only.
- Standalone exploiter league (folded into match training unless gaps
  persist).
- Any further per-deal refresh recipes (closed; see experiment_rules.md).

## Asset inventory (what everything so far contributes)
Gates/measurement discipline; speed stack (SDPA, buckets, AOTI, gate
sharding, certified cloud pipeline); PPO machinery (proven on fresh
objectives); v5 architecture family + tracing/serving knowledge;
TreeSearchPlayer (target generator, still unused for training);
analyzer.py + eval_search_pair.py + neutral_raw_eval.py + entropy_eval.py
instruments; 12.5k-deal fresh bank (evaluation stock); Hall of Fame
lineage; the negative-results map (experiment_rules.md).
