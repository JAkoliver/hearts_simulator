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

## Deliberately deferred / skipped
- Search-side match-awareness in C++ (end-goal is a raw net; revisit
  only if search deployment stays primary).
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
