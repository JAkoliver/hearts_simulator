# Expert Iteration v2 — filtered-target pre-registration (DRAFT, awaiting user sign-off)

Status: DRAFT 2026-07-31. Nothing below is binding until the user approves;
the v2 gate battery is untouched until then.

## Hypothesis (from v1's measured failure)
The match-aware teacher's advantage is concentrated in decisions where its
top-two equity gap is statistically confident. v1 failed because hard-label
training copied the teacher's argmax in the ~94% of decisions where that
argmax is sampling noise. Training ONLY on confident decisions — with the
flat-state policy anchored to the baseline — transfers signal without
overwriting knowledge.

## Data (new generation, record format v2)
- Record format v2 (848B + 32B file header, see selfplay_gen.cpp): every
  decision is recorded (confident or not) with per-decision search
  statistics: best/second equity, gap SE, n_determinizations, phase flag,
  tension flag, match_id. FILTERING HAPPENS AT TRAIN TIME, NEVER AT
  GENERATION TIME — the raw bank stays recipe-agnostic.
- Confidence definition (FROZEN, from the existing flip/SNR analysis — no
  new thresholds invented): confident iff (eq_best − eq_second) > 2 ×
  gap_SE, evaluated at the K actually used for that decision.
- Generation mix: same four families as v1 (natural + knife/leader/trail
  seeded), 02:00–07:00 windows, chunked, hardened pipeline. Target:
  ≥30,000 confident PLAY-phase records (passing-phase records are tagged
  and excluded from v2 policy training). Expected 1–2 nights.

## Training recipe (tuned on holdout ONLY, as in v1)
- Warm start from the current champion (hash recorded at run time).
- Policy loss: one-hot CE on confident play-phase decisions ONLY.
- Anchor loss: KL(candidate ‖ baseline policy) on a uniformly-sampled
  non-confident subset, weight tuned on holdout (candidate values
  {0.25, 1.0}); prevents flat-state drift.
- Value/belief losses unchanged (value-head overfit noted in v1: prefer
  the best-holdout epoch).
- Holdout: by-match tails, 10%. Diagnostics gate NOTHING (rules #5) but
  tune the recipe: entropy must stay within 2x of baseline; stratified
  teacher-match reported on confident vs non-confident slices; a
  candidate goes to the battery only when the recipe is frozen.

## Gate battery (ONE SHOT, halt-is-default)
- Match gate n=3200, mixed v3-m7/v4-m10 anchors, alpha=0.05 placement.
- Evolved match-aware search guard n=4800, one-sided 95% UB <= +0.3.
- Promote only if both pass; baseline und traces update via the standard
  promotion path.

## Stop rule (binding)
If v2 FAILS its battery: expert iteration via equity-scored flat search
CLOSES ENTIRELY (v1 + v2 = whole-distribution and filtered variants both
measured). The next Phase 2 attempt must use a structurally different
target family (TreeSearchPlayer visit counts, which encode preference
strength natively) under its own pre-registration.

## Honest power note
The search's +4.4 win-pts comes from equity judgment at EVERY decision
with rollout context; a raw net absorbing only confident-state behavior
captures an unknown fraction. If the true effect is below the gate's
~0.028-placement MDE, v2 fails "correctly." This is a mechanism-directed
shot, not a sure one.
