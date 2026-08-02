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
  seeded), 02:00–07:00 windows (+ gentle daytime as approved), chunked,
  hardened pipeline.
- TARGET (REVISED by user 2026-08-01, HARD threshold, supersedes the
  50k-total stop): per-type RESERVES sufficient to build EVERY mix in
  the selection experiment at 50k confident records each - i.e.
  NATURAL-confident >= 50,000 AND each of the six seeded families
  >= 8,400 confident (binding mixes: natural-only and balanced
  seeded-spread; mixes share reserves, so max-not-sum governs).
  Passing-phase records are tagged and excluded from policy training.
  NO further step until all seven counters are met. Seeded families
  jittered per match (--start-jitter: knife 2, leader/trail 3,
  mid/asym 6, early 7) - neighborhoods, not points.
- Cost checkpoint: after ~2h of generation the realized confident share
  is measured; if the implied total to 50k exceeds ~60 machine-hours,
  the revised forecast goes to the user before continuing.

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

## Mix-selection stage (added 2026-08-01, user-directed; runs BEFORE the battery)
- Seeded generation widened to SIX families (knife 90/88/86/84, mid
  75/73/70/68, leader 92/70/68/66, asym 85/60/55/50, trail 60/88/86/84,
  early 60/58/40/38) interleaved 1:1 with natural chunks - manifold
  coverage over per-family depth.
- Candidate MIXES (each summing to 50k confident records, drawn from the
  per-family reserves): (a) 60/40 natural/seeded, (b) 50/50,
  (c) 35/65, (d) natural-only, (e) seeded-only-spread.
- DESIGN (upgraded 2026-08-01, user-directed quality-over-speed):
  5 mixes x 2 TRAINING REPLICATES (independent training seeds, same
  data - separates composition effects from training-run luck) x
  2 DISJOINT SEED BLOCKS of n=3200 paired matches each (separates real
  effects from seed-set idiosyncrasy) = 20 evaluations, all vs the
  same baseline on identical per-block deal seeds (CRN).
- ANALYSIS (pre-specified): per-mix delta = mean over its 4 evals with
  a hierarchical SE; variance decomposition reported (between-mix /
  between-replicate / between-block); familywise error over the 5
  mix-vs-baseline tests controlled by MAX-T PERMUTATION on seed-level
  paired deltas (exact under CRN), Bonferroni alpha=.01/mix as the
  transparent backup. Replicate disagreement exceeding between-mix
  spread is itself a REPORTED finding (training noise dominates
  composition - caps what any recipe comparison of this type can show).
- CONFIRMATION: the winning mix is retrained as a FRESH third replicate
  (new training seed) and ONLY that candidate faces the fresh-seed
  battery - the battery tests the MIX, never a lucky training run.
- No other candidate is ever gated.

## Recording & interpretation plan (pre-specified 2026-08-01, BEFORE any
## mix is evaluated - amendments after unblinding are not permitted)

### Artifacts (every one produced, pass or fail)
1. Per mix: a COMPOSITION MANIFEST (counts per family, selection seed,
   sha of the record-index list), the distill config + holdout
   diagnostics, and the raw n=3200 eval CSV with its seed.
2. A verdict JSON per mix (equity_data/verdicts/expert_iter_v2_<mix>.json)
   with delta/SE/CI vs baseline - machine-readable accumulation, the
   project convention.
3. docs/expert_iter_v2_results.md: the full comparison table (all mixes,
   deltas vs baseline with 95% CIs, all pairwise seed-paired contrasts
   with CIs), written BEFORE any narrative interpretation.
4. Ledger entry + memory update + (if a direction closes) an
   experiment_rules.md closed-directions entry.

### What each possible result is ALLOWED to mean
- "Mix X beat baseline" may be CLAIMED only from the fresh-seed
  confirmation battery. The 5-mix comparative runs yield ESTIMATES with
  CIs; with 5 looks at alpha=.05, ~23% chance one "significant" delta
  is luck - the comparative stage therefore selects, never concludes.
- Pairwise mix differences: report seed-paired contrasts with CIs.
  PRE-COMMITTED indistinguishability line: contrasts smaller than
  2x their paired SE are reported as "indistinguishable at this n" -
  no ranking narrative may be built on them.
- A null across ALL mixes (no CI excluding 0 in the helpful direction):
  closes equity-argmax expert iteration per the stop rule. It does NOT
  license claims about visit-count targets, other filters, other anchor
  weights, or other teachers - those were never tested.
- "No mix beat baseline" is NOT "no mix is better": the per-mix MDE vs
  baseline is ~0.028 placement; the results doc must state the CI, so a
  true-but-small effect is recorded as bounded, not erased.
- All telemetry splits (moon rates, family-conditional behavior, deal
  lengths) are EXPLORATORY, labeled as such, and generate hypotheses
  only - never conclusions.
- External validity: all results are conditional on THIS baseline
  (8a89da90 lineage), THIS anchor field (v3-m7 + v4-m10), and match
  play to 100. No claim generalizes past those without new measurement.

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

## AMENDMENT 2026-08-02 (user-directed, registered BEFORE any mix bank
## was built or evaluated): continuous-certainty exploratory arms

Three additional comparative-stage arms testing the CONTINUOUS-certainty
recipe - certainty as a per-record loss weight over ALL play-phase
records, instead of the binary confident/anchor split.

RECIPE (frozen, no new tunables):
- Per play-phase record: z = (eq_best - eq_second) / gap_se (z=0 when
  the stats are absent/invalid); weight w = erf(z/2), i.e. the
  probability the observed preference DIRECTION is real, rescaled to
  exceed chance (w=0 at z=0; ~0.84 at z=2; ~1 at z=4).
- Loss = w x CE(teacher's chosen action) + lambda x (1-w) x
  KL(candidate || baseline), both terms weighted-averaged in-batch;
  lambda = the SAME anchor coefficient frozen at the binary recipe's
  holdout freeze (no separate tuning). The binary recipe is this
  recipe's step-function limit.
- Value/belief losses train on ALL play records in these arms (a
  stated difference from the binary arms, where they see only
  confident records).

ARMS (natural-family data only - isolates the enrichment/size axes
from the seeded-composition axis; enrichment = confident fraction of
the bank; natural play produces ~10%):
  f_ct_nat50   50k records, natural enrichment rate (~10%)
  g_ct_2x50    50k records, 2x natural enrichment (~20%)
  h_ct_nat100  100k records, natural enrichment rate
Each: 2 training replicates x 2 seed blocks n=3,200, same CRN blocks
as the five mix arms; max-T familywise control extends over all 8 arms.

REGISTERED CONTRASTS (the only interpretable ones; everything else
across axes is exploratory):
- ENRICHMENT (f vs g): the direct mechanism test of the v1-failure
  hypothesis. Under continuous weighting, low-certainty records exert
  ~zero CE pull, so enrichment should NOT matter. Flat dose-response =
  the weighting neutralizes noise as designed; g materially better
  than f = weighting insufficient, the noise channel is still open.
- SIZE (f vs h): is data volume a lever for this recipe family?
  (Total and confident count co-vary; practical question, not clean
  attribution.)
- RECIPE BRIDGE (d_natonly vs f/g/h): the only same-composition
  binary-vs-continuous comparison. Continuous-vs-SEEDED-mix
  comparisons confound recipe with composition: exploratory only.

CANDIDACY (binding): these three arms are NOT candidate-eligible.
Whatever their comparative results, the confirmation replicate and the
battery draw only from the five mix arms (a)-(e). Rationale: the arms
probe a different region of design space (low enrichment,
natural-only) as mechanism probes; promoting one would swap the
experiment's question mid-flight.

LIFELINE BRANCH (pre-registered now, exercisable only as written):
IF the binary battery FAILS and at least one continuous arm's
comparative delta is helpful with familywise-adjusted significance
(max-T p < .05 over all 8 arms), the closure entry records that
exception and ONE follow-up pre-registration for the continuous
recipe is permitted (its own one-shot battery, fresh data allowed).
In every other case - including "continuous looked promising but not
familywise-significant" - the stop rule's closure covers the
continuous variant too, and no equity-scored-target recipe of any
kind may be revisited without evidence admissible under
experiment_rules.md's closed-directions preamble.
