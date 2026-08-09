# METHODOLOGY — the measurement discipline

> STATUS: DRAFT - not yet released; project ongoing.

Derived from docs/experiment_rules.md (canonical; every rule there
carries the datestamp of the incident that taught it), the ledger
(docs/speed_ledger.md), and the pre-registration documents. Nothing
below is aspirational — each practice exists because its absence
produced, or nearly produced, a false conclusion.

## 1. Paired deals and common random numbers

Every strength comparison plays candidate and baseline through the
SAME deals (same shuffle seeds, same seats) and analyzes per-deal or
per-match paired deltas. Match-level comparisons extend the pairing to
deal-sequence seeds; search-vs-search comparisons additionally share
determinization sets and rollout seeds where the action sets agree
(CRN; docs/match_aware_search_design.md, validation pre-registration).

The pairing is verified, not assumed: null calibrations (an arm against
itself) must produce exactly-zero paired deltas — the match gate's
first null calibration gave all-zero deltas over 24 matches (ledger
2026-07-23), and the N=8000 fleet's A/A pilot gave 20/20 bit-identical
pairs across arms (ledger 2026-07-26). CRN quality is reported: the
N=8000 run's realized paired placement SD was 1.351 vs the bridge
run's 1.35 (ledger 2026-07-27).

## 2. Neutral anchors — and the flipped sign that taught the rule

Rule #1: never evaluate against a net's own policy or its training
opponent — that hands one side a perfect opponent model. Measured
consequences: a +1.24 "regression" flipped to parity once the
comparison moved to neutral opponents (2026-07-15 incident, rule #1),
and the head-to-head raw guard understated one candidate's true
neutral-opponent gain by 3x (docs/ppo_v5_round2_findings.md,
diagnostic A). All strength comparisons therefore seat both arms
against fixed neutral anchor fields (v3-m7, later alternated with
v4-m10 to retire anchor-overfit risk — ledger 2026-07-28).

## 3. SE and n with every verdict

Every reported delta carries its standard error and sample size
(rule #2). Durations are quoted only from the speed ledger, same
generation and same convention — A/B 50-deal versus steady long-run
figures are never mixed, because dividing one by the other overstates
gains (ledger header; the 2.11x non-figure is called out there).

## 4. Pre-registration and halt-is-default

From the SDPA gate on (docs/sdpa_gate_preregistration.md, registered
while the deciding run was in progress and unobserved), consequential
measurements fix the criterion AND the action on each outcome branch
before results exist. The match-aware search program went further
(docs/match_aware_search_design.md): nine numbered design reviews, a
gate spine where **HALT is the default**, machine-readable verdict
JSONs per gate (equity_data/verdicts/*.json: gate, metrics, thresholds,
pass, git sha, data sha), and a single pre-registered analysis at
N=8000 — no interim looks, no alpha spending.

The halts held when triggered. The flip/SNR spine gate HALTED
(verdicts/flip_snr.json, pass=false) even though one sub-criterion
passed hugely, because the SNR criterion failed — and the built-in
reference comparison (the same ratio computed for the known-working
deal-point scorer) revealed the threshold itself was mis-set. Options
went to the user; nothing proceeded unilaterally (ledger 2026-07-25).

Amendments have a shape too: pre-data, in writing, user-approved, and
recorded in the prereg itself. The expert-iter v2 recipe freeze is the
canonical example (2026-08-04): both pre-registered anchor
coefficients failed an entropy diagnostic, exploration to pick a
replacement ran on HOLDOUT ONLY — before any gate data existed — and
the finding (a monotone dose-response) plus the frozen lambda=4.0 went
into the prereg as a signed amendment before the comparative stage
launched (docs/expert_iter_v2_freeze_report.md). Selection stayed on
one side of the blind; the gate stayed on the other. The same pattern
carried era 9: the round-2 attacker check was re-registered on
measurable quantities before generation analysis, and the
pausable-generation requirement was an amendment at approval time
(docs/exploiter_league_r2_prereg.md).

## 5. Gate power, and the two re-powering campaigns

An underpowered gate is worse than none: it converts real effects into
coin flips and makes "continual failure" uninterpretable.

**Episode 1 — search gate, n=600 -> 2,400 (2026-07-19).** The n=600
gate had ~25% power against a true -0.3 pts/deal. At n=2,400 the
promotion bar moved from -0.51 to ~-0.26 at alpha=0.05, and the first
full-power verdict turned an ambiguous n=600 FAIL into "+0.679
(SE 0.169, t=4.0): definitively worse" (ledger 2026-07-19).

**Episode 2 — match era (2026-07-28), both instruments at once:**
- Match gate n=800 -> 3,200: four trials had pooled to -0.027 ± 0.017
  (p~.06) — a real sub-bar effect the n=800 gate was coin-flipping
  (43% power vs a true -0.05; 90% at n=3,200, SE 0.017, bar -0.028).
  The very next passing trial (-0.029, p=0.0456) is one n=800 would
  have coin-flipped (ledger 2026-07-28).
- Search guard n=2,400 -> 4,800: at n=2,400 a DEAD-NEUTRAL candidate
  passed the +0.3-UB guard only ~61% of the time (false-veto risk);
  at n=4,800, ~86%. The margin was left unchanged — the noise, not the
  tolerance, was the problem. Vindicated within a day: promotion #5's
  guard UB was +0.258 at n=4,800 where the old n would have produced
  ~+0.335, a false veto (ledger 2026-07-28/29).

The shared lesson, stated in the ledger both times: don't half-power.

## 6. Selection is not confirmation

Anything chosen by looking at results is a SELECTION and earns only an
estimate; claims come from a fresh confirmation under a pre-registered
gate. The equity-model component choice is an explicit selection stage
(verdicts/selection.json — "selection", not pass/fail); the expert-iter
v2 design separates a 5-mix comparative stage ("selects, never
concludes" — with 5 looks at alpha=.05, ~23% chance one significant
delta is luck) from a one-shot fresh-seed battery on a freshly
retrained replicate, with a pre-committed indistinguishability line
and a binding stop rule (docs/expert_iter_v2_prereg.md).

The same separation appears in the gate history: hypotheses were probed
cheaply (n=2,500 quick gates, probes, entropy tie-breakers), but every
promotion passed the full pre-registered battery, one shot.

## 7. Telemetry informs, never gates

Every candidate gets a behavioral diff vs baseline (analyzer.py
lineage; history in analyzer_history.csv); large deltas are flagged,
never gated on. Justification is empirical in both directions: the
2026-07-23 comparison showed the strongest model deliberately scoring
WORSE on several "textbook" tactical rates — gating on them would have
Goodharted play style — while the same telemetry found the 2x
moon-concession hole that a week of scalar gates had missed (rule #5;
analyzer_history.csv 2026-07-28 row: moon defense 51.5% vs v4-m10's
61.1%, moons conceded 272 vs 151 per 2,000 deals).

## 8. The closed-directions register

Negative results are recorded as CLOSED directions with the evidence
that closed them, and are not revisited without new evidence
(docs/experiment_rules.md, closed directions): PPO at the ceiling
(one-shot; 255k further games flat), same-lineage distillation (fails
at 3.5k and 12.5k fresh deals; sharpen saturates at 2.0), imitation-only
scale-up (v5-L +3.3..+5.3 on every data mix), search amplification
past the net's ceiling (K>64, ISMCTS, learned leaves), and — era 8,
completed in two rounds — imitation of the equity-scored teacher: v1
closed whole-distribution targets; v2 (5 binary mixes worse, 3
continuous-certainty arms null, 8 arms total) extended the closure to
ANY equity-scored-target recipe and refuted the noise mechanism — the
ordering signal itself is inert for match play. Outside the closure,
recorded explicitly: teachers with a different signal source (visit
counts, exploiter-league demonstrations — era 9 uses the latter). The
register
is why era 5's pivot was a strategy change rather than a bigger budget:
the map of what does NOT work is treated as an asset
(docs/ROADMAP.md, asset inventory).

**"New evidence" is defined, not vibes-based** (canonical wording in
docs/experiment_rules.md, closed-directions preamble): reopening a
closure requires naming which recorded premise no longer holds —
a change to a component the closure's mechanism depends on, a
variation the closure itself recorded as untested, or a diagnostic
contradicting its mechanism. More compute/data/seeds or a rerun of
the same recipe is inadmissible by default (closures are powered
measurements; a re-roll only exploits variance — though "more data"
qualifies exactly when the closure recorded quantity as the untested
variable). The register's one reopening to date followed this shape:
PPO, closed on MLPs with the mechanism "critic cannot learn value on
these features," was re-tried only when the v5 architecture removed
that premise — and produced the 2026-07-15 promotion. Every reopening
gets its own pre-registration.

## 9. Ops discipline that protects measurements

Measurement dies of ops failures as readily as of statistics, so the
rules file covers both (rules #8–#14, #17): unbuffered file-logged
launches via dedicated scripts (the `&&`-chain backgrounding bug was
violated and re-learned three times); `__main__` guards on
multiprocessing drivers (a scratchpad driver crash-respawned workers
for 2.3 h); watchdogs at ~2x the ledger estimate with grep patterns
that cannot match normal output; end-time-capped runs sized from the
verified clock; md5 verification of the baseline after any hard kill;
no cloud spend without explicit per-rental user approval (#13); and
**train in cloud, evaluate locally** (#14) — checkpoints may be
produced anywhere, but every comparison that decides anything runs on
one local machine with both arms sharing hardware.

A clarification for external readers: the discipline deliberately does
NOT chase cross-environment bit-determinism (determinism flags, pinned
kernels). Environment-immunity comes from pairing both arms on one
machine in one run — the comparison, not the environment, is what is
controlled. What determinism the system does and does not promise
(bit-identical replay within a hardware/OS class; bf16 argmax flips
across hardware) is measured and documented in
docs/release/REPRODUCING.md sec. 4.

Two diagnostic signatures worth exporting to any project: an SE of
exactly 0.000 means you gated a net against itself (rule #3); and a
holdout metric that looks too good means the split leaked (per-record
sibling leakage, then per-file-tail late-trick bias — rule #4 has both
incidents).

Cross-references: how these rules arose in time —
docs/release/JOURNEY.md; the numbers they produced —
docs/release/RESULTS.md; project-specific terms (CRN, gate vs guard,
deal-point vs equity scoring, ...) — the glossary in
docs/release/INDEX.md.
