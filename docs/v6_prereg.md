# v6 pre-registration — capacity + structure

Status: SIGNED — user sign-off 2026-08-11 evening ("I sign off,
start"). Binding as written; Stage 2 venue reserved to quote time.
Baseline: 8a89da90 (5th match-era promotion, unchanged since 07-29).
House rules bind throughout (docs/experiment_rules.md; halt-default,
telemetry informs never gates, one pre-unblinding amendment per stage,
durations quoted from docs/speed_ledger.md, no cloud spend without
per-rental approval).

## Why this, why now (the converging evidence)

1. Every search amplifier plateaus at the same ceiling (K-scaling,
   ISMCTS, leaf evals) — the network is the binding constraint.
2. Both encodings of the searched teacher's knowledge are measured
   non-improving at 7.6M: equity-ordering targets (expert-iter v2,
   08-05) and visit-count targets (Phase 2, 08-11, dose-response).
3. League r1 showed defense is TEACHABLE but could not be contained in
   a mature 7.6M net (guard +0.453) — capacity-contention signature.
4. The moon hole is an information problem as much as a training one:
   the observation never says who CAPTURED cards or who won tricks
   (winner identity requires a 13-step recursion the obs does not
   seed), and seats have no representation at all in v5's token set.
5. v5-L closed "scale from the stale bank" (data-bound), NOT "scale
   inside a fresh loop" — that is Phase 3 of the roadmap, never run.

Claim under test: structure (capture information + seat entities +
threat-shaping auxiliary heads) is the lever, scale is the multiplier,
and together they reopen a compounding distill→PPO loop that 7.6M
could not host. Gates decide, as always.

## Stage 0 — observation v2 (instrument change, halt-default)

All additions are PUBLIC information (a human at the table tracks all
of it); information-honesty is unchanged. Appended past dim 556 so
every existing consumer is untouched (the match_proj pattern):

- [556..607]  per-card within-trick position (0 unseen; else k/4)
- [608..659]  per-card led-the-trick flag
- [660..867]  per-card taken-by, 4×52 planes in RELATIVE seats
              (the trick winner captures all four cards)
- [868..871]  tricks won per relative seat (/13)
- [872..875]  moon-alive flag per relative seat (that seat has taken
              every point so far, or no points taken yet)
- [876]       hearts still unseen (/13)
- [877..881]  Q♠ status one-hot (unseen / taken by rel seat 0-3)

Total obs dim 556 → 882. Env emits both layouts (obs v1 = prefix, by
construction). Verification before anything trains (halt-default):
- A/A determinism of the extended observe on fixed seeds.
- Validator invariants on recorded games: taken-by sums reproduce
  round_scores exactly; winner-recursion cross-check on full deals;
  moon-alive consistent with round_scores.
- Extended-v5 identity: a v5 checkpoint loaded with zero-init
  projections over the new dims is BIT-IDENTICAL on 550/556/882-dim
  inputs (the verified match_proj pattern). Traces unchanged.

## Stage 1 — HeartsNetV6 (structure spec + null contracts)

Token set 53 → 57: 1 global + 4 SEAT tokens + 52 card tokens.
- Card tokens: v5's channels EXTENDED with the new per-card slices
  (position, led, taken-by×4) → N_CARD_CH 10 → 16.
- Seat tokens (new, relative seats): per-seat public state — deal
  points taken, tricks won, match score + leader distance, void flags
  (4), moon-alive, passed-to/received-from indicators, on-lead-now.
- Global token: v5's 30 ctx dims + match ctx, unchanged.
- Heads: policy per card token (unchanged surface); value on global;
  belief per card token (kept); NEW aux heads (training-only, the
  belief-head mechanism aimed at the measured hole):
  - moon head: 4 logits per relative seat, label = that seat shot the
    moon this deal (free in every self-play record);
  - per-seat deal-points head: 4 scalars, label = final round_scores.
  - ORACLE HEAD DELETED (measured uninformative since v5).
- Size: d_model=448, L=8, heads=8 (~2.6× v5-M params). ONE size this
  prereg; no size sweep.
- Deployment surface unchanged: forward(obs, mask) → (logits, value);
  tracing/probe contracts as before (ProbeObsDim order rules apply).
- Null contracts before data: --epochs 0 identity; zero-init aux heads
  leave policy/value bit-identical; selftest green.

## Stage 2 — fresh bank with defense pressure (sized by probe)

- Teacher: the deployed flat searcher at standard config (rules #15
  and #16 package), current baseline weights, every decision recorded.
  Record format v3: obs[882], mask, chosen action, seat, match id,
  per-deal flush + trailer-bounded resume (r2 contract), plus
  per-deal outcome labels (final round_scores, mooned-by) for the aux
  heads. Belief labels recorded (Phase 2's policy-only deviation is
  NOT repeated).
- DEFENSE PRESSURE registered into the distribution: 1/8 of matches
  seat ONE certified r1 shooter clone (shooter_agg_v1b / shooter_sel_v1
  alternating) against 3 teacher-searcher seats; shooter seat's
  decisions are EXCLUDED from training records (we imitate the
  defenders, not the attacker). Remaining 7/8 natural self-play.
- Sizing: target ≥ 1.5M play-decision records (~25k deals). Pace probe
  first (20 deals), quote from the measured rate before launch
  (launcher discipline). Ledger reference 6.32 s/deal local steady →
  order ~44h full-throttle local; the certified H100+AOTI path
  (2.22 s/deal) may be proposed — separate per-rental approval.
- Seed block: 220,000,000+ (stride 1M), audited disjoint at launch
  against all used blocks (20-179M, 190M, 200M, 210M, 212-213M,
  520/620M, 720-722M).

## Stage 3 — from-scratch distillation, 3 registered arms

From-scratch CE on the search-chosen action (the recipe that BUILT
v5-M; the closures cover warm-start refresh of an RL-sharpened
champion and soft equity/visit targets — neither is this) + value on
match-outcome targets + belief + aux heads.

Arms (attribution by construction):
- (a) v6-full: obs v2, seat tokens, aux heads, d448 L8.
- (b) structure control: v5-M architecture (d320 L6) + obs-v2 card
  channels + aux heads — isolates scale (a vs b).
- (c) data control: v5-M architecture, obs v1, no aux — isolates
  everything (c vs today's lineage answers "was fresh data alone
  enough?").
Recipe freeze on holdout ONLY (by-match split): epochs {2,3,4} ×
lr {1e-4, 3e-4} per arm, ≤ 6 trainings/arm; entropy diagnostic band =
2× both directions of the baseline milestone (0.434 → [0.22, 0.87]),
one pre-unblinding re-band amendment available (the Stage-D lesson:
anchor to the reference that the recipe should converge to — argue it
BEFORE unblinding if the guess is wrong).

Stage-3 screen (not a gate): best pick per arm gets neutral-raw eval
(n=2500). Registered expectation: from-scratch imitation does NOT
reach the RL-sharpened baseline (v5 lesson). BAND: proceed to Stage 4
if v6-full's neutral-raw UB ≤ +1.5/deal vs baseline; halt-default
below that (a from-scratch net further behind than that has no
realistic PPO runway — v5-M's own scratch-distill landed AHEAD of its
era's baseline).

## Stage 4 — match-PPO ladder (where the gain lives)

run_loop match-mode on the v6-full pick (arm (a) only; (b)/(c) are
measurement controls, not candidates). Standard discipline: headroom
pacing, autostop, explicit-snapshot gating, optimizer carry-through.
Trials until EITHER a gate pass (Stage 5) OR 3 consecutive structural
nulls (pooled |Δ| < 0.02 placement) → HALT and report. Defense
telemetry recorded EVERY trial (moons conceded/match vs the fixed
baseline arm, r1 harness) — informs, never gates.

## Stage 5 — gates (standard battery) + registered defense outcome

Per candidate (≤ 2 picks from the ladder): match gate n=3200 α=0.05
placement superiority vs 8a89da90 + search guard n=4800 K=32 one-sided
95% UB ≤ +0.3. Unchanged, conjunctive, halt-default.

SECONDARY registered outcome (reported, never gating): moons
conceded/match ratio vs baseline defenders on the gate telemetry. The
v6 claim about moon defense is falsifiable here: structure+pressure
should show ≤ 0.75× baseline concessions. If a candidate PROMOTES and
the defense ratio lands ≤ 0.75, league round 4 (validation vs live
shooters) unlocks behind its own prereg; if it promotes with defense
unmoved, the moon hole is declared architecture-resistant at this
scale and the league resumes as the only open path.

PASS additionally triggers the roadmap's compounding question: one
regeneration + redistill cycle with the promoted net (own prereg
rider) — a second-generation pass is the Phase-3 "loop works"
declaration.

## Costs (rule 7: measured where a ledger entry exists)

Stage 0-1: session work, no GPU spend. Stage 2: pace probe ~1h; bank
~44h local full-throttle (rule 17 windows) OR ~15h/$12-15 H100 —
venue decision at quote time. Stage 3: trainings minutes-to-~1h each
(≤18 total), neutral-raw screens ~2.5 min each. Stage 4: 2-2.75h per
fail-fast trial. Stage 5: ~1.5h + ~2.7h per candidate. No cloud spend
without a specific approved rental.

## What results are allowed to mean

- Stage-0/1 verification failures: instrument bugs; fix and re-verify
  (no scientific claim spent).
- Stage-3 screen halt: from-scratch v6 cannot get close enough for
  PPO; capacity/structure at this size does not even reproduce the
  v5-M scratch result — the v6 design returns to the drawing board
  WITHOUT burning the gates.
- (a) vs (b) separation on holdout + screen: scale's isolated
  contribution, measured (the datum the v5-L closure could not give).
- (c) near today's lineage: fresh data alone was not the story.
- Stage-4 structural nulls: v6 does not reopen the PPO regime — the
  capacity hypothesis fails at 2.6×, and the honest next conversation
  is signal sources (league round 4), not more scale.
- Stage-5 PASS: 6th promotion, new model era (site leaderboards
  archive per the era system), compounding rider unlocks.
- Defense ratio ≤ 0.75 with promotion: the moon hole finally moved;
  round-4 validation prereg unlocks.

## Signature

SIGNED by user 2026-08-11 ("I sign off, start"). Scope: stages 0-5 as
written; venue decision for Stage 2 reserved to quote time; any
band change after data = the registered one-amendment path only.

## Stage-3 amendment (the registered one; user-approved 2026-08-12, BEFORE any screen ran)

Arm (c) produced no snapshot inside the entropy band [0.22, 0.87]
(best 0.877, range to 0.955 - the obs-v1/no-aux control imitates the
teacher more diffusely at every recipe). Arms (a) and (b) froze
in-band, so the band itself stands for the candidates. AMENDMENT:
for arm (c) ONLY, the entropy eligibility is waived and its
lowest-holdout-CE snapshot is frozen (lr 3e-4, epoch 2, CE 0.8681,
entropy 0.891 - flagged out-of-band). Rationale: (c) is a measurement
control, not a promotion candidate; discarding its screen datum wastes
the control while the flag preserves honesty. The amendment budget for
Stage 3 is now SPENT. Holdout freeze table recorded in
v6_stage3/arm*_lr*.json; screens run at the registered n=2500
(measured instrument SE 0.136-0.144/deal - the +1.5 band is >6 SE wide,
so deeper n buys nothing the decision uses).
